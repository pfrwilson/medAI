from medAI.datasets import ExactNCT2013BModeImages, CohortSelectionOptions
import torch 
from simple_parsing import parse 
from dataclasses import dataclass
from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
import wandb
from torchvision.models.resnet import ResNet18_Weights
ResNet18_Weights


@dataclass(frozen=True)
class Config: 
    batch_size: int = 4
    debug: bool = False
    fold: int = 0
    epochs: int = 30
    n_folds: int = 5
    benign_to_cancer_ratio: float | None = None


def main(config): 
    train_loader, val_loader, test_loader = get_dataloaders(config)

    from segment_anything import sam_model_registry
    sam_model = sam_model_registry["vit_b"](
        checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
    )
    image_encoder = sam_model.image_encoder
    pool = torch.nn.AdaptiveMaxPool2d((1, 1))
    flatten = torch.nn.Flatten()
    fc = torch.nn.Linear(256, 1)
    model = torch.nn.Sequential(image_encoder, pool, flatten, fc)
    model = model.cuda()
    model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, 5, config.epochs, warmup_start_lr=1e-9, eta_min=1e-7)
    amp_model = torch.cuda.amp.autocast()(model)
    scaler = torch.cuda.amp.GradScaler()
    loss = torch.nn.BCEWithLogitsLoss()

    best_score = 0 

    wandb.init(project="medsam_cancer_detection_corewise_simple")

    for epoch in range(config.epochs):
        model.train()

        pred_list = []
        label_list = []
        involvement_list = []

        for i, (image, involvement, label) in enumerate(tqdm(train_loader, desc='training')): 
  
            image = image.cuda()
            involvement = involvement.cuda()
            label = label.cuda()

            optimizer.zero_grad()
        
            pred = amp_model(image)
            pred = pred.squeeze(-1)

            loss_val = loss(pred, label.float())
            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()
            
            wandb.log({
                "loss": loss_val.item(),
                "lr": scheduler.get_last_lr()[0],
            })

            pred_list.append(pred.sigmoid().detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            involvement_list.append(involvement.detach().cpu().numpy())

        metrics = compute_metrics(pred_list, label_list, involvement_list)
        metrics = {f"train/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)

        pred_list = []
        label_list = []
        involvement_list = []

        model.eval()
        with torch.no_grad():
            for i, (image, involvement, label) in enumerate(tqdm(val_loader, desc='validation')): 
                if config.debug and i > 10:
                    break

                image = image.cuda()
                involvement = involvement.cuda()
                label = label.cuda()

                pred = amp_model(image)
                pred = pred.squeeze(-1)

                pred_list.append(pred.sigmoid().detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())
                involvement_list.append(involvement.detach().cpu().numpy())

        metrics = compute_metrics(pred_list, label_list, involvement_list)
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)
        if metrics["val/auc_high_involvement"] > best_score:
            best_score = metrics["val/auc_high_involvement"]
            
            pred_list = []
            label_list = []
            involvement_list = []

            with torch.no_grad():
                for i, (image, involvement, label) in enumerate(tqdm(test_loader, desc='testing')):
                    if config.debug and i > 10:
                        break
                    image = image.cuda()
                    involvement = involvement.cuda()
                    label = label.cuda()

                    pred = amp_model(image)
                    pred = pred.squeeze()

                    pred_list.append(pred.sigmoid().detach().cpu().numpy())
                    label_list.append(label.detach().cpu().numpy())
                    involvement_list.append(involvement.detach().cpu().numpy())

                metrics = compute_metrics(pred_list, label_list, involvement_list)
                metrics = {f"test/{k}": v for k, v in metrics.items()}
                wandb.log(metrics)

        scheduler.step()    
    

def compute_metrics(pred_list, label_list, involvement_list):
    metrics = {}
    import numpy as np
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    involvement_list = np.concatenate(involvement_list)

    from sklearn.metrics import roc_auc_score
    try: 
        auc = roc_auc_score(label_list, pred_list)
    except ValueError:
        auc = np.nan
    metrics["auc"] = auc

    high_involvement = (involvement_list > 0.4) | (involvement_list == 0)
    pred_list = pred_list[high_involvement]
    label_list = label_list[high_involvement]

    try: 
        auc = roc_auc_score(label_list, pred_list)
    except ValueError:
        auc = np.nan
    metrics["auc_high_involvement"] = auc

    return metrics


def get_dataloaders(config):
    
    class Transform:
        def __init__(self, augment=False):
            self.augment = augment
        def __call__(self, item): 
            image = item['bmode'] 
            image = torch.from_numpy(image.copy()).float()
            image = image.unsqueeze(0)
            image = (image - image.min()) / (image.max() - image.min())
            from torchvision import transforms as T 
            image = T.Resize((1024, 1024), antialias=True)(image)
            if self.augment:
                image = T.RandomResizedCrop(1024, scale=(0.8, 1.0), antialias=True)(image)
            image = image.repeat(3, 1, 1)

            involvement = torch.tensor(item['pct_cancer']).float() / 100.0
            if torch.isnan(involvement).item():
                involvement = torch.tensor(0.0).float()

            label = torch.tensor(item['grade'] != "Benign").float()
            return image, involvement, label
        
    train_ds = ExactNCT2013BModeImages(
        split='train',
        transform=Transform(augment=True),
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=40, 
            remove_benign_from_positive_patients=True, 
            benign_to_cancer_ratio=config.benign_to_cancer_ratio
        )
    )

    val_ds = ExactNCT2013BModeImages(
        split='val',
        transform=Transform(augment=False),
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=None, 
            remove_benign_from_positive_patients=False, 
            benign_to_cancer_ratio=None
        )
    )

    test_ds = ExactNCT2013BModeImages(
        split='test',
        transform=Transform(augment=False),
        cohort_selection_options=CohortSelectionOptions(
            fold=config.fold, n_folds=config.n_folds,
            min_involvement=None, 
            remove_benign_from_positive_patients=False, 
            benign_to_cancer_ratio=None
        )
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = parse(Config)
    main(config)
