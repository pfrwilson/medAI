from dataclasses import dataclass, field, asdict
from typing import Iterator
from simple_parsing import choice, parse, ArgumentParser
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch
from torch.nn.parameter import Parameter
from medAI.datasets import ExactNCT2013BModeImages, ExactNCT2013RFImagePatches, PatchOptions
from tqdm import tqdm
import sys
import os
import json
import logging
import wandb

from utils import create_epoch_report, basic_experiment_setup, basic_ddp_experiment_setup, DataFrameCollector, compute_optimal_threshold
from medAI.utils.sampler import StatefulSampler, GenericStatefulSampler, GenericDistributedStatefulSampler
from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from datetime import datetime
from time import time
import coolname
import matplotlib.pyplot as plt
import copy
import typing as tp
import numpy as np 
from torch import distributed as dist 


CHECKPOINT_TIME_S = 60 * 60 * 0.5  # 30 mins


class ModelRegistry:
    @staticmethod
    def resnet10():
        from trusnet.modeling.registry import resnet10_feature_extractor

        model = resnet10_feature_extractor()
        model.features_dim = 512
        return model

    @staticmethod
    def resnet18():
        from trusnet.modeling.registry import resnet18_feature_extractor
        model = resnet18_feature_extractor()
        model.features_dim = 512    
        return model

    @staticmethod
    def rfline_resnet_small():
        from trusnet.modeling.rfline_resnet import RFLineResNet

        model = RFLineResNet(layers=[1, 1, 1, 1], num_classes=2, in_channels=1)
        model.fc = nn.Identity()
        model.features_dim = 256
        return model

    @staticmethod
    def rfline_resnet():
        from trusnet.modeling.rfline_resnet import RFLineResNet

        model = RFLineResNet(layers=[2, 2, 2, 2], num_classes=2, in_channels=1)
        model.fc = nn.Identity()
        model.features_dim = 256
        return model


@dataclass
class JobConfig:
    log_dir: str = 'experiments'  # top level directory for all experiments
    exp_dir: str | None = None  # specific experiment directory (if none, generate one in the log_dir)
    submitit: bool = False  # whether to use submitit for training on the cluster
    kfold: bool = False  # whether to submit as many jobs as there are folds
    gpus: int = 2 
    qos: str = 'm2'


@dataclass
class TrainConfig:
    """Configuration for the experiment"""   
    group: str = "default"
    wandb_project: str = "trusnet2.0-ssl"
    imaging_mode: str = choice("bmode_undersampled", "rf", default="rf")
    ssl_patch_options: PatchOptions = field(
        default_factory=lambda: PatchOptions(shift_delta_mm=2.5, strides=(0.5, 0.5))
    )
    ssl_augmentations: tp.Literal['none', 'cv'] = 'none'
    scheduler: tp.Literal['cosine', 'plateau', 'linear_warmup_cosine'] = 'plateau'
    optimizer: tp.Literal['adam', 'sgd', 'lars'] = 'adam'
    batch_size: int = 64
    sl_patch_options: PatchOptions = field(default_factory=PatchOptions)
    input_size: tuple[int, int] | None = (256, 256) 
    num_epochs: int = 10 # total number of self-supervised learning epochs
    linear_evaluation_frequency: int = 5 # run linear evaluation every n epochs if None, don't do linear evaluation
    ssl_method: tp.Literal['vicreg', 'simclr', 'moco', 'swav'] = 'vicreg'
    debug: bool = False
    ssl: bool = True  # whether to do the self-supervised learning task
    model_name: str = choice(
        [k for k in vars(ModelRegistry).keys() if not k.startswith("_")],
        default="resnet10",
    )
    batch_grouping: tp.Literal['none', 'position_wise'] = 'none'
    undersample_benign_ssl: bool = True
    test_frequency: int | None = None  # test every n epochs
    lr: float = 1e-3 # relative to a batch size of 256
    wd: float = 0.0
    momentum: float = 0.9
    resume: bool = True


class Experiment:
    def __init__(self, config: TrainConfig, config_dict, exp_dir):
        self.config = config
        self.config_dict = config_dict  # we have to store this because we can't serialize the dataclass properly
        self.exp_dir = exp_dir

    def setup(self):
        basic_ddp_experiment_setup(
            self.exp_dir,
            self.config.group,
            self.config_dict,
            self.config.wandb_project,
            self.config.resume,
            self.config.debug
        )

        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        if (
            self.config.resume
            and self.ckpt_dir is not None
            and "experiment_state.pth" in os.listdir(self.ckpt_dir)
        ):
            self.state = torch.load(os.path.join(self.ckpt_dir, "experiment_state.pth"))
            logging.info(f'Resuming from epoch {self.state["epoch"]}')
        else:
            self.state = None

        if dist.get_rank() != 0: 
            # disable wandb 
            wandb.log = lambda x: None 
            from argparse import Namespace
            wandb.run = Namespace()
            wandb.run.summary = {}

        # main training loop
        logging.info("Setting up data...")
        self.setup_data()
        logging.info(f"Number of SSL training examples: {len(self.ssl_dataset)}")
        logging.info(f"Number of SSL training cores: {len(self.ssl_dataset._dataset.core_ids)}")
        if wandb.run is not None:
            wandb.run.summary["num_ssl_training_examples"] = len(self.ssl_dataset)
            wandb.run.summary["num_ssl_training_cores"] = len(self.ssl_dataset._dataset.core_ids)

        self.setup_model()
        self.setup_optimizer()
        

        self.epoch = self.state["epoch"] if self.state is not None else 0
        self.best_score = self.state["best_score"] if self.state is not None else -1e9

    def setup_model(self):
        logging.info(f"Creating model {self.config.model_name}")
        self.backbone = ModelRegistry.__dict__[self.config.model_name]()
        n_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logging.info(f"Number of trainable parameters: {n_params}")
        wandb.run.summary["num_trainable_params"] = n_params
        
        match self.config.ssl_method:
            case 'vicreg': 
                from medAI.modeling.vicreg import VICReg
                self.ssl_model = VICReg(
                    self.backbone, [self.backbone.features_dim] * 2, self.backbone.features_dim
                )
            case 'simclr': 
                from medAI.modeling.simclr import SimCLR
                self.ssl_model = SimCLR(
                    self.backbone, [self.backbone.features_dim] * 3
                )
            case 'moco': 
                from medAI.modeling.moco import MoCo
                self.ssl_model = MoCo(
                    self.backbone, self.backbone.features_dim, self.config.batch_size * 20, projector_dims=[self.backbone.features_dim] * 3
                )
            case 'swav': 
                from medAI.modeling.swav import SwAV
                d = self.backbone.features_dim
                self.ssl_model = SwAV(
                    self.backbone, projector_dims=[d, d//2, d//4], features_dim=d//4, n_stored_features=1024, n_prototypes=64,
                )
            case _: 
                raise ValueError(f"Unknown ssl method {self.config.ssl_method}")

        self.ssl_model.cuda()
        self.ssl_model = DistributedDataParallel(self.ssl_model)
        if self.state is not None:
            self.ssl_model.load_state_dict(self.state["ssl_model"])

    def setup_optimizer(self): 

        lr = self.config.lr
        batch_size = self.config.batch_size * dist.get_world_size()
        K = batch_size / 256
        adjusted_lr = lr * K

        match self.config.optimizer: 
            case 'adam': 
                self.ssl_optimizer = torch.optim.Adam(
                    self.ssl_model.parameters(), lr=adjusted_lr, weight_decay=self.config.wd
                )
            case 'sgd': 
                self.ssl_optimizer = torch.optim.SGD(
                    self.ssl_model.parameters(), lr=adjusted_lr, momentum=0.9
                )
            case 'lars':
                from medAI.utils.optimizer import LARS
                self.ssl_optimizer = LARS(
                    lr=adjusted_lr, weight_decay=self.config.wd, params=self.ssl_model.parameters()
                )
            case _: 
                raise ValueError(f'Unknown optimizer {self.config.optimizer}')

        match self.config.scheduler: 
            case 'plateau':
                self.ssl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.ssl_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=2,
                )
            case 'cosine':
                self.ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.ssl_optimizer,
                    T_max=self.config.num_epochs,
                )
            case 'linear_warmup_cosine':
                self.ssl_scheduler = LinearWarmupCosineAnnealingLR(
                    self.ssl_optimizer,
                    max_epochs=self.config.num_epochs,
                    warmup_start_lr=self.config.lr * 1e-2,
                    warmup_epochs=5,
                )
            case _:
                raise ValueError(f"Unknown scheduler {self.config.scheduler}")

        if self.state is not None:
            self.ssl_optimizer.load_state_dict(self.state["ssl_optimizer"])
            if self.ssl_scheduler is not None:
                self.ssl_scheduler.load_state_dict(self.state["ssl_scheduler"])

    def commit_state(self):
        if dist.get_rank() == 0: 
            state = {}
            state["epoch"] = self.epoch
            state["best_score"] = self.best_score
            state["stage"] = self.stage
            state["rng"] = torch.get_rng_state()
            state["cuda_rng"] = torch.cuda.get_rng_state()
            state["ssl_optimizer"] = self.ssl_optimizer.state_dict()
            if self.ssl_scheduler is not None:
                state["ssl_scheduler"] = self.ssl_scheduler.state_dict()
            state["ssl_loader_sampler"] = self.ssl_loader.sampler.state_dict(
                self.ssl_loader_iter
            )
            state["ssl_model"] = self.ssl_model.state_dict()
            torch.save(state, os.path.join(self.ckpt_dir, "tmp.pth"))
            os.rename(
                os.path.join(self.ckpt_dir, "tmp.pth"),
                os.path.join(self.ckpt_dir, "experiment_state.pth"),
            )
        
        dist.barrier()

    def setup_data(self):
        train_cores, val_cores, test_cores = generate_splits(self.config.splits)
        from copy import deepcopy

        ssl_splits_config = deepcopy(self.config.splits)
        if not self.config.undersample_benign_ssl:
            ssl_splits_config.benign_to_cancer_ratio = None
        ssl_cores = generate_splits(ssl_splits_config)[0]

        if self.config.debug:
            import random

            ssl_cores = random.sample(ssl_cores, 10)
            train_cores = random.sample(train_cores, 10)
            val_cores = random.sample(val_cores, 100)
            test_cores = random.sample(test_cores, 100)

        class ZipDataset(torch.utils.data.Dataset):
            def __init__(self, *datasets):
                self.datasets = datasets

            def __getitem__(self, idx):
                return tuple(d[idx] for d in self.datasets)

            def __len__(self):
                return len(self.datasets[0])

        dataset_cls = ExactNCT2013RFImagePatches if self.config.imaging_mode == "rf" else ExactNCT2013BModeImages
    
        self.sl_dataset_train = PatchesDatasetV3(
            train_cores,
            mode=self.config.imaging_mode,
            transform=self.patch_transform,
            target_transform=self.label_transform,
            patch_options=self.config.sl_patch_options,
        )
        self.sl_dataset_val = PatchesDatasetV3(
            val_cores,
            mode=self.config.imaging_mode,
            transform=self.patch_transform,
            target_transform=self.label_transform,
            patch_options=self.config.sl_patch_options,
        )
        self.sl_dataset_test = PatchesDatasetV3(
            test_cores,
            mode=self.config.imaging_mode,
            transform=self.patch_transform,
            target_transform=self.label_transform,
            patch_options=self.config.sl_patch_options,
        )
        
        if self.config.batch_grouping == 'none': 
            def generate_indices(): 
                indices = list(range(len(self.ssl_dataset)))
                np.random.shuffle(indices)
                return indices
        elif self.config.batch_grouping == 'position_wise': 
            generate_indices = lambda: list(self.ssl_dataset.position_wise_indices(switch_position_every_n_samples=self.config.batch_size))
        else: 
            raise ValueError(f"Unknown batch grouping {self.config.batch_grouping}")
        sampler = GenericDistributedStatefulSampler(generate_indices, len(self.ssl_dataset))

        # self.ssl_loader = self._create_loader(ssl_dataset)
        self.ssl_loader = torch.utils.data.DataLoader(
            self.ssl_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
        if self.state is not None and "ssl_loader_sampler" in self.state:
            self.ssl_loader.sampler.load_state_dict(self.state["ssl_loader_sampler"])

    def patch_transform(self, patch):
        if self.config.input_size is not None:
            import cv2

            patch = cv2.resize(patch, self.config.input_size)
        patch = (patch - patch.mean()) / patch.std()
        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0)

        if self.config.ssl_augmentations == 'cv': 
            from src.transform import ComputerVisionAugmentations

            patch = ComputerVisionAugmentations()(patch)

        return patch

    def label_transform(self, label):
        return torch.tensor(label).long()

    def ssl_train_epoch(self):
        logging.info(f"\n=========================\nSTARTING EPOCH {self.epoch}")
        logging.info(f"Starting SSL training epoch {self.epoch}")
        self.stage = "ssl_train"
        self.ssl_model.train()
        total_loss = 0

        len_loader = len(self.ssl_loader)
        start_iter = self.ssl_loader.sampler.data_counter
        logging.info(f"Starting at iteration {start_iter}")
        self.ssl_loader_iter = iter(self.ssl_loader)
        self.commit_state()
        start_time = time()
        for batch in tqdm(
            self.ssl_loader_iter, total=len_loader, desc="Self-supervised Training"
        ):
            patch = batch["patch"]
            X1, X2 = patch
            X1 = X1.cuda()
            X2 = X2.cuda()
            loss = self.ssl_model(X1, X2)
            self.ssl_optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            wandb.log({"loss": loss.item()})
            self.ssl_optimizer.step()

            if time() - start_time > CHECKPOINT_TIME_S:
                logging.info(
                    f"Committing state at epoch {self.epoch}, iteration {self.ssl_loader.sampler.data_counter}"
                )
                self.commit_state()
                start_time = time()

        if self.ssl_scheduler is not None:
            if self.config.scheduler == 'plateau':
                self.ssl_scheduler.step(total_loss)
            else:
                self.ssl_scheduler.step()

        wandb.log({"lr": self.ssl_optimizer.param_groups[0]["lr"], "epoch": self.epoch})

        if hasattr(self.ssl_model.module, 'epoch_end'):
            # logging.info(f"Running epoch end function of ssl model {self.ssl_model.__class__}")
            self.ssl_model.module.epoch_end(self.epoch)

    def linear_evaluation(self): 
        if dist.get_rank() == 0: 
            logging.info("STARTING LINEAR EVALUATION")
            backbone = getattr(ModelRegistry, self.config.model_name)().cuda()
            backbone.load_state_dict(self.backbone.state_dict())
            lin_eval = FastLinearEvaluation(backbone, self.sl_dataset_train, self.sl_dataset_val)
            metrics = lin_eval.run()
            if metrics["val/corewise_auc"] > self.best_score:
                logging.info(f'New best score {metrics["val/corewise_auc"]} exceeds previous best score {self.best_score}!!!')
                self.best_score = metrics["val/corewise_auc"]
                wandb.run.summary["best_val_corewise_auc"] = self.best_score
                wandb.run.summary["best_val_epoch"] = self.epoch
                torch.save(
                    backbone.state_dict(),
                    os.path.join(self.ckpt_dir, "best_backbone.pth"),
                )
            wandb.log(metrics)
        else: 
            logging.info("Waiting for rank 0 to finish linear evaluation.")
        dist.barrier()

    def test(self):
        if dist.get_rank() == 0: 

            self.backbone.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_backbone.pth')))
            lin_eval = FastLinearEvaluation(self.backbone, self.sl_dataset_train, self.sl_dataset_val, self.sl_dataset_test)
            metrics = lin_eval.run()
            wandb.log(metrics)

        dist.barrier()

    def finetune_state(self):
        if not self.stage == "finetuning":
            return None
        
        return {
            "epoch": self.finetune_epoch,
            "classifier": self.classifier.state_dict(),
            "optimizer": self.sl_optimizer.state_dict(),
            "best_finetune_score": self.best_finetune_score,
        }

    def __call__(self):
        self.setup()
        logging.info("Running experiment")
        if self.config.ssl:
            while self.epoch < self.config.num_epochs:
                logging.info(f"Epoch {self.epoch}")
                self.ssl_train_epoch()
                if self.config.linear_evaluation_frequency is not None and self.epoch % self.config.linear_evaluation_frequency == 0: 
                    self.linear_evaluation()
                self.epoch += 1
        self.test()

    def checkpoint(self):
        logging.info("Checkpointing")
        from submitit.helpers import DelayedSubmission

        exp = Experiment(self.config, self.config_dict, self.exp_dir)
        return DelayedSubmission(exp)


class FastLinearEvaluation:
    def __init__(self, backbone, train_ds, val_ds, test_ds=None):
        self.backbone = backbone 
        self.train_ds = train_ds
        self.val_ds = val_ds 
        self.test_ds = test_ds

    def run(self): 
        logging.info("Extracting train features...")
        X_tr, df_tr = self.collect_feats_and_data(self.train_ds)
        logging.info("Features were extracted.")

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_tr, df_tr['label'])

        df_tr['prob_1'] = clf.predict_proba(X_tr)[:, 1]

        X_val, df_val = self.collect_feats_and_data(self.val_ds)
        df_val['prob_1'] = clf.predict_proba(X_val)[:, 1]

        if self.test_ds is not None: 
            X_test, df_test = self.collect_feats_and_data(self.test_ds)
            df_test['prob_1'] = clf.predict_proba(X_test)[:, 1]
        else: 
            df_test = None 

        metrics = create_epoch_report(df_tr, df_val, df_test)

        return metrics 

    def collect_feats_and_data(self, ds):
        self.backbone.eval()
        from src.utils import DataFrameCollector
        acc = DataFrameCollector()
        all_feats = []
        with torch.no_grad(): 
            for batch in tqdm(torch.utils.data.DataLoader(ds, batch_size=256, pin_memory=True, num_workers=4)):
                feats = self.backbone(batch.pop('patch').cuda())
                all_feats.append(feats)
                acc(batch)
        return torch.cat(all_feats).cpu().numpy(), acc.df


class SSLClassifier(nn.Module): 
    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone 
        self.linear_layer = nn.Linear(backbone.features_dim, 2)
        self.freeze_backbone = freeze_backbone
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
    def forward(self, x):
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            x = self.backbone(x)
        x = self.linear_layer(x)
        return x


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="config")
    parser.add_arguments(JobConfig, dest="job_config")

    args = parser.parse_args()
    job_config: JobConfig = args.job_config
    config: TrainConfig = args.config

    os.makedirs(job_config.log_dir, exist_ok=True)
    num_experiments = len(os.listdir(job_config.log_dir))

    from dataclasses import asdict
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if job_config.exp_dir is None:
        name = coolname.generate_slug(2)
        tag = f"{start_time}-{name}"
        exp_dir = os.path.join(job_config.log_dir, tag)
    else:
        exp_dir = job_config.exp_dir

    os.makedirs(exp_dir, exist_ok=True)

    exp = Experiment(config, asdict(config), exp_dir)

    if not job_config.submitit: 
        exp()
    else: 
        time_for_qos={'m2': 60*8, 'm3': 60*4, 'm4': 60*2, 'm5': 60*1}
        from submitit import SlurmExecutor
        executor = SlurmExecutor(folder=exp_dir, max_num_timeout=10)
        executor.update_parameters(
            mem_per_gpu=1024 * 128 // job_config.gpus,
            cpus_per_task=4,
            nodes=1,
            ntasks_per_node=job_config.gpus,
            time=time_for_qos[job_config.qos],
            gres=f"gpu:{job_config.gpus}",
            partition="a40,t4v2,rtx6000",
            qos=job_config.qos,
        )

        if not job_config.kfold:
            job = executor.submit(exp)
            print(job.job_id)
            print("Output: ", job.paths.stdout)
            print("Error: ", job.paths.stderr)

        else: 
            jobs = []
            with executor.batch():
                for fold in range(config.splits.n_folds):
                    fold_config = copy.deepcopy(config)
                    fold_config.splits.fold = fold
                    fold_exp_dir = os.path.join(exp_dir, f"fold-{fold}")
                    fold_config.group = os.path.basename(exp_dir)
                    job = executor.submit(
                        Experiment(fold_config, asdict(fold_config), fold_exp_dir)
                    )
                    jobs.append(job)

            for job in jobs: 
                print(job.job_id)
                print("Output: ", job.paths.stdout)
                print("Error: ", job.paths.stderr)

            print("Submitted all jobs")
