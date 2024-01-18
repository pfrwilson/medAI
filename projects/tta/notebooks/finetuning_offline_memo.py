import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import sys
sys.path.append(os.getenv("PROJECT_PATH"))


import torch
import torch.nn as nn
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig

from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import copy, deepcopy
import pandas as pd

from datasets.datasets import ExactNCT2013RFImagePatches
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)

for LEAVE_OUT in ["UVA", ]: #"JH", "PMCC",  "PCC"]:

    ## Data Finetuning
    ###### No support dataset ######

    from vicreg_pretrain_experiment import PretrainConfig
    config = PretrainConfig(cohort_selection_config=LeaveOneCenterOutCohortSelectionOptions(leave_out=f"{LEAVE_OUT}"))

    from baseline_experiment import BaselineConfig
    from torchvision.transforms import v2 as T
    from torchvision.tv_tensors import Image as TVImage

    class Transform:
        def __init__(selfT, augment=False):
            selfT.augment = augment
            selfT.size = (256, 256)
            # Augmentation
            selfT.transform = T.Compose([
                T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])  
        
        def __call__(selfT, item):
            patch = item.pop("patch")
            patch = copy(patch)
            patch = (patch - patch.min()) / (patch.max() - patch.min()) \
                if config.instance_norm else patch
            patch = TVImage(patch)
            patch = T.Resize(selfT.size, antialias=True)(patch).float()
            
            label = torch.tensor(item["grade"] != "Benign").long()
            
            if selfT.augment:
                patch_augs = torch.stack([selfT.transform(patch) for _ in range(2)], dim=0)
                return patch_augs, patch, label, item
            
            return -1, patch, label, item


    cohort_selection_options_train = copy(config.cohort_selection_config)
    cohort_selection_options_train.min_involvement = config.min_involvement_train
    cohort_selection_options_train.benign_to_cancer_ratio = config.benign_to_cancer_ratio_train
    cohort_selection_options_train.remove_benign_from_positive_patients = config.remove_benign_from_positive_patients_train

    train_ds = ExactNCT2013RFImagePatches(
        split="train",
        transform=Transform(augment=False),
        cohort_selection_options=cohort_selection_options_train,
        patch_options=config.patch_config,
        debug=config.debug,
    )

    val_ds = ExactNCT2013RFImagePatches(
        split="val",
        transform=Transform(augment=True),
        cohort_selection_options=config.cohort_selection_config,
        patch_options=config.patch_config,
        debug=config.debug,
    )

    test_ds = ExactNCT2013RFImagePatches(
        split="test",
        transform=Transform(augment=True),
        cohort_selection_options=config.cohort_selection_config,
        patch_options=config.patch_config,
        debug=config.debug,
    )


    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )


    ## Model
    from vicreg_pretrain_experiment import TimmFeatureExtractorWrapper
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


    fe_config = config.model_config

    # Create the model
    model: nn.Module = timm.create_model(
        fe_config.model_name,
        num_classes=fe_config.num_classes,
        in_chans=1,
        features_only=fe_config.features_only,
        norm_layer=lambda channels: nn.GroupNorm(
                        num_groups=fe_config.num_groups,
                        num_channels=channels
                        ))

    # Separate creation of classifier and global pool from feature extractor
    global_pool = SelectAdaptivePool2d(
        pool_type='avg',
        flatten=True,
        input_fmt='NCHW',
        )

    model = nn.Sequential(TimmFeatureExtractorWrapper(model), global_pool)


    # CHECkPOINT_PATH = os.path.join(os.getcwd(), f'logs/tta/vicreg_pretrain_gn_loco/vicreg_pretrain_gn_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    # CHECkPOINT_PATH = os.path.join(os.getcwd(), f'logs/tta/vicreg_pretrn_5e-3-20linprob_gn_loco/vicreg_pretrn_5e-3-20linprob_gn_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    CHECkPOINT_PATH = os.path.join(os.getcwd(), f'logs/tta/vicreg_pretrn_2048zdim_gn_loco/vicreg_pretrn_2048zdim_gn_loco_{LEAVE_OUT}/', 'best_model.ckpt')

    model.load_state_dict(torch.load(CHECkPOINT_PATH)['model'])
    model.eval()
    model.cuda()

    a = True
    ## Get train reprs
    from models.linear_prob import LinearProb

    loader = train_loader

    desc = "train"
    metric_calculator = MetricCalculator()
    # linear_prob = nn.Linear(512, 2).cuda()
    # optimizer = optim.Adam(linear_prob.parameters(), lr=1e-4)
    all_reprs_labels_metadata_train = []
    all_reprs = []
    all_labels = []
    for i, batch in enumerate(tqdm(loader, desc=desc)):
        batch = deepcopy(batch)
        images_augs, images, labels, meta_data = batch
        images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        reprs = model(images).detach()
        all_reprs.append(reprs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_reprs_labels_metadata_train.append((reprs, labels, meta_data))

        # logits = linear_prob(reprs)
        # loss = nn.CrossEntropyLoss()(logits, labels)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    all_reprs = np.concatenate(all_reprs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    ## Train linear model on reprs
    ### SKlearn logistic regression
    # from sklearn.linear_model import LogisticRegression

    # LR = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
    # LR.fit(all_reprs, all_labels)

    # # Assuming your input features have the same dimension as the scikit-learn model
    # input_features = LR.coef_.shape[1]  # Replace with the actual number of features
    # linear_prob = nn.Linear(input_features, 1) # Binary classification (1 output unit)

    # # Step 4: Assign the weights and bias from scikit-learn model to PyTorch model
    # with torch.no_grad():  # Disable gradient computation for this operation
    #     linear_prob.weight.data = torch.from_numpy(LR.coef_).float()
    #     linear_prob.bias.data = torch.from_numpy(LR.intercept_).float()

    # linear_prob.cuda()
    ### Linear prob 
    # os.environ["WANDB_MODE"] = "disabled"
    linear_prob: LinearProb = LinearProb(512, 2, metric_calculator=metric_calculator, log_wandb=False)
    linear_prob.train(all_reprs_labels_metadata_train,
                    epochs=15,
                    lr=5e-3
                    )
    ## Get test reprs
    ## MEMO on linear model
    loader = test_loader


    from memo_experiment import batched_marginal_entropy
    metric_calculator = MetricCalculator()
    desc = "test"

    criterion = nn.CrossEntropyLoss()


    for i, batch in enumerate(tqdm(loader, desc=desc)):
        batch = deepcopy(batch)
        images_augs, images, labels, meta_data = batch
        images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        batch_size, aug_size= images_augs.shape[0], images_augs.shape[1]

        # Adapt to test
        _images_augs = images_augs.reshape(-1, *images_augs.shape[2:]).cuda()
        adaptation_fe_model = deepcopy(model)
        # adaptation_head_model = deepcopy(linear_prob)
        adaptation_head_model = deepcopy(linear_prob.linear)
        # adaptation_fe_model.eval()
        params = [{"params": adaptation_head_model.parameters()}, {"params": adaptation_fe_model.parameters()}]
        optimizer = optim.SGD(params, lr=1e-3)
        
        # optimizer = optim.SGD(adaptation_head_model.parameters(), lr=1e-10)
        # reprs = adaptation_fe_model(_images_augs).detach() # for only adapting head
        for j in range(1):
            optimizer.zero_grad()
            reprs = adaptation_fe_model(_images_augs) # for only adapting head
            outputs = adaptation_head_model(reprs).reshape(batch_size, aug_size, -1)  
            loss, logits = batched_marginal_entropy(outputs)
            loss.mean().backward()
            optimizer.step()
        
        # Evaluate
        
        reprs = adaptation_fe_model(images)
        logits = adaptation_head_model(reprs)
        loss = criterion(logits, labels)
                        
        # Update metrics   
        metric_calculator.update(
            batch_meta_data = meta_data,
            probs = nn.functional.softmax(logits, dim=-1).detach().cpu(),
            # probs = nn.functional.tanh(logits).detach().cpu(),
            labels = labels.detach().cpu(),
        )
    
    ## Find metrics
    # Log metrics every epoch
    metrics = metric_calculator.get_metrics()

    # Update best score
    (
        best_score_updated,
        best_score
        ) = metric_calculator.update_best_score(metrics, desc)

    best_score_updated = copy(best_score_updated)
    best_score = copy(best_score)
            
    # Log metrics
    metrics_dict = {
        f"{desc}/{key}": value for key, value in metrics.items()
        }
    metrics_dict.update(best_score) if desc == "val" else None 


    # wandb.log(
    #     metrics_dict,
    #     )
    metrics_dict
    ## Log with wandb
    import wandb
    group=f"offline_vicreg_fintn_2048zdim_gn_loco"
    name= group + f"_{LEAVE_OUT}"
    wandb.init(project="tta", entity="mahdigilany", name=name, group=group)
    # os.environ["WANDB_MODE"] = "enabled"
    metrics_dict.update({"epoch": 0})
    wandb.log(
        metrics_dict,
        )
    wandb.finish()
    
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, loader, model