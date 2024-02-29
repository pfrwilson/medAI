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

from utils.sam_optimizer import SAM
from models.sar_model import SAR, configure_model, collect_params

import math

for LEAVE_OUT in ["JH", "PCC", "CRCEO", "PMCC", "UVA"]: #
    print("Leave out", LEAVE_OUT)

    ## Data Finetuning
    ###### No support dataset ######

    from baseline_experiment import BaselineConfig
    config = BaselineConfig(cohort_selection_config=LeaveOneCenterOutCohortSelectionOptions(leave_out=f"{LEAVE_OUT}"))

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

    # val_ds = ExactNCT2013RFImagePatches(
    #     split="val",
    #     transform=Transform(augment=True),
    #     cohort_selection_options=config.cohort_selection_config,
    #     patch_options=config.patch_config,
    #     debug=config.debug,
    # )
    
    if isinstance(config.cohort_selection_config, LeaveOneCenterOutCohortSelectionOptions):
        if config.cohort_selection_config.leave_out == "UVA":
            config.cohort_selection_config.benign_to_cancer_ratio = 5.0 
    
    test_ds = ExactNCT2013RFImagePatches(
        split="test",
        transform=Transform(augment=True),
        cohort_selection_options=config.cohort_selection_config,
        patch_options=config.patch_config,
        debug=config.debug,
    )


    # val_loader = DataLoader(
    #     val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    # )

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

    CHECkPOINT_PATH = os.path.join(f'/fs01/home/abbasgln/codes/medAI/projects/tta/logs/tta/baseline_gn_crtd3ratio_loco/baseline_gn_crtd3ratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    # CHECkPOINT_PATH = os.path.join(f'/fs01/home/abbasgln/codes/medAI/projects/tta/logs/tta/baseline_gn_avgprob_3ratio_loco/baseline_gn_avgprob_3ratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    # CHECkPOINT_PATH = os.path.join(f'/fs01/home/abbasgln/codes/medAI/projects/tta/logs/tta/baseline_gn_1nratio_loco/baseline_gn_1nratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')


    model.load_state_dict(torch.load(CHECkPOINT_PATH)['model'])
    model.eval()
    model.cuda()
    
    
    ## MEMO
    loader = test_loader
    
    # Get adapt model
    base_model = configure_model(deepcopy(model))
    params, param_names = collect_params(base_model) # only affine params in norm layers
            
    sam_optimizer = SAM(
        params,
        torch.optim.SGD,
        lr=config.optimizer_config.lr, 
        rho=0.05,
        momentum=0.9
        )
            
    adapt_model = SAR(
        base_model,
        sam_optimizer,
        episodic=True,
        steps=2,
        margin_e0=0.2*math.log(2),
        reset_constant_em=0.01,
        )

    metric_calculator = MetricCalculator()
    desc = "test"

    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(loader, desc=desc)):
        batch = deepcopy(batch)
        images_augs, images, labels, meta_data = batch
        # images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        logits = adapt_model(images)
        loss = criterion(logits, labels)
                        
        # Update metrics   
        metric_calculator.update(
            batch_meta_data = meta_data,
            probs = nn.functional.softmax(logits, dim=-1).detach().cpu(),
            labels = labels.detach().cpu(),
        )
    
    ## Find metrics
    # Log metrics every epoch
    metric_calculator.avg_core_probs_first = True
    metrics = metric_calculator.get_metrics(acc_threshold=0.3)

    # Update best score
    (best_score_updated,best_score) = metric_calculator.update_best_score(metrics, desc)

    best_score_updated = copy(best_score_updated)
    best_score = copy(best_score)
            
    # Log metrics
    metrics_dict = {
        f"{desc}/{key}": value for key, value in metrics.items()
        }

    print(metrics_dict)
    print(metric_calculator.get_metrics(acc_threshold=0.2))
    print(metric_calculator.get_metrics(acc_threshold=0.4))
    print(metric_calculator.get_metrics(acc_threshold=0.6))
    print(metric_calculator.get_metrics(acc_threshold=0.7))
    
    
    ## Log with wandb
    import wandb
    group=f"results_offline_sar_.2mrgn_3nratio_loco2"

    print(group)
    name= group + f"_{LEAVE_OUT}"
    wandb.init(project="tta", entity="mahdigilany", name=name, group=group)
    # os.environ["WANDB_MODE"] = "enabled"
    metrics_dict.update({"epoch": 0})
    wandb.log(
        metrics_dict,
        )
    wandb.finish()
    
    del test_ds, test_loader, loader, model