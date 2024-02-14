import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import sys
sys.path.append(os.getenv("PROJECT_PATH"))


import torch
import torch.nn as nn
import torch.nn.functional as F
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

for LEAVE_OUT in ["JH", "PCC", "PMCC", "UVA", "CRCEO"]: # 
    print("Leave out", LEAVE_OUT)
    
    ## Data Finetuning
    ###### No support dataset ######

    from ensemble_experiment import EnsembleConfig
    config = EnsembleConfig(cohort_selection_config=LeaveOneCenterOutCohortSelectionOptions(leave_out=f"{LEAVE_OUT}"),
    )

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


    # val_ds = ExactNCT2013RFImagePatches(
    #     split="val",
    #     transform=Transform(augment=False),
    #     cohort_selection_options=config.cohort_selection_config,
    #     patch_options=config.patch_config,
    #     debug=config.debug,
    # )

    test_ds = ExactNCT2013RFImagePatches(
        split="test",
        transform=Transform(augment=False),
        cohort_selection_options=config.cohort_selection_config,
        patch_options=config.patch_config,
        debug=config.debug,
    )


    # val_loader = DataLoader(
    #     val_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    # )

    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )


    ## Model
    from baseline_experiment import FeatureExtractorConfig
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
    from divemble_experiment import TimmFeatureExtractorWrapper

    fe_config = FeatureExtractorConfig()

    # Create the model
    models: tp.List[nn.Module] = [timm.create_model(
        fe_config.model_name,
        num_classes=fe_config.num_classes,
        in_chans=1,
        features_only=True,
        norm_layer=lambda channels: nn.GroupNorm(
                        num_groups=fe_config.num_groups,
                        num_channels=channels
                        )) for _ in range(5)]
    
    global_pools = [SelectAdaptivePool2d(pool_type='avg',flatten=True,input_fmt='NCHW').cuda() 
                for _ in range(config.num_ensembles)]
    
    fe_models = [nn.Sequential(TimmFeatureExtractorWrapper(model), global_pool) 
                for model, global_pool in zip(models, global_pools)]
    
    # linears = [nn.Linear(512, self.config.model_config.num_classes).cuda()
    #            for _ in range(self.config.num_ensembles)]
        
    shared_linear = nn.Linear(512, config.model_config.num_classes).cuda()
    linears = [shared_linear for _ in range(config.num_ensembles)]

    CHECkPOINT_PATH = os.path.join(f'/fs01/home/abbasgln/codes/medAI/projects/tta/logs/tta/Divemble-logt_gn_5mdls_0.5var0.05cov_3ratio_loco/Divemble-logt_gn_5mdls_0.5var0.05cov_3ratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')

    state = torch.load(CHECkPOINT_PATH)
    [model.load_state_dict(state["list_fe_models"][i]) for i, model in enumerate(fe_models)]
    [linear.load_state_dict(state["list_linears"][i]) for i, linear in enumerate(linears)]
    
    list_models = [nn.Sequential(fe_model, linear) for fe_model, linear in zip(fe_models, linears)]
    [model.eval() for model in list_models]
    [model.cuda() for model in list_models]
    
    
    # ## Temp Scaling
    # loader = val_loader

    # metric_calculator = MetricCalculator()
    # desc = "val"


    # temp = torch.tensor(1.0).cuda().requires_grad_(True)
    # beta = torch.tensor(0.0).cuda().requires_grad_(True)


    # params = [temp, beta]
    # _optimizer = optim.Adam(params, lr=1e-3)

    # for epoch in range(1):
    #     metric_calculator.reset()
    #     for i, batch in enumerate(tqdm(loader, desc=desc)):
    #         images_augs, images, labels, meta_data = batch
    #         images = images.cuda()
    #         labels = labels.cuda()
            

    #         # Evaluate
    #         with torch.no_grad():
    #             stacked_logits = torch.stack([model(images) for model in list_models])
    #         scaled_stacked_logits = stacked_logits/ temp + beta
    #         losses = [nn.CrossEntropyLoss()(
    #             scaled_stacked_logits[i, ...],
    #             labels
    #             ) for i in range(5)
    #         ]
            
    #         # optimize
    #         _optimizer.zero_grad()
    #         sum(losses).backward()
    #         _optimizer.step()
                        
    #         # Update metrics   
    #         metric_calculator.update(
    #             batch_meta_data = meta_data,
    #             probs = nn.functional.softmax(scaled_stacked_logits, dim=-1).mean(dim=0).detach().cpu(), # Take mean over ensembles
    #             labels = labels.detach().cpu(),
    #         )
    # print("temp beta", temp, beta)
    
    
    temp = 1.0
    beta = 0.0
    if LEAVE_OUT == "JH":
        temp = 1.6793
        beta = -1.0168
    elif LEAVE_OUT == "PCC":
        temp = 1.5950
        beta = -0.8514
    elif LEAVE_OUT == "PMCC":
        temp = 0.6312
        beta = -1.0017
    elif LEAVE_OUT == "UVA":
        temp = 0.9333
        beta = -0.7474
    elif LEAVE_OUT == "CRCEO":
        temp = 1.2787
        beta = -0.8716
        
    temp = torch.tensor(temp).cuda()
    beta = torch.tensor(beta).cuda()
    
    
    ## Test-time Adaptation
    # loader = test_test_loader
    loader = test_loader
    enable_pseudo_label = True
    temp_scale = False
    certain_threshold = 0.8

    metric_calculator = MetricCalculator()
    desc = "test"

    for i, batch in enumerate(tqdm(loader, desc=desc)):
        images_augs, images, labels, meta_data = batch
        # images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        adaptation_model_list = [deepcopy(model) for model in list_models] 
        [model.eval() for model in adaptation_model_list]

        
        if enable_pseudo_label:
            params = []
            for model in adaptation_model_list:
                params.append({'params': model.parameters()})
            optimizer = optim.SGD(params, lr=5e-4)
            
            # Adapt to test
            for j in range(1):
                optimizer.zero_grad()
                # Forward pass
                stacked_logits = torch.stack([model(images) for model in adaptation_model_list])
                if temp_scale:
                    stacked_logits = stacked_logits / temp + beta
                
                # Remove uncertain samples from test-time adaptation
                certain_idx = F.softmax(stacked_logits, dim=-1).mean(dim=0).max(dim=-1)[0] >= certain_threshold
                stacked_logits = stacked_logits[:, certain_idx, ...]
                
                list_losses = []
                for k, outputs in enumerate(adaptation_model_list):
                    loss = nn.CrossEntropyLoss()(stacked_logits[k, ...], F.softmax(stacked_logits, dim=-1).mean(dim=0).argmax(dim=-1))
                    list_losses.append(loss.mean())
                # Backward pass
                sum(list_losses).backward()
                optimizer.step()
            
        # Evaluate
        logits = torch.stack([model(images) for model in adaptation_model_list])
        if temp_scale:
            logits = logits / temp + beta
        losses = [nn.CrossEntropyLoss()(
            logits[i, ...],
            labels
            ) for i in range(5)
        ]
                        
        # Update metrics   
        metric_calculator.update(
            batch_meta_data = meta_data,
            probs = nn.functional.softmax(logits, dim=-1).mean(dim=0).detach().cpu(), # Take mean over ensembles
            labels = labels.detach().cpu(),
        )
    
    ## Get metrics    
    avg_core_probs_first = True
    metric_calculator.avg_core_probs_first = avg_core_probs_first

    # Log metrics every epoch
    metrics = metric_calculator.get_metrics()

    # Update best score
    (best_score_updated,best_score) = metric_calculator.update_best_score(metrics, desc)

    best_score_updated = copy(best_score_updated)
    best_score = copy(best_score)
            
    # Log metrics
    metrics_dict = {
        f"{desc}/{key}": value for key, value in metrics.items()
        }

    print(metrics_dict)
        
    ## Log with wandb
    import wandb
    # group=f"offline_combDivEnsmPsdo_gn_3ratio_loco"
    group=f"offline_combDivEnsmPsdo_.8uncrtnty_gn_3ratio_loco"
    name= group + f"_{LEAVE_OUT}"
    wandb.init(project="tta", entity="mahdigilany", name=name, group=group)
    # os.environ["WANDB_MODE"] = "enabled"
    metrics_dict.update({"epoch": 0})
    wandb.log(
        metrics_dict,
        )
    
    
    wandb.finish()
    # del val_ds, test_ds, val_loader, test_loader, loader, list_models
    del test_ds,  test_loader, loader, list_models