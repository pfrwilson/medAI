from attr import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
from utils.metrics import MetricCalculator, CoreMetricCalculator
import wandb
from copy import copy
import medAI
# from transformers.models.bert.modeling_bert import BertModel

class Fineturner:
    def __init__(
        self,
        feature_extractor: nn.Module,
        in_features,
        out_features,
        ssl_epoch=0,
        metric_calculator=MetricCalculator(),
        log_wandb=True
        ):
        self.feature_extractor = feature_extractor.cuda()
        self.linear = nn.Linear(in_features, out_features).cuda()
        self.metric_calculator = metric_calculator
        self.ssl_epoch = ssl_epoch
        self.log_wandb = log_wandb

    def train(self, loader, epochs, train_backbone=False, lr=1e-3):
        if train_backbone:
            params = [{"params": self.feature_extractor.parameters()}, {"params": self.linear.parameters()}]
        else:
            params = [{"params": self.linear.parameters()}]
        
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-6)
        scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=2 * len(loader),
            max_epochs=epochs * len(loader),
        )
        
        for epoch in range(epochs):
            self.run_epoch(loader, optimizer, scheduler, 'train')
    
    def validate(self, loader, desc='val'):
        return self.run_epoch(loader, None, None, desc)
    
    def run_epoch(self, loader, optimizer, scheduler, desc):
        self.feature_extractor.train() if desc == 'train' else self.feature_extractor.eval()
        self.linear.train() if desc == 'train' else self.linear.eval()
        
        for batch in tqdm(loader, desc=desc):
            images_augs, images, labels, meta_data = batch
            images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            reprs = self.feature_extractor(images)
            logits = self.linear(reprs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            if desc == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            
            if self.log_wandb:
                self.log_losses(loss.item(), desc)
               
            # Update metrics   
            self.metric_calculator.update(
                batch_meta_data = meta_data,
                probs = F.softmax(logits, dim=-1).detach().cpu(),
                labels = labels.detach().cpu(),
            )
            
        # Log metrics
        if self.log_wandb:
            best_score_updated, best_score = self.log_metrics(desc)
            return best_score_updated, best_score
    
    def log_losses(self, batch_loss_avg, desc):
        wandb.log(
            {f"{desc}/loss": batch_loss_avg, "epoch": self.ssl_epoch},
            commit=False
            )
    
    def log_metrics(self, desc):
        metrics = self.metric_calculator.get_metrics()
        
        # Reset metrics after each epoch
        self.metric_calculator.reset()
        
        # Update best score
        (
            best_score_updated,
            best_score
            ) = self.metric_calculator.update_best_score(metrics, desc)
                
        # Log metrics
        metrics_dict = {
            f"{desc}/{key}": value for key, value in metrics.items()
            }
        metrics_dict.update({"epoch": self.ssl_epoch})
        metrics_dict.update(best_score) if desc == "val" else None 
        wandb.log(
            metrics_dict,
            commit=True
            )
        
        return copy(best_score_updated), copy(best_score)
    

@dataclass
class AttentionConfig:
    nhead: int= 8
    dropout: float= 0.

class AttentionFineturner:
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim,
        num_classes,
        core_batch_size=10,
        ssl_epoch=0,
        attention_config: AttentionConfig = AttentionConfig(),
        metric_calculator: CoreMetricCalculator = CoreMetricCalculator(),
        log_wandb=True
        ):
        self.feature_extractor = feature_extractor.cuda()
        
        # self.attention = nn.TransformerEncoderLayer(
        #     d_model=attention_config.d_model,
        #     nhead=attention_config.nhead,
        #     dim_feedforward=attention_config.dim_feedforward,
        #     dropout=attention_config.dropout,
        #     activation=attention_config.activation,
        #     batch_first=True,
        #     ).cuda()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=attention_config.nhead,
            dropout=attention_config.dropout,
            batch_first=True,
            ).cuda()
        
        self.linear = nn.Linear(feature_dim, num_classes).cuda()
        
        self.metric_calculator = metric_calculator
        self.core_batch_size = core_batch_size
        self.ssl_epoch = ssl_epoch
        self.log_wandb = log_wandb

    def train(self, loader, epochs, train_backbone=False, backbone_lr=1e-4, attention_lr=1e-3):
        if train_backbone:
            params = [{"params": self.feature_extractor.parameters(), "lr": backbone_lr}, {"params": self.attention.parameters(), "lr": attention_lr}]
        else:
            params = [{"params": self.attention.parameters()}]
        
        optimizer = optim.Adam(params, weight_decay=1e-6)
        scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=2 * len(loader),
            max_epochs=epochs * len(loader),
        )
        
        for epoch in range(epochs):
            self.run_epoch(loader, optimizer, scheduler, 'train')
    
    def validate(self, loader, desc='val'):
        return self.run_epoch(loader, None, None, desc)
    
    def run_epoch(self, loader, optimizer, scheduler, desc):
        self.feature_extractor.train() if desc == 'train' else self.feature_extractor.eval()
        self.attention.train() if desc == 'train' else self.attention.eval()
        
        self.metric_calculator.reset()
        
        batch_attention_reprs = []
        batch_labels = []
        batch_meta_data = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images_augs, images, labels, meta_data = batch
            images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward
            reprs = self.feature_extractor(images[0, ...])
            attention_reprs = self.attention(reprs, reprs, reprs)[0].mean(dim=0)[None, ...]
            
            # Collect
            batch_attention_reprs.append(attention_reprs)
            batch_labels.append(labels[0])
            batch_meta_data.append(meta_data)
            
            if ((i + 1) % self.core_batch_size == 0) or (i == len(loader) - 1):
                batch_attention_reprs = torch.cat(batch_attention_reprs, dim=0)
                logits = self.linear(batch_attention_reprs)
                
                labels = torch.stack(batch_labels, dim=0).cuda()
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                if desc == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                if self.log_wandb:
                    self.log_losses(loss.item(), desc)
                
                # Update metrics   
                self.metric_calculator.update(
                    batch_meta_data = batch_meta_data,
                    probs = F.softmax(logits, dim=-1).detach().cpu(),
                    labels = labels.detach().cpu(),
                )
                
                batch_attention_reprs = []
                batch_labels = []
                batch_meta_data = []
            
        # Log metrics
        if self.log_wandb:
            best_score_updated, best_score = self.log_metrics(desc)
            return best_score_updated, best_score
    
    def log_losses(self, batch_loss_avg, desc):
        wandb.log(
            {f"{desc}/loss": batch_loss_avg, "epoch": self.ssl_epoch},
            commit=False
            )
    
    def log_metrics(self, desc):
        metrics = self.metric_calculator.get_metrics()
        
        # Reset metrics after each epoch
        self.metric_calculator.reset()
        
        # Update best score
        (
            best_score_updated,
            best_score
            ) = self.metric_calculator.update_best_score(metrics, desc)
                
        # Log metrics
        metrics_dict = {
            f"{desc}/{key}": value for key, value in metrics.items()
            }
        metrics_dict.update({"epoch": self.ssl_epoch})
        metrics_dict.update(best_score) if desc == "val" else None 
        wandb.log(
            metrics_dict,
            commit=True
            )
        
        return copy(best_score_updated), copy(best_score)
    