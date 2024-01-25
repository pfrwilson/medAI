from attr import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
from utils.metrics import MetricCalculator, CoreMetricCalculator
import wandb
from copy import copy, deepcopy
import medAI
from memo_experiment import batched_marginal_entropy
from models.attention import MultiheadAttention as SimpleMultiheadAttention
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
        assert desc in ['val', 'test'], "desc should be either 'val' or 'test'"
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
    nhead: int = 8
    qk_dim: int = 64
    v_dim: int = 128
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
        
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=attention_config.nhead,
        #     dropout=attention_config.dropout,
        #     batch_first=True,
        #     ).cuda()
        
        self.attention = SimpleMultiheadAttention(
            input_dim=feature_dim,
            qk_dim=attention_config.qk_dim,
            v_dim=attention_config.v_dim,
            num_heads=attention_config.nhead,
            drop_out=attention_config.dropout
        ).cuda()
        
        # self.linear = nn.Linear(feature_dim, num_classes).cuda()
        self.linear = torch.nn.Sequential(
            # torch.nn.Linear(feature_dim, 64),
            torch.nn.Linear(attention_config.nhead*attention_config.v_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
            # torch.nn.Softmax(dim=1)
        ).cuda()
        
        self.metric_calculator = metric_calculator
        self.core_batch_size = core_batch_size
        self.ssl_epoch = ssl_epoch
        self.log_wandb = log_wandb

    def train(self, loader, epochs, train_backbone=False, backbone_lr=1e-4, head_lr=1e-3):
        if train_backbone:
            params = [
                {"params": self.feature_extractor.parameters(), "lr": backbone_lr},
                {"params": self.attention.parameters(), "lr": head_lr},
                {"params": self.linear.parameters(), "lr": head_lr}
                ]
        else:
            params = [
                {"params": self.attention.parameters(), "lr": head_lr},
                {"params": self.linear.parameters(), "lr": head_lr}
                ]
        
        optimizer = optim.Adam(params, weight_decay=1e-6)
        scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=2 * len(loader),
            max_epochs=epochs * len(loader),
        )
        
        for epoch in range(epochs):
            self.run_epoch(loader, optimizer, scheduler, 'train')
    
    def validate(self, loader, desc='val', use_memo=False, memo_lr=1e-3):
        assert desc in ['val', 'test'], "desc should be either 'val' or 'test'"
        if use_memo:
            return self.run_epoch_memo(loader, desc, memo_lr=memo_lr)
        return self.run_epoch(loader, None, None, desc)
    
    def run_epoch(self, loader, optimizer, scheduler, desc):
        self.feature_extractor.train() if desc == 'train' else self.feature_extractor.eval()
        self.attention.train() if desc == 'train' else self.attention.eval()
        self.linear.train() if desc == 'train' else self.linear.eval()
        
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
    
    def run_epoch_memo(self, loader, desc, memo_lr=1e-3):
        self.feature_extractor.eval()
        self.attention.eval()
        self.linear.eval()
        
        self.metric_calculator.reset()
        
        # batch_attention_reprs = []
        # batch_labels = []
        # batch_meta_data = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images_augs, images, labels, meta_data = batch
            images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            aug_size, seq_size, *img_shape = images_augs[0, ...].shape
            
            adaptation_fe_model = deepcopy(self.feature_extractor)
            adaptation_attention_model = deepcopy(self.attention)
            adaptation_linear_model = deepcopy(self.linear)

            params = [
                {"params": adaptation_attention_model.parameters()}, 
                {"params": adaptation_fe_model.parameters()},
                {"params": adaptation_linear_model.parameters()}
                ]
            optimizer = optim.SGD(params, lr=memo_lr)
            optimizer.zero_grad()
            
            
            ## Adapt to test ##
            
            # Update based on entropy
            _images_augs = images_augs[0, ...].reshape(aug_size*seq_size, *img_shape).cuda()
            reprs = adaptation_fe_model(_images_augs).reshape(aug_size, seq_size, -1)
            attention_reprs = adaptation_attention_model(reprs, reprs, reprs)[0].mean(dim=1) # [aug_size, feature_dim]
            
            outputs = adaptation_linear_model(attention_reprs).reshape(1, aug_size, -1)
            loss, logits = batched_marginal_entropy(outputs)
            loss.mean().backward()
            optimizer.step()
            
            # Forward
            reprs = adaptation_fe_model(images[0, ...]) # [seq_size, feature_dim]
            attention_reprs = adaptation_attention_model(reprs, reprs, reprs)[0].mean(dim=0)[None, ...]
            logits = adaptation_linear_model(attention_reprs)
            
            # Update metrics   
            self.metric_calculator.update(
                batch_meta_data = [meta_data],
                probs = F.softmax(logits, dim=-1).detach().cpu(),
                labels = labels.detach().cpu(),
            )
            
            
            '''
            # # Collect
            # batch_attention_reprs.append(attention_reprs)
            # batch_labels.append(labels[0])
            # batch_meta_data.append(meta_data)
            
            # if ((i + 1) % self.core_batch_size == 0) or (i == len(loader) - 1):
            #     batch_attention_reprs = torch.cat(batch_attention_reprs, dim=0) # [core_batch_size*aug_size, feature_dim]
            #     outputs = adaptation_linear_model(batch_attention_reprs).reshape(self.core_batch_size, aug_size, -1)
                
            #     labels = torch.stack(batch_labels, dim=0).cuda()
            #     loss, logits = batched_marginal_entropy(outputs)
            #     loss.mean().backward()
            #     optimizer.step()

            #     # Update metrics   
            #     self.metric_calculator.update(
            #         batch_meta_data = batch_meta_data,
            #         probs = F.softmax(logits, dim=-1).detach().cpu(),
            #         labels = labels.detach().cpu(),
            #     )
                
            #     batch_attention_reprs = []
            #     batch_labels = []
            #     batch_meta_data = []
            '''
        # Log metrics
        if self.log_wandb:
            best_score_updated, best_score = self.log_metrics(desc)
            return best_score_updated, best_score
        