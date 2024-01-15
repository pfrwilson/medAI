import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
from utils.metrics import MetricCalculator
import wandb
from copy import copy

class LinearProb:
    def __init__(
        self,
        in_features,
        out_features,
        ssl_epoch=0,
        metric_calculator=MetricCalculator()
        ):
        self.linear = nn.Linear(in_features, out_features).cuda()
        self.metric_calculator = metric_calculator
        self.ssl_epoch = ssl_epoch

    def train(self, loader, epochs, lr=1e-3):
        optimizer = optim.Adam(self.linear.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.run_epoch(loader, optimizer, 'train')
    
    def validate(self, loader, desc='val'):
        return self.run_epoch(loader, None, desc)
    
    def run_epoch(self, loader, optimizer, desc):
        self.linear.train() if desc == 'train' else self.linear.eval()
        
        for batch in tqdm(loader, desc=desc+"_linear_prob"):
            reprs, labels, meta_data = batch
            
            logits = self.linear(reprs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            if desc == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.log_losses(loss.item(), desc)
               
            # Update metrics   
            self.metric_calculator.update(
                batch_meta_data = meta_data,
                probs = F.softmax(logits, dim=-1).detach().cpu(),
                labels = labels.detach().cpu(),
            )
            
        # Log metrics
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