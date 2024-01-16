import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
import wandb
from copy import copy

class LinearProb:
    def __init__(
        self,
        in_features,
        out_features,
        ssl_epoch=0,
        best_val_score=0.0,
        log_wandb=True
        ):
        self.linear = nn.Linear(in_features, out_features).cuda()
        self.ssl_epoch = ssl_epoch
        self.log_wandb = log_wandb
        self.best_val_score = best_val_score
        self.best_val_score_updated = False

    def train(self, loader, epochs, lr=1e-3):
        optimizer = optim.Adam(self.linear.parameters(), lr=lr, weight_decay=1e-6)
        
        for epoch in range(epochs):
            self.run_epoch(loader, optimizer, 'train')
    
    def validate(self, loader, desc='val'):
        return self.run_epoch(loader, None, desc)
    
    def run_epoch(self, loader, optimizer, desc):
        self.linear.train() if desc == 'train' else self.linear.eval()
        
        
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=desc+"_linear_prob"):
            reprs, labels = batch
            
            logits = self.linear(reprs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            if desc == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if self.log_wandb:
                self.log_losses(loss.item(), desc)
            
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        
        # Log metrics
        if self.log_wandb: 
            best_score_updated, best_score = self.log_metrics(accuracy, desc=desc)
            return best_score_updated, best_score
        
        return False, accuracy
    
    def log_losses(self, batch_loss_avg, desc):
        wandb.log(
            {f"{desc}/loss": batch_loss_avg, "epoch": self.ssl_epoch},
            commit=False
            )
    
    def log_metrics(self, accuracy, desc):      
        # Log metrics
        metrics_dict = {
            f"{desc}/accuracy": accuracy,
            }
        metrics_dict.update({"epoch": self.ssl_epoch})
        
        if desc=='val' and self.best_val_score <= accuracy:
            self.best_val_score_updated = True
            self.best_val_score = accuracy
            metrics_dict.update({f"{desc}/best_accuracy": self.best_val_score})
        elif desc=='val':
            self.best_val_score_updated = False
        elif desc=='test':
            metrics_dict.update({f"{desc}/best_accuracy": accuracy})
        
        wandb.log(
            metrics_dict,
            commit=True
            )
        
        return copy(self.best_val_score_updated), copy(self.best_val_score)