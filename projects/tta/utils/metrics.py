from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from typing import Dict, List, Tuple
# import numpy as np
import torch
import torchmetrics
from copy import deepcopy

def accuracy(*args, **kwargs):
    if "threshold" in kwargs:
        threshold = kwargs["threshold"]
        del kwargs["threshold"]
    else:
        threshold = 0.4
    return torchmetrics.functional.accuracy(*args, **kwargs, threshold=threshold)

class MetricCalculator(object):
    # list_of_metrics: List[str] = [
    #     roc_auc_score,
    #     balanced_accuracy_score,
    # ]
    list_of_metrics: List = [
        torchmetrics.functional.auroc,
        accuracy,
    ]
    
    def __init__(self, high_inv_thresh = 40.0, include_all_inv = True, avg_core_logits_first: bool = False):
        self.avg_core_logits_first = avg_core_logits_first
        self.high_inv_thresh = high_inv_thresh
        self.include_all_inv = include_all_inv
        self.best_score_updated = False
        self.initialize_best_score()
        self.reset()

    def reset(self):
        self.core_id_logits = {}
        self.core_id_labels = {}
        self.core_id_invs = {}

    def update(self, batch_meta_data, logits, labels):
        invs = deepcopy(batch_meta_data["pct_cancer"])
        ids = deepcopy(batch_meta_data["id"])
        for i, id_tensor in enumerate(ids):
            id = id_tensor.item()
            
            # Dict of invs
            self.core_id_invs[id] = invs[i]
            
            # Dict of logits and labels
            if id in self.core_id_logits:
                self.core_id_logits[id].append(logits[i])
                self.core_id_labels[id].append(labels[i])
            else:
                self.core_id_logits[id] = [logits[i]]
                self.core_id_labels[id] = [labels[i]]

    def remove_low_inv_ids(self, core_id_invs):
        high_inv_ids = []
        for id, inv in core_id_invs.items():
            if (inv >= self.high_inv_thresh) or (inv == 0.0) or torch.isnan(inv):
                high_inv_ids.append(id)
        return high_inv_ids
       
    def get_metrics(self):
        high_inv_core_ids = self.remove_low_inv_ids(self.core_id_invs)
        patch_metrics: Dict = self.get_patch_metrics(high_inv_core_ids)
        core_metrics: Dict = self.get_core_metrics(high_inv_core_ids)
        if self.include_all_inv:
            all_inv_patch_metrics: Dict = self.get_patch_metrics()
            all_inv_core_metrics: Dict = self.get_core_metrics()
            patch_metrics.update(all_inv_patch_metrics)
            core_metrics.update(all_inv_core_metrics)
        patch_metrics.update(core_metrics)
        return patch_metrics
    
    def get_patch_metrics(self, core_ids = None):
        if core_ids is None:
            ids = self.core_id_logits.keys()
        else:
            ids = core_ids
            
        logits = torch.cat(
            [torch.stack(logits_list) for id, logits_list in self.core_id_logits.items() if id in ids]
            )
        labels = torch.cat(
            [torch.tensor(labels_list) for id, labels_list in self.core_id_labels.items() if id in ids]
            )
        return self._get_metrics(
            logits, 
            labels, 
            prefix="all_inv_patch_" if core_ids is None else "patch_"
            )
    
    def get_core_metrics(self, core_ids = None):
        if core_ids is None:
            ids = self.core_id_logits.keys()
        else:
            ids = core_ids
        
        if self.avg_core_logits_first:
            logits = torch.cat(
                [torch.stack(logits_list).mean(dim=0) for id, logits_list in self.core_id_logits.items() if id in ids])          
        else:
            logits = torch.stack(
                [torch.stack(logits_list).argmax(dim=1).mean(dim=0, dtype=torch.float32)
                for id, logits_list in self.core_id_logits.items() if id in ids])
            logits = torch.cat([(1 - logits).unsqueeze(1), logits.unsqueeze(1)], dim=1)

        labels = torch.stack(
            [labels_list[0] for id, labels_list in self.core_id_labels.items() if id in ids])
        
        return self._get_metrics(
            logits, 
            labels, 
            prefix="all_inv_core_" if core_ids is None else "core_"
            )
            
    def _get_metrics(self, logits, labels, prefix=""):
        metrics: Dict = {}
        for metric in self.list_of_metrics:
            metric_name: str = metric.__name__
            try:
                metric_value: float = metric(logits, labels, task="multiclass", num_classes=2)
            except:
                metric_value: float = metric(logits, labels, task="binary")
            metrics[prefix + metric_name] = metric_value
        return metrics

    def _get_best_score_dict(self):
        return {
            "val/best_core_auroc": self.best_val_score,
            "val/best_patch_auroc": self.best_val_patch_score,
            "val/best_all_inv_core_auroc": self.best_all_inv_val_score,
            "test/best_core_auroc": self.best_test_score,
            "test/best_patch_auroc": self.best_test_patch_score,
            "test/best_all_inv_core_auroc": self.best_all_inv_test_score,
        }
    
    def initialize_best_score(self, best_score_dict: Dict = None):
        if best_score_dict is None:
            self.best_val_score = 0.0
            self.best_val_patch_score = 0.0
            self.best_all_inv_val_score = 0.0
            self.best_test_score = 0.0
            self.best_test_patch_score = 0.0
            self.best_all_inv_test_score = 0.0
        else:
            self.best_val_score = best_score_dict["val/best_core_auroc"]
            self.best_val_patch_score = best_score_dict["val/best_patch_auroc"]
            self.best_all_inv_val_score = best_score_dict["val/best_all_inv_core_auroc"]
            self.best_test_score = best_score_dict["test/best_core_auroc"]
            self.best_test_patch_score = best_score_dict["test/best_patch_auroc"]
            self.best_all_inv_test_score = best_score_dict["test/best_all_inv_core_auroc"]
            
        return self._get_best_score_dict()
    
    def update_best_score(self, metrics, desc):
        """This function assumes test is after val and it should receive metrics from val first
        Also, it should receive metrics to calculate the best score"""
        self.best_score_updated = False
            
        if desc == "val" and metrics["core_auroc"] >= self.best_val_score:
                self.best_val_score = metrics["core_auroc"]
                self.best_val_patch_score = metrics["patch_auroc"]
                self.best_all_inv_val_score = metrics["all_inv_core_auroc"]
                self.best_score_updated = True
            
        if desc == "test":
            self.best_test_score = metrics["core_auroc"]
            self.best_test_patch_score = metrics["patch_auroc"]
            self.best_all_inv_test_score = metrics["all_inv_core_auroc"]
                
        return self.best_score_updated, self._get_best_score_dict()
