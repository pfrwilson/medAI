from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from typing import Dict, List, Tuple
# import numpy as np
import torch
import torchmetrics


class MetricCalculator(object):
    # list_of_metrics: List[str] = [
    #     roc_auc_score,
    #     balanced_accuracy_score,
    # ]
    list_of_metrics: List = [
        torchmetrics.functional.auroc,
        torchmetrics.functional.accuracy,
    ]
    
    def __init__(self, avg_core_logits_first: bool = False):
        self.avg_core_logits_first = avg_core_logits_first
        self.best_score = 0.0
        self.best_score_updated = False
        self.reset()

    def reset(self):
        self.core_id_logits = {}
        self.core_id_labels = {}

    def update(self, batch_meta_data, logits, labels):
        for i, id_tensor in enumerate(batch_meta_data["id"]):
            id = id_tensor.item()
            
            if id in self.core_id_logits:
                self.core_id_logits[id].append(logits[i])
                self.core_id_labels[id].append(labels[i])
            else:
                self.core_id_logits[id] = [logits[i]]
                self.core_id_labels[id] = [labels[i]]

    def get_metrics(self):
        patch_metrics: Dict = self.get_patch_metrics()
        core_metrics: Dict = self.get_core_metrics()
        patch_metrics.update(core_metrics)
        return patch_metrics
    
    def get_patch_metrics(self):
        logits = torch.cat(
            [torch.stack(logits_list) for logits_list in self.core_id_logits.values()]
            )
        labels = torch.cat(
            [torch.tensor(labels_list) for labels_list in self.core_id_labels.values()]
            )
        return self._get_metrics(logits, logits.argmax(dim=1), labels, prefix="patch_")
    
    def get_core_metrics(self):
        if self.avg_core_logits_first:
            logits = torch.cat(
                [torch.stack(logits_list).mean(dim=0) for logits_list in self.core_id_logits.values()]
                )          
            predicted_labels = logits.argmax(dim=1)
        else:
            logits = torch.stack(
                [torch.stack(logits_list).argmax(dim=1).mean(dim=0, dtype=torch.float32)
                for logits_list in self.core_id_logits.values()]
                )
            predicted_labels = logits >= 0.5

        labels = torch.stack(
            [labels_list[0] for labels_list in self.core_id_labels.values()]
            )
            
        return self._get_metrics(logits, predicted_labels, labels, prefix="core_")
    
    def _get_metrics(self, logits, predicted_labels, labels, prefix=""):
        metrics: Dict = {}
        for metric in self.list_of_metrics:
            metric_name: str = metric.__name__
            try:
                metric_value: float = metric(logits, labels, task="multiclass", num_classes=2)
            except:
                metric_value: float = metric(logits, labels, task="binary")
            metrics[prefix + metric_name] = metric_value
        return metrics

    def update_best_score(self, metrics, desc):
        if desc == "train":
            self.best_score_updated = False
            
        if desc == "val" and metrics["core_auroc"] > self.best_score:
                self.best_score = metrics["core_auroc"]
                self.best_score_updated = True
            
        if desc == "test" and self.best_score_updated:
            self.best_score_test = metrics["core_auroc"]
        
        return self.best_score_updated, self.best_score