from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from typing import Dict, List, Tuple
import numpy as np


class MetricCalculator(object):
    list_of_metrics: List[str] = [
        roc_auc_score,
        balanced_accuracy_score,
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
        for id_tensor in batch_meta_data["id"]:
            id = id_tensor.item()
            
            if id in self.core_id_logits:
                self.core_id_logits[id].append(logits)
                self.core_id_labels[id].append(labels)
            else:
                self.core_id_logits[id] = [logits]
                self.core_id_labels[id] = [labels]

    def get_metrics(self):
        patch_metrics: Dict = self.get_patch_metrics()
        core_metrics: Dict = self.get_core_metrics()
        return patch_metrics.update(core_metrics)
    
    def get_patch_metrics(self):
        logits = np.concatenate(
            [np.concatenate(logits_list) for logits_list in self.core_id_logits.values()]
            )
        labels = np.concatenate(
            [np.concatenate(labels_list) for labels_list in self.core_id_labels.values()]
            )
        return self._get_metrics(logits, labels, prefix="patch_")
    
    def get_core_metrics(self):
        
        if self.avg_core_logits_first:
            logits = np.concatenate(
                [np.mean(
                    np.concatenate(logits_list),
                    axis=0
                    ) for logits_list in self.core_id_logits.values()]
                )          
        else:
            logits = np.asarray(
                [np.mean(
                    np.argmax(np.concatenate(logits_list), axis=1),
                    axis=0
                    ) for logits_list in self.core_id_logits.values()]
                )

        labels = np.asarray(
            [labels_list[0][0] for labels_list in self.core_id_labels.values()]
            )
            
        return self._get_metrics(logits, labels, prefix="core_")
    
    def _get_metrics(self, logits, labels, prefix=""):
        metrics: Dict = {}
        for metric in self.list_of_metrics:
            metric_name: str = metric.__name__
            try:
                metric_value: float = metric(labels, logits)
            except:
                metric_value: float = metric(labels, logits.argmax(axis=1))
            metrics[prefix + metric_name] = metric_value
        return metrics

    def update_best_score(self, metrics, desc):
        if desc == "train":
            self.best_score_updated = False
            
        if desc == "val" and metrics["core_roc_auc_score"] > self.best_score:
                self.best_score = metrics["core_roc_auc_score"]
                self.best_score_updated = True
            
        if desc == "test" and self.best_score_updated:
            self.best_score_test = metrics["core_roc_auc_score"]
        
        return self.best_score_updated, self.best_score