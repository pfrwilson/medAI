from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from typing import Dict, List, Tuple
import numpy as np
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
    
    def __init__(self, high_inv_thresh = 40.0, include_all_inv = True, avg_core_probs_first: bool = False):
        self.avg_core_probs_first = avg_core_probs_first
        self.high_inv_thresh = high_inv_thresh
        self.include_all_inv = include_all_inv
        self.best_score_updated = False
        self.initialize_best_score()
        self.reset()

    def reset(self):
        self.core_id_probs = {}
        self.core_id_labels = {}
        self.core_id_invs = {}

    def update(self, batch_meta_data, probs, labels):
        invs = deepcopy(batch_meta_data["pct_cancer"])
        ids = deepcopy(batch_meta_data["id"])
        for i, id_tensor in enumerate(ids):
            id = id_tensor.item()
            
            # Dict of invs
            self.core_id_invs[id] = invs[i]
            
            # Dict of probs and labels
            if id in self.core_id_probs:
                self.core_id_probs[id].append(probs[i])
                self.core_id_labels[id].append(labels[i])
            else:
                self.core_id_probs[id] = [probs[i]]
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
            ids = self.core_id_probs.keys()
        else:
            ids = core_ids
            
        probs = torch.cat(
            [torch.stack(probs_list) for id, probs_list in self.core_id_probs.items() if id in ids]
            )
        labels = torch.cat(
            [torch.tensor(labels_list) for id, labels_list in self.core_id_labels.items() if id in ids]
            )
        return self._get_metrics(
            probs, 
            labels, 
            prefix="all_inv_patch_" if core_ids is None else "patch_"
            )
    
    def get_core_metrics(self, core_ids = None):
        if core_ids is None:
            ids = self.core_id_probs.keys()
        else:
            ids = core_ids
        
        if self.avg_core_probs_first:
            probs = torch.cat(
                [torch.stack(probs_list).mean(dim=0) for id, probs_list in self.core_id_probs.items() if id in ids])          
        else:
            probs = torch.stack(
                [torch.stack(probs_list).argmax(dim=1).mean(dim=0, dtype=torch.float32)
                for id, probs_list in self.core_id_probs.items() if id in ids])
            probs = torch.cat([(1 - probs).unsqueeze(1), probs.unsqueeze(1)], dim=1)

        labels = torch.stack(
            [labels_list[0] for id, labels_list in self.core_id_labels.items() if id in ids])
        
        return self._get_metrics(
            probs, 
            labels, 
            prefix="all_inv_core_" if core_ids is None else "core_"
            )
            
    def _get_metrics(self, probs, labels, prefix=""):
        metrics: Dict = {}
        for metric in self.list_of_metrics:
            metric_name: str = metric.__name__
            try:
                metric_value: float = metric(probs, labels, task="multiclass", num_classes=2)
            except:
                metric_value: float = metric(probs, labels, task="binary")
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
        """This function assumes test is after val"""           
        if desc == "val" and metrics["core_auroc"] >= self.best_val_score:
                self.best_val_score = metrics["core_auroc"]
                self.best_val_patch_score = metrics["patch_auroc"]
                self.best_all_inv_val_score = metrics["all_inv_core_auroc"]
                self.best_score_updated = True
        elif desc == "val":
            self.best_score_updated = False
            
        if desc == "test" and self.best_score_updated:
            self.best_test_score = metrics["core_auroc"]
            self.best_test_patch_score = metrics["patch_auroc"]
            self.best_all_inv_test_score = metrics["all_inv_core_auroc"]
            self.best_score_updated = False
                
        return self.best_score_updated, self._get_best_score_dict()


def brier_score(probs, targets, weighted=True):
    brier = (probs - targets) ** 2
    ind_pos = targets == 1
    ind_neg = targets == 0

    if weighted:
        brier[ind_pos] *= ind_neg.sum() / ind_pos.sum()

    return np.mean(brier)


def expected_calibration_error(preds, confidence, targets, n_bins=10):
    # make everything numpy
    preds = np.array(preds)
    confidence = np.array(confidence)
    targets = np.array(targets)

    # make sure confidence is between 0 and 1
    assert np.all(confidence >= 0) and np.all(
        confidence <= 1
    ), "confidence must be between 0 and 1"

    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.digitize(confidence, bins)

    acc_by_bin = np.zeros(n_bins)
    n_by_bin = np.zeros(n_bins)
    conf_by_bin = np.zeros(n_bins)

    for i in range(n_bins):
        bin_indices = indices == i
        if np.sum(bin_indices) == 0:
            continue
        acc_by_bin[i] = np.mean(preds[bin_indices] == targets[bin_indices])
        n_by_bin[i] = len(preds[bin_indices])
        conf_by_bin[i] = np.mean(confidence[bin_indices])

    ece = np.sum(np.abs(acc_by_bin - conf_by_bin) * n_by_bin) / np.sum(n_by_bin)

    return ece, {
        "acc_by_bin": acc_by_bin,
        "conf_by_bin": conf_by_bin,
        "n_by_bin": n_by_bin,
        "bins": bins,
    }


class CoreMetricCalculator(MetricCalculator):
    def update(self, batch_meta_data, probs, labels):
        invs = [meta_data["pct_cancer"][0] for meta_data in batch_meta_data]
        ids = [meta_data["id"][0] for meta_data in batch_meta_data]
        for i, id_tensor in enumerate(ids):
            id = id_tensor.item()
            
            # Dict of invs
            self.core_id_invs[id] = invs[i]
            
            # Dict of probs and labels
            if id in self.core_id_probs:
                self.core_id_probs[id].append(probs[i])
                self.core_id_labels[id].append(labels[i])
            else:
                self.core_id_probs[id] = [probs[i]]
                self.core_id_labels[id] = [labels[i]]
                
    def get_metrics(self):
        high_inv_core_ids = self.remove_low_inv_ids(self.core_id_invs)
        core_metrics: Dict = self.get_core_metrics(high_inv_core_ids)
        if self.include_all_inv:
            all_inv_core_metrics: Dict = self.get_core_metrics()
            core_metrics.update(all_inv_core_metrics)
        return core_metrics
    
    def update_best_score(self, metrics, desc):
        """This function assumes test is after val"""           
        if desc == "val" and metrics["core_auroc"] >= self.best_val_score:
                self.best_val_score = metrics["core_auroc"]
                self.best_all_inv_val_score = metrics["all_inv_core_auroc"]
                self.best_score_updated = True
        elif desc == "val":
            self.best_score_updated = False
            
        if desc == "test" and self.best_score_updated:
            self.best_test_score = metrics["core_auroc"]
            self.best_all_inv_test_score = metrics["all_inv_core_auroc"]
            self.best_score_updated = False
                
        return self.best_score_updated, self._get_best_score_dict()
    
    def _get_best_score_dict(self):
        return {
            "val/best_core_auroc": self.best_val_score,
            "val/best_all_inv_core_auroc": self.best_all_inv_val_score,
            "test/best_core_auroc": self.best_test_score,
            "test/best_all_inv_core_auroc": self.best_all_inv_test_score,
        }
    
    def initialize_best_score(self, best_score_dict: Dict = None):
        if best_score_dict is None:
            self.best_val_score = 0.0
            self.best_all_inv_val_score = 0.0
            self.best_test_score = 0.0
            self.best_all_inv_test_score = 0.0
        else:
            self.best_val_score = best_score_dict["val/best_core_auroc"]
            self.best_all_inv_val_score = best_score_dict["val/best_all_inv_core_auroc"]
            self.best_test_score = best_score_dict["test/best_core_auroc"]
            self.best_all_inv_test_score = best_score_dict["test/best_all_inv_core_auroc"]
    
    # def get_core_metrics(self, core_ids = None):
    #     if core_ids is None:
    #         ids = self.core_id_probs.keys()
    #     else:
    #         ids = core_ids
        
    #     probs = torch.stack(
    #         [torch.stack(probs_list).argmax(dim=1).mean(dim=0, dtype=torch.float32)
    #         for id, probs_list in self.core_id_probs.items() if id in ids])
    #     probs = torch.cat([(1 - probs).unsqueeze(1), probs.unsqueeze(1)], dim=1)

    #     labels = torch.stack(
    #         [labels_list[0] for id, labels_list in self.core_id_labels.items() if id in ids])
        
    #     return self._get_metrics(
    #         probs, 
    #         labels, 
    #         prefix="all_inv_core_" if core_ids is None else "core_"
    #         )