import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def calculate_metrics(predictions, labels, log_images=False): 
    """Calculate metrics for the cancer classification problem."""

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    metrics = {}

    # core predictions
    core_probs = predictions
    core_labels = labels
    metrics["core_auc"] = roc_auc_score(core_labels, core_probs)
    
    # find the sensitivity at fixed specificities 
    fpr, tpr, thresholds = roc_curve(core_labels, core_probs)
    
    for specificity in [0.20, 0.40, 0.60, 0.80]: 
        sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
        metrics[f"sens_at_{specificity*100:.0f}_spe"] = sensitivity

    # choose the threshold that maximizes balanced accuracy
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    metrics["f1"] = f1_score(core_labels, core_probs > best_threshold)

    if log_images:
        plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
        plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
        plt.legend(["Benign", "Cancer"])
        plt.xlabel(f"Probability of cancer")
        plt.ylabel("Density")
        plt.title(f"Core AUC: {metrics['core_auc']:.3f}")
        metrics["histogram"] = wandb.Image(
                    plt, caption="Histogram of core predictions"
        )
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        metrics["roc_curve"] = wandb.Image(
                    plt, caption="ROC curve"
        )
        plt.close()

    return metrics