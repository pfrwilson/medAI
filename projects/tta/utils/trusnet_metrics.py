from typing import Callable, Dict, Sequence
import torch
import itertools
import warnings
import numpy as np
from typing import Literal
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss
from torchmetrics.classification import CalibrationError

import logging


class EarlyStoppingMonitor:
    def __init__(self, patience, on_early_stop_triggered):
        self.patience = patience
        self.callback_fn = on_early_stop_triggered

        self.strikes = 0
        self.best_score = -1e9

        self.logger = logging.getLogger("Early Stopping")

    def update(self, score):
        if score > self.best_score:
            self.strikes = 0
            self.logger.info(
                f"Registered score of {score} which is higher than previous best {self.best_score}"
            )
            self.best_score = score
        else:
            self.strikes += 1
            if self.strikes >= self.patience:
                self.callback_fn()
                self.logger.info("Early stopping triggered. ")


class ScoreMonitor:
    def __init__(self, mode, verbose=True):
        assert mode in ["min", "max"], f"Only modes `min` and `max` are supported."
        self.mode = mode
        self.best = -1e9 if mode == "max" else 1e9
        self.verbose = verbose

    def condition(self, old_score, new_score):
        if self.mode == "max":
            return new_score > old_score
        else:
            return new_score < old_score

    def __call__(self, new_value):
        """Updates the value and returns True if this is the best score"""
        if self.condition(self.best, new_value):
            if self.verbose:
                logging.getLogger(str(self.__class__)).info(
                    f"Current score {new_value:.2f} better than previous {self.best:.2f}"
                )

            self.best = new_value
            return True
        else:
            return False


def patch_out_to_core_out(patch_pred_out: Dict):
    """
    Converts the dictionary of batch outputs of patchwise predictions to dictionary of
    core-wise predictions.

    input:
        patch_pred_out(Dict) - dictionary of tensors and numpy arrays containing at least the keys
            `core specifier` and `preds`.

    returns:
        dictionary of tensors containing core predictions aggregated from patches within the core,
        as well as the other

    """

    core_out = {k: [] for k in patch_pred_out}

    for core_specifier in np.unique(patch_pred_out["core_specifier"]):

        ind = np.where(patch_pred_out["core_specifier"] == core_specifier)[0]

        for k, v in patch_pred_out.items():
            if k == "preds":
                continue

            core_out[k].append(v[ind][0])
            # global _has_been_warned
            # if not np.all(v[ind][0] == v[ind]) and not _has_been_warned:
            #    warnings.warn(
            #        f"Found multiple values for key `{k}` which aggregating patches into cores. Choosing one from an arbitrary patch."
            #    )
            #    _has_been_warned = True
        #
        core_out["preds"].append(patch_pred_out["preds"][ind].mean(0))

    for k, v in core_out.items():
        core_out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else np.stack(v)

    return core_out


def exclude_low_inv(
    prediction_output,
    threshold=0.4,
):
    """
    Excludes low involvement from the prediction output.
    args:
        prediction_output - output dictionary with at least
        the key "pct_cancer"
        threshold - the threshold below which outputs will be
            dropped.
    """
    inv = prediction_output["pct_cancer"]
    inv = np.nan_to_num(inv)
    ind_high_inv = np.where(inv > threshold)[0]
    high_inv = {}
    for k, v in prediction_output.items():
        high_inv[k] = v[ind_high_inv]
    return high_inv


def group_output_by_centers(prediction_output):
    """
    Groups predictions into a dictionary of prediction outputs, one for each center
    args:
        prediction_output - dictionary of outputs with at least the key "center".
    """

    outs = {}

    for center in np.unique(prediction_output["center"]):
        ind_for_center = np.where(prediction_output["center"] == center)[0]

        outs_for_center = {}
        for k, v in prediction_output.items():
            outs_for_center[k] = v[ind_for_center]

        outs[center] = outs_for_center

    return outs


def add_prefix(dict: Dict, prefix: str, sep: str = "_"):
    return {f"{prefix}{sep}{k}": v for k, v in dict.items()}


def compute_center_and_macro_metrics(out, metrics_fn: Callable):
    """
    Computes breakdown of metrics across different centers.
    args:
        out: dictionary of outputs with at least the keys "center"
        metrics_fn: function which computes metrics on dictionaries
    """
    metrics = {}
    micro_metrics = metrics_fn(out)
    metric_keys = micro_metrics.keys()
    metrics.update(add_prefix(micro_metrics, "micro_avg"))
    macro_avg_tracker = {}
    for center, out_ in group_output_by_centers(out).items():
        metrics_for_center = metrics_fn(out_)
        for k, v in metrics_for_center.items():
            macro_avg_tracker.setdefault(k, []).append(v)
        metrics.update(add_prefix(metrics_for_center, center))

    for k, v in macro_avg_tracker.items():
        metrics[f"macro_avg_{k}"] = sum(v) / len(v)

    return metrics


def apply_metrics_to_patch_and_core_output(patch_pred_out, metrics_fn: Callable):
    """
    Computes metrics on a patch and core basis:

    args:
        patch_pred_out(Dict) - dictionary of tensors and numpy arrays containing at least the keys
            `core specifier` and `preds`.

        metrics_fn: the function that will be applied to the patch output and core output
            dictionaries.
    """
    metrics = {}
    metrics.update(add_prefix(metrics_fn(patch_pred_out), "patch"))
    core_out = patch_out_to_core_out(patch_pred_out)
    metrics.update(add_prefix(metrics_fn(core_out), "core"))
    return metrics


def patch_and_core_metrics(patch_out, base_metrics: Callable):
    """
    Given the dictionary of patch output, computes patch metrics,
    core metrics, and center_wise metrics.
    """
    metrics = {}
    metrics.update(
        add_prefix(compute_center_and_macro_metrics(patch_out, base_metrics), "patch")
    )
    core_out = patch_out_to_core_out(patch_out)
    metrics.update(
        add_prefix(compute_center_and_macro_metrics(core_out, base_metrics), "core")
    )
    return metrics


def acc_micro_manual(out, t=0.5):
    preds = out["preds"][:, 1]
    labels = out["labels"]
    score = ((preds > t).long() == labels).float().mean()
    return score


def acc_macro_manual(out, t=0.5):
    preds = out["preds"][:, 1]
    labels = out["labels"]
    tr = torch.where(labels == 1)[0]
    f = torch.where(labels == 0)[0]
    preds = (preds > t).long()
    correct = preds == labels
    return (correct[tr].float().mean() + correct[f].float().mean()) / 2


def class_scores_manual(out, t=0.5):
    preds = (out["preds"][:, 1] > t).long()
    labels = out["labels"]

    def _sum_and_eq(pred, true):
        return torch.logical_and(preds == pred, labels == true).sum()

    return {
        "tp": _sum_and_eq(1, 1),
        "fp": _sum_and_eq(1, 0),
        "tn": _sum_and_eq(0, 0),
        "fn": _sum_and_eq(0, 1),
    }


def precision(out, t=0.5):
    scores = class_scores_manual(out, t)
    return (scores["tp"]) / (scores["tp"] + scores["fp"])


def specificity(out, t=0.5):
    scores = class_scores_manual(out, t)
    return (scores["tn"]) / (scores["tn"] + scores["fp"])


def recall(out, t=0.5):
    scores = class_scores_manual(out, t)
    return (scores["tp"]) / (scores["tp"] + scores["fn"])


def spec_at_fixed_sens(out, sens=0.85, return_t=False):
    thresholds = np.linspace(0, 1, 20)
    sensitivities = np.array([recall(out, t) for t in thresholds])
    closest_index = np.argmin(np.abs(sensitivities - sens))
    closest_threshold = thresholds[closest_index]
    out = specificity(out, closest_threshold)
    if return_t:
        return out, closest_threshold
    else:
        return out


def f1(out, t=0.5):
    r = recall(out, t)
    p = precision(out, t)
    return 2 * p * r / (p + r)


def avg_prec(out):
    from torchmetrics.functional import average_precision

    return average_precision(out["preds"], out["labels"], num_classes=2)


def positive_guess_ratio(out):
    return out["preds"][:, 1].mean()


def find_best_threshold(score_func: Callable):
    scores = []
    ts = np.linspace(0, 1, 20)
    for t in ts:
        scores.append(score_func(t))

    scores = np.array(scores)

    # ignore nans in calculation
    ind_notnan = ~np.isnan(scores)
    scores = scores[ind_notnan]
    ts = ts[ind_notnan]

    return ts[np.argmax(scores)]


def binary_auroc(out):
    from torchmetrics.functional import auroc

    return auroc(out["preds"], out["labels"], num_classes=2)

def brier_score(out):
    y_pred = out["preds"]
    y_true = out["labels"].numpy()
    return brier_score_loss(y_true, out["preds"][:, 1])

def neg_log_likelihood(out):
    y_pred = out["preds"]
    y_true = out["labels"].numpy()
    return log_loss(y_true, out["preds"][:, 1])

def expected_calibration_error(out, num_bins=10):

    y_pred = out["preds"]
    y_true = out["labels"]

    metric=CalibrationError(num_bins=num_bins, norm='l1')

    return metric(y_pred, y_true)


from torchmetrics.functional import auroc, accuracy, average_precision


class PatchwiseAndCorewiseMetrics:

    SUPPORTED_METRICS = [
        "auroc",
        "avg_prec",
        "acc_macro",
        "acc_micro",
        "spe_at_85_sens",
        "positive_guess_ratio",
        "brier_score",
        "expected_calibration_error",
        "neg_log_likelihood"
    ]

    def __init__(
        self,
        compute_corewise_and_patchwise=True,
        breakdown_by_center=True,
        metrics: list = ["auroc"],
    ):
        self.compute_corewise_and_patchwise = compute_corewise_and_patchwise
        self.breakdown_by_center = breakdown_by_center
        self.metrics = metrics
        assert all([metric in self.SUPPORTED_METRICS for metric in self.metrics])

    def __call__(self, out):
        return self._compute_metrics(out)

    def _compute_metrics(self, out):
        out["preds"] = out["logits"].softmax(-1)

        if self.compute_corewise_and_patchwise:
            if self.breakdown_by_center:
                base_metrics = self._center_breakdown_metrics
            else:
                base_metrics = self._base_metrics
            return apply_metrics_to_patch_and_core_output(out, base_metrics)

        else:
            if self.breakdown_by_center:
                return self._center_breakdown_metrics(out)
            else:
                return self._base_metrics(out)

    def _center_breakdown_metrics(self, out):
        return compute_center_and_macro_metrics(out, self._base_metrics)

    def _base_metrics(self, out):
        metrics = {}
        if "auroc" in self.metrics:
            metrics["auroc"] = auroc(out["preds"], out["labels"], num_classes=2)
        if "acc_macro" in self.metrics:
            metrics["macro_acc"] = accuracy(
                out["preds"], out["labels"], average="macro", num_classes=2
            )
        if "acc_micro" in self.metrics:
            metrics["micro_acc"] = accuracy(
                out["preds"], out["labels"], average="micro"
            )
        if "avg_prec" in self.metrics:
            metrics["avg_prec"] = average_precision(
                out["logits"], out["labels"], num_classes=2
            )
        if "spe_at_85_sens" in self.metrics:
            metrics["spec_at_85_sens"] = spec_at_fixed_sens(
                out,
            )
        if "positive_guess_ratio" in self.metrics:
            metrics["positive_guess_ratio"] = positive_guess_ratio(out)

        if "brier_score" in self.metrics:
            metrics["brier_score"] = brier_score(out)

        if "expected_calibration_error" in self.metrics:
            metrics["expected_calibration_error"] = expected_calibration_error(out)

        if "neg_log_likelihood" in self.metrics:
            metrics["neg_log_likelihood"] = neg_log_likelihood(out)

        return metrics


def get_correct_table(df, threshold=0.5):
    """Makes a table whose rows are patients, columns are locations, and values are whether the model predicted correctly for that patient/location"""

    all_locations = df["loc"].unique()

    table = pd.DataFrame(columns=all_locations)

    for patient, sub_df in df.groupby("patient_specifier"):
        for core, sub_sub_df in sub_df.groupby("core_specifier"):
            loc = sub_sub_df["loc"].unique()[0]
            pred = int(sub_sub_df["prob_1"].mean() > threshold)
            label = sub_sub_df["y"].unique()[0]
            correct = pred == label

            table.loc[patient, loc] = correct

    return table


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


def reliability_diagram(preds, confidence, targets, n_bins=10, ax=None):

    # bar plot of confidence vs. accuracy
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()

    val, outs = expected_calibration_error(preds, confidence, targets, n_bins=n_bins)

    for i in range(n_bins):
        plt.bar(
            (outs["bins"][i] + outs["bins"][i + 1]) / 2,
            outs["bins"][i + 1],
            width=1 / n_bins,
            color="#F3B6B6",
            linewidth=0.2,
            edgecolor="black",
        )
        if i > 0:
            plt.bar(
                (outs["bins"][i - 1] + outs["bins"][i]) / 2,
                outs["acc_by_bin"][i],
                width=1 / n_bins,
                color="#0000F7",
                linewidth=0.8,
                edgecolor="black",
            )

    plt.plot(
        outs["bins"][:-1],
        outs["bins"][:-1],
        color="black",
        linewidth=0.8,
        linestyle="--",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")


def acc_reject_curve(
    preds: np.ndarray, confidence: np.ndarray, targets: np.ndarray, score_fn=None
):
    """If specified, score_fn should accept args (y_true, y_pred) and return a float.
    Returns:
        x: percentage of samples rejected
        y: score when computed on remaining samples
    """
    if score_fn is None:
        from sklearn.metrics import accuracy_score

        score_fn = accuracy_score

    num_samples = len(preds)
    ind = confidence.argsort()
    preds = preds[ind]
    confidence = confidence[ind]
    targets = targets[ind]

    increment = int(num_samples / 100)

    x = []
    y = []
    while True:
        if len(preds) < increment:
            break
        x.append(len(preds) / num_samples)
        y.append(score_fn(targets, preds))
        preds = preds[increment:]
        confidence = confidence[increment:]
        targets = targets[increment:]

    x = np.array(x)
    y = np.array(y)

    return (1 - x) * 100, y


def brier_score(probs, targets, weighted=True):

    brier = (probs - targets) ** 2
    ind_pos = targets == 1
    ind_neg = targets == 0

    if weighted:
        brier[ind_pos] *= ind_neg.sum() / ind_pos.sum()

    return np.mean(brier)