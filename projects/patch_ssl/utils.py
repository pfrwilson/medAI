import torch 
from typing import Sequence, Literal
import numpy as np
import pandas as pd
import itertools
from torch import nn 
import logging
from dataclasses import dataclass, field
import wandb
import matplotlib.pyplot as plt
import typing as tp
import os
import sys 
from torch import distributed as dist
import submitit
import copy 
import numpy as np 
from typing import overload


class DictConcatenation:
    def __init__(self, auto_reset=True, dist_mode: bool = False):
        self.reset()
        self.auto_reset = auto_reset
        self.dist_mode = dist_mode

    def __call__(self, data_dict):

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
            elif isinstance(v, np.ndarray):
                pass
            elif not isinstance(v, Sequence):
                v = [v]
            self._data.setdefault(k, []).append(v)

    def update(self, data_dict):
        self(data_dict)

    def _aggregate(self):
        out = {}
        for k, v in self._data.items():
            out[k] = self._concat_any(v)

        for k, v in out.items():
            if isinstance(v, list):
                out[k] = np.array(v)

        if torch.distributed.is_initialized() and self.dist_mode:
            # we need to gather concatenate accross all processes
            for key, value in out.items():
                obj_list = [value for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, value)
                out[key] = self._concat_any(obj_list)
                torch.distributed.barrier()
        return out

    def _concat_any(self, obj_list): 
        if isinstance(obj_list[0], torch.Tensor): 
            return torch.cat(obj_list, dim=0)
        elif isinstance(obj_list[0], np.ndarray): 
            return np.concatenate(obj_list, axis=0)
        else:
            l = list(itertools.chain(*obj_list))
            return np.array(l)

    def dict_to_df(self, data_dict):
        out = data_dict
        out_new = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray) and v.ndim == 2:
                for i in range(v.shape[1]):
                    out_new[f"{k}_{i}"] = v[:, i]
            else:
                out_new[k] = v
        return pd.DataFrame(out_new)

    def compute(self, out_fmt: Literal["dict", "dataframe"] = "dict"):
        out = self._aggregate()
        if out_fmt == "dataframe":
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    out = self.dict_to_df(out)
                else: 
                    out = None
            else: 
                out = self.dict_to_df(out)
        return out

    def reset(self):
        self._data = {}


class DictConcatenatorNew: 
    def __init__(self): 
        self.reset()

    def reset(self): 
        self._data = {}

    def update(self, data: dict): 
        
        # cast everything to numpy
        for key, value in data.items(): 
            data[key] = self.to_numpy(value)
        
        # check that all data have the same number of rows and are at most 2D
        nrows = None 
        for key, value in data.items(): 
            if nrows is None:
                nrows = value.shape[0]
            else: 
                assert nrows == value.shape[0], f"Data with key {key} has {value.shape[0]} rows, but expected {nrows} rows."
        
        if torch.distributed.is_initialized():
            # we need to gather concatenate accross all processes
            for key, value in self._data.items():
                obj_list = [value for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, value)
                self._data[key] = np.concatenate(obj_list, axis=0)
                torch.distributed.barrier()

        # append data to internal dict
        for key, value in data.items():
            if key not in self._data: 
                self._data[key] = value 
            else: 
                self._data[key] = np.concatenate([self._data[key], value], axis=0) 

    def collect(self): 
        out = copy.deepcopy(self._data)
        self.reset()
        return out

    @property
    def df(self): 
        out = self.collect()
        out_new = {}
        for k, v in out.items():
            if v.ndim == 2:
                for i in range(v.shape[1]):
                    out_new[f"{k}_{i}"] = v[:, i]
            elif v.ndim == 1: 
                out_new[k] = v
            else: 
                raise ValueError(f"Data with key {k} has {v.ndim} dimensions, but expected 1 or 2.")
        return pd.DataFrame(out_new)

    def to_numpy(self, object): 
        if isinstance(object, torch.Tensor): 
            object = object.detach().cpu()
            object = object.numpy()
        elif not isinstance(object, np.ndarray): 
            object = np.array(object)
        return object
    

class DataFrameCollector(DictConcatenation):
    def compute(self):
        return super().compute(out_fmt="dataframe")
    
    @property
    def df(self): 
        return self.compute()
    

class TemperatureCalibration(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1)) 
    
    def fit(self, logits, targets, lr=1e-2, max_iter=10000, balance_classes=True, verbose=True): 
        from torch.nn import functional as F
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)

        if balance_classes: 
            weight = torch.ones(logits.shape[-1], device=logits.device)
            for i in range(logits.shape[-1]):
                n_targets = (targets == i).sum()
                n_non_targets = (targets != i).sum()
                weight[i] = n_non_targets / n_targets
        else: 
            weight = None

        def closure():
            optimizer.zero_grad()
            output = self(logits)
            loss = F.cross_entropy(output, targets, weight=weight)
            loss.backward()
            return loss
        
        loss_init = closure()
        optimizer.step(closure)
        loss_final = closure()

        # check that loss is decreasing
        if loss_final >= loss_init: 
            logging.info("Loss did not decrease during calibration. Consider increasing the number of iterations.")
            self.__init__()

        if verbose: 
            logging.getLogger(__name__).info(f"Loss before calibration: {loss_init:.3f}")
            logging.getLogger(__name__).info(f"Loss after calibration: {loss_final:.3f}")
            logging.getLogger(__name__).info(f"Temperature: {self.temperature.item():.3f}, Bias: {self.bias.item():.3f}")

    def forward(self, logits): 
        return logits / self.temperature + self.bias
    

def slurm_checkpoint_dir(): 
    import os 
    if 'SLURM_JOB_ID' not in os.environ: 
        return None
    return os.path.join(
        '/checkpoint', os.environ['USER'], os.environ['SLURM_JOB_ID']
    )


def add_prefix_to_keys(d, prefix, sep='/'):
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def make_corewise_df(df):
    corewise_df = df.groupby(["core_specifier"]).prob_1.mean().reset_index()
    corewise_df["label"] = (
        df.groupby(["core_specifier"]).label.first().reset_index().label.astype(int)
    )
    return corewise_df


def compute_metrics(df, threshold=None):
    from sklearn.metrics import roc_auc_score, recall_score, balanced_accuracy_score
    from trusnet.utils.metrics import brier_score, expected_calibration_error

    if threshold is None:
        threshold = compute_optimal_threshold(df.prob_1, df.label)

    auc = roc_auc_score(df.label, df.prob_1)
    sensitivity = recall_score(df.label, df.prob_1 > threshold)
    specificity = recall_score(1 - df.label, df.prob_1 < threshold)
    acc_balanced = balanced_accuracy_score(df.label, df.prob_1 > threshold)

    metrics = {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "acc_balanced": acc_balanced,
    }

    return metrics


def generate_experiment_name(): 
    from coolname import generate_slug
    from datetime import datetime
    return f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-{generate_slug(2)}'


def basic_experiment_setup(exp_dir, group=None, config_dict=None, wandb_project=None, resume=True, debug=False): 
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(os.path.join(exp_dir, 'checkpoints')): 
        if slurm_checkpoint_dir() is not None: 
            # sym link slurm checkpoints dir to local checkpoints dir
            os.symlink(slurm_checkpoint_dir(), os.path.join(exp_dir, 'checkpoints'))
        else: 
            os.makedirs(os.path.join(exp_dir, 'checkpoints'))
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')

    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(exp_dir, 'out.log'))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler], force=True)
    # also log tracebacks with excepthook
    def excepthook(type, value, tb):
        logging.error("Uncaught exception: {0}".format(str(value)))
        import traceback
        traceback.print_tb(tb, file=open(os.path.join(exp_dir, 'out.log'), 'a'))
        logging.error(f'Exception type: {type}')
        sys.__excepthook__(type, value, tb)
    sys.excepthook = excepthook

    import json 
    if config_dict is not None: 
        json.dump(config_dict, open(os.path.join(exp_dir, 'config.json'), 'w'), indent=4)

    if resume and 'wandb_id' in os.listdir(exp_dir): 
        wandb_id = open(os.path.join(exp_dir, 'wandb_id')).read().strip()
        logging.info(f'Resuming wandb run {wandb_id}')
    else: 
        wandb_id = wandb.util.generate_id()
        open(os.path.join(exp_dir, 'wandb_id'), 'w').write(wandb_id)

    wandb.init(
        project=wandb_project if not debug else f'{wandb_project}-debug',
        group=group,
        config=config_dict,
        resume='allow',
        name=os.path.basename(exp_dir),
        id=wandb_id,
        dir=ckpt_dir,
    )


def basic_ddp_experiment_setup(exp_dir, group=None, config_dict=None, wandb_project=None, resume=True, debug=False):
    dist_env = submitit.helpers.TorchDistributedEnvironment().export()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)    
    
    # setup logging for each process
    file_handler = logging.FileHandler(os.path.join(exp_dir, f'out_rank{dist_env.rank}.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stdout_handler], force=True)
    # also log tracebacks with excepthook
    def excepthook(type, value, tb):
        logging.error("Uncaught exception: {0}".format(str(value)))
        import traceback
        traceback.print_tb(tb, file=open(os.path.join(exp_dir, 'out.log'), 'a'))
        logging.error(f'Exception type: {type}')
        sys.__excepthook__(type, value, tb)
    sys.excepthook = excepthook

    logging.info(f"master: {dist_env.master_addr}:{dist_env.master_port}")
    logging.info(f"rank: {dist_env.rank}")
    logging.info(f"world size: {dist_env.world_size}")
    logging.info(f"local rank: {dist_env.local_rank}")
    logging.info(f"local world size: {dist_env.local_world_size}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logging.info(f"{torch.cuda.is_available()=}")
    logging.info(f"{torch.cuda.device_count()=}")
    logging.info(f"{torch.cuda.current_device()=}")
    logging.info(f"{torch.cuda.get_device_name()=}")
    logging.info(f"{torch.cuda.get_device_capability()=}")
    logging.info('initializing process group')
    # Using the (default) env:// initialization method
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    logging.info('process group initialized')
    
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')

    if dist_env.rank == 0:
        if not os.path.exists(os.path.join(exp_dir, 'checkpoints')): 
            if slurm_checkpoint_dir() is not None: 
                # sym link slurm checkpoints dir to local checkpoints dir
                os.symlink(slurm_checkpoint_dir(), os.path.join(exp_dir, 'checkpoints'))
            else: 
                os.makedirs(os.path.join(exp_dir, 'checkpoints'))

        import json 
        if config_dict is not None: 
            json.dump(config_dict, open(os.path.join(exp_dir, 'config.json'), 'w'), indent=4)

        if resume and 'wandb_id' in os.listdir(exp_dir): 
            wandb_id = open(os.path.join(exp_dir, 'wandb_id')).read().strip()
            logging.info(f'Resuming wandb run {wandb_id}')
        else: 
            wandb_id = wandb.util.generate_id()
            open(os.path.join(exp_dir, 'wandb_id'), 'w').write(wandb_id)

        wandb.init(
            project=wandb_project if not debug else 'debug',
            group=group,
            config=config_dict,
            resume='allow',
            name=os.path.basename(exp_dir),
            id=wandb_id,
            dir=ckpt_dir,
        )
        logging.info(f'wandb run: {wandb.run.name}')
        logging.info(f'wandb dir: {wandb.run.dir}')
        logging.info(f'wandb id: {wandb.run.id}')
        logging.info(f'wandb url: {wandb.run.get_url()}')

    dist.barrier()
    return ckpt_dir, dist_env


def create_epoch_report(train_df, val_df, test_df=None): 
    """Creates a report for a single epoch.
    input dataframes should have the following columns:
    - label: 0 or 1
    - prob_1: probability of class 1
    - core_specifier: identifier for the core 
    """

    train_metrics = {}
    t1 = compute_optimal_threshold(train_df.prob_1, train_df.label)
    train_metrics.update(add_prefix_to_keys(compute_metrics(train_df, t1), 'patchwise', sep='_'))
    train_df_corewise = make_corewise_df(train_df)
    t2 = compute_optimal_threshold(train_df_corewise.prob_1, train_df_corewise.label)
    train_metrics.update(add_prefix_to_keys(compute_metrics(train_df_corewise, t2), 'corewise', sep='_'))

    val_metrics = {}
    t3 = compute_optimal_threshold(val_df.prob_1, val_df.label)
    val_metrics.update(add_prefix_to_keys(compute_metrics(val_df, t3), 'patchwise', sep='_'))
    val_df_corewise = make_corewise_df(val_df)
    t4 = compute_optimal_threshold(val_df_corewise.prob_1, val_df_corewise.label)
    val_metrics.update(add_prefix_to_keys(compute_metrics(val_df_corewise, t4), 'corewise', sep='_'))

    if test_df is not None:
        test_metrics = {}
        t5 = t3
        test_metrics.update(add_prefix_to_keys(compute_metrics(test_df, t5), 'patchwise', sep='_'))
        test_df_corewise = make_corewise_df(test_df)
        t6 = t4
        test_metrics.update(add_prefix_to_keys(compute_metrics(test_df_corewise, t6), 'corewise', sep='_'))

    # make probability histograms
    fig_train, ax_train = plt.subplots(1, 2, figsize=(10, 5))
    train_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_train[0])
    ax_train[0].axvline(t1, linestyle="--", color="red")
    train_df_corewise.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_train[1])
    ax_train[1].axvline(t2, linestyle="--", color="red")
    ax_train[0].set_title("Train Patchwise")
    ax_train[1].set_title("Train Corewise")
    train_metrics['histogram'] = wandb.Image(fig_train)

    fig_val, ax_val = plt.subplots(1, 2, figsize=(10, 5))
    val_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_val[0])
    ax_val[0].axvline(t3, linestyle="--", color="red")
    val_df_corewise.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_val[1])
    ax_val[1].axvline(t4, linestyle="--", color="red")
    ax_val[0].set_title("Val Patchwise")
    ax_val[1].set_title("Val Corewise")
    val_metrics['histogram'] = wandb.Image(fig_val)

    if test_df is not None:
        fig_test, ax_test = plt.subplots(1, 2, figsize=(10, 5))
        test_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_test[0])
        ax_test[0].axvline(t5, linestyle="--", color="red")
        test_df_corewise.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax_test[1])
        ax_test[1].axvline(t6, linestyle="--", color="red")
        ax_test[0].set_title("Test Patchwise")
        ax_test[1].set_title("Test Corewise")
        test_metrics['histogram'] = wandb.Image(fig_test)
        test_metrics['confusion_matrix'] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_df_corewise.label,
            preds=test_df_corewise.prob_1 > t6,
            class_names=["benign", "cancer"],
        )

    all_metrics = {}
    all_metrics.update(add_prefix_to_keys(train_metrics, 'train'))
    all_metrics.update(add_prefix_to_keys(val_metrics, 'val'))
    if test_df is not None:
        all_metrics.update(add_prefix_to_keys(test_metrics, 'test'))

    return all_metrics


def corewise_epoch_report(train_df, val_df, test_df=None, threshold=None): 

    metrics = {}
    t1 = threshold or compute_optimal_threshold(train_df.prob_1, train_df.label)
    train_metrics = compute_metrics(train_df, t1)
    metrics.update(add_prefix_to_keys(train_metrics, 'train', sep='/'))
    t2 = threshold or compute_optimal_threshold(val_df.prob_1, val_df.label)
    val_metrics = compute_metrics(val_df, t2)
    metrics.update(add_prefix_to_keys(val_metrics, 'val', sep='/'))
    if test_df is not None:
        test_metrics = compute_metrics(test_df, t2)
        metrics.update(add_prefix_to_keys(test_metrics, 'test', sep='/'))

    fig, ax = plt.subplots(1, 3 if test_df is not None else 2, figsize=(15 if test_df is not None else 10, 5))
    train_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax[0])
    ax[0].axvline(t1, linestyle="--", color="red")
    ax[0].set_title("Train")
    val_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax[1])
    ax[1].axvline(t2, linestyle="--", color="red")
    ax[1].set_title("Val")
    if test_df is not None:
        test_df.groupby(["label"]).prob_1.hist(bins=100, alpha=0.5, density=True, ax=ax[2])
        ax[2].axvline(t2, linestyle="--", color="red")
        ax[2].set_title("Test")
    metrics['histogram'] = wandb.Image(fig)

    if test_df is not None:
        metrics['confusion_matrix'] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_df.label,
            preds=test_df.prob_1 > t2,
            class_names=["benign", "cancer"],
        )
    
    return metrics


def compute_optimal_threshold(probs, labels):
    """
    Computes the optimal threshold for a given set of probabilities and labels.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


class NullScheduler:
    """
    Dummy lr scheduler that does nothing but still supports the same interface as a real scheduler.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.improvement = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            if self.verbose:
                logging.info(f'EarlyStopping - no improvement: {self.counter}/{self.patience}')
            self.improvement = False
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose: 
                    logging.info(f'EarlyStopping - patience reached: {self.counter}/{self.patience}')
                self.early_stop = True
        else:
            if self.verbose:
                logging.info(f'Early stopping - score improved from {self.best_score:.4f} to {score:.4f}')
            self.improvement = True
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'improvement': self.improvement,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.improvement = state_dict['improvement']
