import itertools
from abc import ABC, abstractmethod
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch


class Accumulator(ABC):
    @abstractmethod
    def __call__(self, value):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def compute(self):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.compute()})"


class Average(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.sum += value
        self.count += 1
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def compute(self):
        return self.sum / self.count


class ExponentialMovingAverage(Accumulator):
    def __init__(self, alpha=0.9):
        self.reset()
        self.alpha = alpha

    def __call__(self, value):
        self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class MovingAverage(Accumulator):
    def __init__(self, window_size=10):
        self.reset()
        self.window_size = window_size

    def __call__(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []

    def compute(self):
        return sum(self.values) / len(self.values)


class Max(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.value = max(self.value, value)
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class Sum(Accumulator):
    def __init__(self):
        self.reset()

    def __call__(self, value):
        self.value += value
        return self.value

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value


class DictConcatenation(Accumulator):
    def __init__(self):
        self.reset()

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

    def compute(self, out_fmt: Literal["dict", "dataframe"] = "dict"):
        out = {}
        for k, v in self._data.items():
            out[k] = (
                torch.concat(v)
                if isinstance(v[0], torch.Tensor)
                else list(itertools.chain(*v))
            )

        for k, v in out.items():
            if isinstance(v, list):
                out[k] = np.array(v)

        if out_fmt == "dict":
            return out

        else:
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

    def reset(self):
        self._data = {}


class DataFrameCollector(DictConcatenation):
    def compute(self):
        return super().compute(out_fmt="dataframe")
