from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


class MMDataParallel(torch.nn.DataParallel):
    pass


@dataclass
class DataContainer:
    data: Any
    cpu_only: bool = False


def collate(batch, samples_per_gpu=1):
    if not batch:
        return batch
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], DataContainer):
            result[key] = DataContainer(
                [value.data for value in values],
                cpu_only=values[0].cpu_only)
        elif isinstance(values[0], (torch.Tensor, np.ndarray)):
            result[key] = default_collate(values)
        else:
            result[key] = values
    return result


def auto_fp16(*_args, **_kwargs):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped

    return decorator
