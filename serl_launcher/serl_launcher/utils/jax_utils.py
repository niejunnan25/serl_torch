import random
from typing import Any, Dict

import numpy as np
import torch


def _to_torch(x, device=None):
    if isinstance(x, dict):
        return {k: _to_torch(v, device=device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(x)
    if t.dtype == torch.float64:
        t = t.float()
    return t.to(device) if device is not None else t


def batch_to_torch(batch, device=None):
    return _to_torch(batch, device=device)


def batch_to_jax(batch):
    return batch_to_torch(batch)


class TorchRNG:
    """Stateful torch RNG wrapper used as a drop-in replacement for JaxRNG."""

    @classmethod
    def from_seed(cls, seed):
        return cls(int(seed))

    def __init__(self, seed: int):
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def _next_seed(self):
        return self._rng.randint(0, 2**31 - 1)

    def __call__(self, keys=None):
        if keys is None:
            return self._next_seed()
        if isinstance(keys, int):
            return tuple(self._next_seed() for _ in range(keys))
        return {key: self._next_seed() for key in keys}


JaxRNG = TorchRNG


def wrap_function_with_rng(rng):
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            split_rng = rng()
            return function(split_rng, *args, **kwargs)

        return wrapped

    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = TorchRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)
