from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
