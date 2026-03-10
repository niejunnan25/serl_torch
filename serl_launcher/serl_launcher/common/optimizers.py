import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    clip_grad_norm: Optional[float]


def _make_lr_lambda(
    warmup_steps: int,
    cosine_decay_steps: Optional[int],
):
    warmup_steps = max(0, int(warmup_steps))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if cosine_decay_steps is None:
            return 1.0

        post_warmup_step = max(0, step - warmup_steps)
        decay_steps = max(1, int(cosine_decay_steps))
        progress = min(1.0, post_warmup_step / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lr_lambda


def make_optimizer(
    parameters,
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cosine_decay_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    return_lr_schedule: bool = False,
):
    optimizer_cls = torch.optim.AdamW if weight_decay is not None else torch.optim.Adam
    optimizer = optimizer_cls(
        parameters,
        lr=learning_rate,
        weight_decay=0.0 if weight_decay is None else weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(warmup_steps=warmup_steps, cosine_decay_steps=cosine_decay_steps),
    )

    if return_lr_schedule:
        return OptimizerBundle(
            optimizer=optimizer,
            scheduler=scheduler,
            clip_grad_norm=clip_grad_norm,
        ), scheduler

    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_norm=clip_grad_norm,
    )
