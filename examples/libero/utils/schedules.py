"""Training schedule helpers."""
from __future__ import annotations

from omegaconf import DictConfig

from .alpha_utils import validate_alpha


def _scheduled_alpha(cfg: DictConfig, base_alpha: float, schedule_step: int) -> float:
    base_alpha = validate_alpha(base_alpha, name="base_alpha", allow_zero=True)
    sched_cfg = cfg.training.get("alpha_scheduler", None)
    if sched_cfg is None or (not bool(sched_cfg.get("enabled", False))):
        return float(base_alpha)

    min_alpha = validate_alpha(
        sched_cfg.get("min_alpha", base_alpha),
        name="training.alpha_scheduler.min_alpha",
        allow_zero=True,
    )
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    anneal_steps = int(sched_cfg.get("anneal_steps", 1))
    if schedule_step < warmup_steps:
        return float(min_alpha)
    if anneal_steps <= 0:
        return float(base_alpha)
    progress = min(1.0, max(0.0, (schedule_step - warmup_steps) / float(anneal_steps)))
    return float(min_alpha + (float(base_alpha) - min_alpha) * progress)
