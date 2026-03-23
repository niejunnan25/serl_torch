"""Training schedule helpers."""
from __future__ import annotations

from omegaconf import DictConfig


def _scheduled_xi(cfg: DictConfig, base_xi: float, global_policy_step: int) -> float:
    sched_cfg = cfg.training.get("xi_scheduler", None)
    if sched_cfg is None or (not bool(sched_cfg.get("enabled", False))):
        return float(base_xi)

    min_xi = float(sched_cfg.get("min_xi", base_xi))
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    anneal_steps = int(sched_cfg.get("anneal_steps", 1))
    if global_policy_step < warmup_steps:
        return float(min_xi)
    if anneal_steps <= 0:
        return float(base_xi)
    progress = min(1.0, max(0.0, (global_policy_step - warmup_steps) / float(anneal_steps)))
    return float(min_xi + (float(base_xi) - min_xi) * progress)
