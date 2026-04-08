"""Training schedule helpers."""
from __future__ import annotations

from omegaconf import DictConfig

from serl_launcher.utils.alpha_utils import validate_alpha


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


def _residual_epsilon_gating_cfg(cfg: DictConfig):
    residual_cfg = cfg.get("residual", None)
    if residual_cfg is None:
        return None
    gating_cfg = residual_cfg.get("epsilon_gating", None)
    return gating_cfg


def _epsilon_gating_enabled(cfg: DictConfig) -> bool:
    gating_cfg = _residual_epsilon_gating_cfg(cfg)
    return bool(gating_cfg is not None and gating_cfg.get("enabled", False))


def _epsilon_gating_clock(cfg: DictConfig) -> str:
    gating_cfg = _residual_epsilon_gating_cfg(cfg)
    clock = str("env_step" if gating_cfg is None else gating_cfg.get("clock", "env_step"))
    clock = clock.strip().lower()
    if clock not in {"env_step", "decision_step"}:
        raise ValueError(
            "residual.epsilon_gating.clock must be one of "
            "['env_step', 'decision_step'], "
            f"got {clock!r}"
        )
    return clock


def _epsilon_gating_eval_force_on(cfg: DictConfig) -> bool:
    gating_cfg = _residual_epsilon_gating_cfg(cfg)
    if gating_cfg is None:
        return True
    return bool(gating_cfg.get("eval_force_on", True))


def _scheduled_epsilon_gating_probability(
    cfg: DictConfig, *, schedule_step: int
) -> float:
    gating_cfg = _residual_epsilon_gating_cfg(cfg)
    if gating_cfg is None or (not bool(gating_cfg.get("enabled", False))):
        return 1.0

    schedule_type = str(gating_cfg.get("schedule", "linear")).strip().lower()
    if schedule_type not in {"linear", "constant"}:
        raise ValueError(
            "residual.epsilon_gating.schedule must be one of "
            "['linear', 'constant'], "
            f"got {schedule_type!r}"
        )

    min_prob = float(gating_cfg.get("min_prob", 0.0))
    max_prob = float(gating_cfg.get("max_prob", 1.0))
    min_prob = float(min(1.0, max(0.0, min_prob)))
    max_prob = float(min(1.0, max(0.0, max_prob)))
    if max_prob < min_prob:
        min_prob, max_prob = max_prob, min_prob

    if schedule_type == "constant":
        return float(max_prob)

    warmup_steps = int(gating_cfg.get("warmup_steps", 0))
    ramp_steps = int(gating_cfg.get("ramp_steps", 1))
    if schedule_step < warmup_steps:
        return float(min_prob)
    if ramp_steps <= 0:
        return float(max_prob)
    progress = min(1.0, max(0.0, (schedule_step - warmup_steps) / float(ramp_steps)))
    return float(min_prob + (max_prob - min_prob) * progress)
