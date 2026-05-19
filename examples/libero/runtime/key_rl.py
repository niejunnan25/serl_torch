from __future__ import annotations

"""Runtime helpers for fixed-step and fixed-stage gated residual RL."""

from typing import Any

StageRange = tuple[int, int | None]

_STAGE_NAMES = ("stage1", "stage2", "stage3")


def key_rl_enabled(key_rl_cfg: Any) -> bool:
    return bool(getattr(key_rl_cfg, "enabled", False))


def key_rl_mode(key_rl_cfg: Any) -> str:
    return str(getattr(key_rl_cfg, "mode", "fixed_step"))


def key_rl_start_step(key_rl_cfg: Any) -> int:
    return int(getattr(key_rl_cfg, "start_step", 0))


def key_rl_stage_ranges(key_rl_cfg: Any) -> tuple[StageRange, ...]:
    ranges: list[StageRange] = []
    for stage_name in _STAGE_NAMES:
        stage_cfg = getattr(key_rl_cfg, stage_name, None)
        if stage_cfg is None or not bool(getattr(stage_cfg, "enabled", False)):
            continue
        ranges.append(
            (
                int(getattr(stage_cfg, "start_step", 0)),
                int(getattr(stage_cfg, "end_step", 0)),
            )
        )
    return tuple(ranges)


def key_rl_active_step_ranges(key_rl_cfg: Any) -> tuple[StageRange, ...] | None:
    if not key_rl_active_only_replay(key_rl_cfg):
        return None
    mode = key_rl_mode(key_rl_cfg)
    if mode == "fixed_step":
        return ((key_rl_start_step(key_rl_cfg), None),)
    if mode == "fixed_stages":
        return key_rl_stage_ranges(key_rl_cfg)
    raise ValueError(f"key_rl.mode currently supports only fixed_step/fixed_stages")


def key_rl_step_in_ranges(
    episode_step: int,
    active_step_ranges: tuple[StageRange, ...] | None,
) -> bool:
    if active_step_ranges is None:
        return True
    step = int(episode_step)
    for start_step, end_step in active_step_ranges:
        if step < int(start_step):
            continue
        if end_step is not None and step >= int(end_step):
            continue
        return True
    return False


def key_rl_active(key_rl_cfg: Any, *, episode_step: int) -> bool:
    if not key_rl_enabled(key_rl_cfg):
        return True
    mode = key_rl_mode(key_rl_cfg)
    if mode == "fixed_step":
        return int(episode_step) >= key_rl_start_step(key_rl_cfg)
    if mode == "fixed_stages":
        return key_rl_step_in_ranges(
            int(episode_step),
            key_rl_stage_ranges(key_rl_cfg),
        )
    raise ValueError(f"key_rl.mode currently supports only fixed_step/fixed_stages")


def key_rl_active_only_replay(key_rl_cfg: Any) -> bool:
    return key_rl_enabled(key_rl_cfg) and str(
        getattr(key_rl_cfg, "replay_mode", "active_only")
    ) == "active_only"


def key_rl_min_replay_episode_step(key_rl_cfg: Any) -> int | None:
    if not key_rl_active_only_replay(key_rl_cfg):
        return None
    if key_rl_mode(key_rl_cfg) != "fixed_step":
        return None
    return key_rl_start_step(key_rl_cfg)


def _stage_progress_payload(key_rl_cfg: Any) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for stage_name in _STAGE_NAMES:
        stage_cfg = getattr(key_rl_cfg, stage_name, None)
        payload[stage_name] = {
            "enabled": bool(
                False if stage_cfg is None else getattr(stage_cfg, "enabled", False)
            ),
            "start_step": int(
                0 if stage_cfg is None else getattr(stage_cfg, "start_step", 0)
            ),
            "end_step": int(
                0 if stage_cfg is None else getattr(stage_cfg, "end_step", 0)
            ),
        }
    return payload


def _validate_stage(
    *,
    stage_cfg: Any,
    stage_name: str,
    chunk_horizon: int,
    require_chunk_boundary: bool,
    context: str,
) -> None:
    if stage_cfg is None or not bool(getattr(stage_cfg, "enabled", False)):
        return
    start_step = int(getattr(stage_cfg, "start_step", 0))
    end_step = int(getattr(stage_cfg, "end_step", 0))
    if start_step < 0:
        raise ValueError(f"{context}.{stage_name}.start_step must be >= 0")
    if end_step <= start_step:
        raise ValueError(
            f"{context}.{stage_name} must satisfy start_step < end_step: "
            f"start_step={start_step} end_step={end_step}"
        )
    if bool(require_chunk_boundary):
        for field_name, value in (
            ("start_step", start_step),
            ("end_step", end_step),
        ):
            if int(value) % int(chunk_horizon) != 0:
                raise ValueError(
                    f"{context}.{stage_name}.{field_name} must align to "
                    f"chunk_horizon when {context}.require_chunk_boundary=true: "
                    f"{field_name}={int(value)} chunk_horizon={int(chunk_horizon)}"
                )


def validate_key_rl_chunk_boundary(
    key_rl_cfg: Any,
    *,
    chunk_horizon: int,
    context: str = "key_rl",
) -> None:
    if not key_rl_enabled(key_rl_cfg):
        return
    mode = key_rl_mode(key_rl_cfg)
    if mode not in {"fixed_step", "fixed_stages"}:
        raise ValueError(f"{context}.mode currently supports only fixed_step/fixed_stages")
    if str(getattr(key_rl_cfg, "replay_mode", "active_only")) != "active_only":
        raise ValueError(f"{context}.replay_mode currently supports only active_only")
    horizon = int(chunk_horizon)
    if horizon <= 0:
        raise ValueError(f"chunk_horizon must be positive, got {horizon}")
    require_chunk_boundary = bool(
        getattr(key_rl_cfg, "require_chunk_boundary", True)
    )
    if mode == "fixed_step":
        start_step = key_rl_start_step(key_rl_cfg)
        if start_step < 0:
            raise ValueError(f"{context}.start_step must be >= 0, got {start_step}")
        if require_chunk_boundary and start_step % horizon != 0:
            raise ValueError(
                f"{context}.start_step must align to chunk_horizon when "
                f"{context}.require_chunk_boundary=true: start_step={start_step} "
                f"chunk_horizon={horizon}"
            )
        return

    for stage_name in _STAGE_NAMES:
        _validate_stage(
            stage_cfg=getattr(key_rl_cfg, stage_name, None),
            stage_name=stage_name,
            chunk_horizon=horizon,
            require_chunk_boundary=require_chunk_boundary,
            context=context,
        )


def build_key_rl_progress(
    *,
    key_rl_cfg: Any,
    env_steps: int,
    active_steps: int | None = None,
    replay_inserted_steps: int,
    skipped_base_only_steps: int,
) -> dict[str, Any]:
    resolved_active_steps = (
        int(replay_inserted_steps) if active_steps is None else int(active_steps)
    )
    resolved_active_steps = max(0, int(resolved_active_steps))
    inserted_steps = max(0, int(replay_inserted_steps))
    total_env_steps = max(0, int(env_steps))
    skipped_steps = max(0, int(skipped_base_only_steps))
    return {
        "enabled": bool(key_rl_enabled(key_rl_cfg)),
        "mode": key_rl_mode(key_rl_cfg),
        "start_step": int(key_rl_start_step(key_rl_cfg)),
        "stages": _stage_progress_payload(key_rl_cfg),
        "active_step_ranges": [
            [int(start), None if end is None else int(end)]
            for start, end in (key_rl_active_step_ranges(key_rl_cfg) or ())
        ],
        "replay_mode": str(getattr(key_rl_cfg, "replay_mode", "active_only")),
        "require_chunk_boundary": bool(
            getattr(key_rl_cfg, "require_chunk_boundary", True)
        ),
        "active_steps": int(resolved_active_steps),
        "replay_inserted_steps": int(inserted_steps),
        "skipped_base_only_steps": int(skipped_steps),
        "active_ratio": (
            float(resolved_active_steps) / float(total_env_steps)
            if total_env_steps > 0
            else 0.0
        ),
    }


__all__ = [
    "StageRange",
    "build_key_rl_progress",
    "key_rl_active",
    "key_rl_active_only_replay",
    "key_rl_active_step_ranges",
    "key_rl_enabled",
    "key_rl_min_replay_episode_step",
    "key_rl_mode",
    "key_rl_stage_ranges",
    "key_rl_start_step",
    "key_rl_step_in_ranges",
    "validate_key_rl_chunk_boundary",
]
