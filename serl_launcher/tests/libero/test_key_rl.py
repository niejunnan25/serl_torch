from __future__ import annotations

from types import SimpleNamespace

import pytest

from serl_torch.examples.libero.runtime.key_rl import key_rl_active
from serl_torch.examples.libero.runtime.key_rl import key_rl_active_step_ranges
from serl_torch.examples.libero.runtime.key_rl import key_rl_min_replay_episode_step
from serl_torch.examples.libero.runtime.key_rl import validate_key_rl_chunk_boundary


def _stage(
    *,
    enabled: bool = False,
    start_step: int = 0,
    end_step: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        start_step=start_step,
        end_step=end_step,
    )


def _cfg(**overrides: object) -> SimpleNamespace:
    values = {
        "enabled": False,
        "mode": "fixed_step",
        "start_step": 0,
        "replay_mode": "active_only",
        "require_chunk_boundary": True,
        "stage1": _stage(),
        "stage2": _stage(),
        "stage3": _stage(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_key_rl_is_always_active_and_has_no_filter() -> None:
    cfg = _cfg(enabled=False, start_step=70)

    assert key_rl_active(cfg, episode_step=0)
    assert key_rl_active(cfg, episode_step=69)
    assert key_rl_min_replay_episode_step(cfg) is None


def test_fixed_step_key_rl_switches_on_at_start_step() -> None:
    cfg = _cfg(enabled=True, start_step=70)

    assert not key_rl_active(cfg, episode_step=69)
    assert key_rl_active(cfg, episode_step=70)
    assert key_rl_min_replay_episode_step(cfg) == 70
    assert key_rl_active_step_ranges(cfg) == ((70, None),)


def test_fixed_stages_single_window_uses_half_open_interval() -> None:
    cfg = _cfg(
        enabled=True,
        mode="fixed_stages",
        stage1=_stage(enabled=True, start_step=30, end_step=75),
    )

    assert not key_rl_active(cfg, episode_step=29)
    assert key_rl_active(cfg, episode_step=30)
    assert key_rl_active(cfg, episode_step=74)
    assert not key_rl_active(cfg, episode_step=75)
    assert key_rl_min_replay_episode_step(cfg) is None
    assert key_rl_active_step_ranges(cfg) == ((30, 75),)


def test_fixed_stages_supports_multiple_independent_windows() -> None:
    cfg = _cfg(
        enabled=True,
        mode="fixed_stages",
        stage1=_stage(enabled=True, start_step=30, end_step=75),
        stage2=_stage(enabled=True, start_step=110, end_step=160),
        stage3=_stage(enabled=True, start_step=180, end_step=200),
    )

    assert key_rl_active(cfg, episode_step=30)
    assert not key_rl_active(cfg, episode_step=90)
    assert key_rl_active(cfg, episode_step=120)
    assert not key_rl_active(cfg, episode_step=170)
    assert key_rl_active(cfg, episode_step=190)
    assert key_rl_active_step_ranges(cfg) == (
        (30, 75),
        (110, 160),
        (180, 200),
    )


def test_fixed_stages_all_disabled_is_base_only() -> None:
    cfg = _cfg(enabled=True, mode="fixed_stages")

    assert not key_rl_active(cfg, episode_step=0)
    assert not key_rl_active(cfg, episode_step=10_000)
    assert key_rl_active_step_ranges(cfg) == ()


def test_chunk_boundary_validation_rejects_misaligned_start() -> None:
    cfg = _cfg(enabled=True, start_step=72, require_chunk_boundary=True)

    with pytest.raises(ValueError, match="must align to chunk_horizon"):
        validate_key_rl_chunk_boundary(cfg, chunk_horizon=5)


def test_chunk_boundary_validation_can_be_disabled() -> None:
    cfg = _cfg(enabled=True, start_step=72, require_chunk_boundary=False)

    validate_key_rl_chunk_boundary(cfg, chunk_horizon=5)


def test_stage_validation_rejects_invalid_or_misaligned_window() -> None:
    invalid_order = _cfg(
        enabled=True,
        mode="fixed_stages",
        stage1=_stage(enabled=True, start_step=75, end_step=75),
    )
    with pytest.raises(ValueError, match="start_step < end_step"):
        validate_key_rl_chunk_boundary(invalid_order, chunk_horizon=5)

    misaligned = _cfg(
        enabled=True,
        mode="fixed_stages",
        stage1=_stage(enabled=True, start_step=31, end_step=75),
    )
    with pytest.raises(ValueError, match="must align to chunk_horizon"):
        validate_key_rl_chunk_boundary(misaligned, chunk_horizon=5)


def test_stage_boundary_validation_can_be_disabled() -> None:
    cfg = _cfg(
        enabled=True,
        mode="fixed_stages",
        require_chunk_boundary=False,
        stage1=_stage(enabled=True, start_step=31, end_step=76),
    )

    validate_key_rl_chunk_boundary(cfg, chunk_horizon=5)
