from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from serl_torch.examples.libero.runtime.stage_reward_model import (
    StageRewardEpisodeState,
)
from serl_torch.examples.libero.runtime.stage_reward_model import (
    apply_stage_reward_scores,
)


@dataclass(frozen=True)
class _RawChunk:
    episode_step_start: int
    rewards: list[float]
    reward_sum: float
    executed_steps: int


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


def _key_cfg() -> SimpleNamespace:
    return SimpleNamespace(stage1=_stage(enabled=True, start_step=30, end_step=35))


def _rm_cfg(**overrides: object) -> SimpleNamespace:
    values = {
        "enabled": True,
        "threshold": 0.8,
        "bonus": 0.5,
        "stage": "stage1",
        "one_shot": True,
        "apply_only_when_key_rl_active": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage_reward_bonus_is_added_once_to_triggering_transition() -> None:
    raw = _RawChunk(
        episode_step_start=30,
        rewards=[0.0, 0.0, 0.0],
        reward_sum=0.0,
        executed_steps=3,
    )
    state = StageRewardEpisodeState()

    shaped, stats = apply_stage_reward_scores(
        raw_chunk=raw,
        scores=[0.1, 0.9, 0.95],
        reward_model_cfg=_rm_cfg(),
        key_rl_cfg=_key_cfg(),
        episode_state=state,
        key_active=True,
    )

    assert shaped.rewards == [0.0, 0.5, 0.0]
    assert shaped.reward_sum == 0.5
    assert stats.triggered
    assert stats.trigger_step == 31
    assert state.bonus_given
    assert state.trigger_step == 31
    assert state.bonus_sum == 0.5

    shaped_again, stats_again = apply_stage_reward_scores(
        raw_chunk=raw,
        scores=[0.95, 0.95, 0.95],
        reward_model_cfg=_rm_cfg(),
        key_rl_cfg=_key_cfg(),
        episode_state=state,
        key_active=True,
    )

    assert shaped_again is raw
    assert stats_again.bonus_sum == 0.0
    assert state.bonus_sum == 0.5


def test_stage_reward_respects_stage_window_end() -> None:
    raw = _RawChunk(
        episode_step_start=34,
        rewards=[0.0, 0.0],
        reward_sum=0.0,
        executed_steps=2,
    )

    shaped, stats = apply_stage_reward_scores(
        raw_chunk=raw,
        scores=[0.9, 0.9],
        reward_model_cfg=_rm_cfg(one_shot=False),
        key_rl_cfg=_key_cfg(),
        episode_state=StageRewardEpisodeState(),
        key_active=True,
    )

    assert shaped.rewards == [0.5, 0.0]
    assert stats.trigger_step == 34
    assert stats.bonuses == (0.5, 0.0)


def test_stage_reward_skips_key_inactive_chunks() -> None:
    raw = _RawChunk(
        episode_step_start=30,
        rewards=[0.0],
        reward_sum=0.0,
        executed_steps=1,
    )

    shaped, stats = apply_stage_reward_scores(
        raw_chunk=raw,
        scores=[0.95],
        reward_model_cfg=_rm_cfg(),
        key_rl_cfg=_key_cfg(),
        episode_state=StageRewardEpisodeState(),
        key_active=False,
    )

    assert shaped is raw
    assert stats.bonus_sum == 0.0
