from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.runtime.cadence import EnvStepCadenceTracker


class _FakeTrainerSession:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, str | None]] = []

    def update(
        self,
        *,
        context: str,
        failure_message: str | None = None,
    ) -> bool:
        self.update_calls.append((str(context), failure_message))
        return True


class _FailingTrainerSession:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, str | None]] = []

    def update(
        self,
        *,
        context: str,
        failure_message: str | None = None,
    ) -> bool:
        self.update_calls.append((str(context), failure_message))
        raise RuntimeError("boom")


class _BlockingTrainerSession:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, str | None]] = []
        self.update_until_success_calls: list[str] = []

    def update(
        self,
        *,
        context: str,
        failure_message: str | None = None,
    ) -> bool:
        self.update_calls.append((str(context), failure_message))
        raise RuntimeError("cadence should prefer update_until_success")

    def update_until_success(
        self,
        *,
        context: str,
    ) -> bool:
        self.update_until_success_calls.append(str(context))
        return True


def test_env_step_cadence_tracker_advances_boundaries_across_chunks() -> None:
    trainer_session = _FakeTrainerSession()
    tracker = EnvStepCadenceTracker(steps_per_update=5, log_period=4)

    assert (
        tracker.advance(
            env_steps_after_chunk=3,
            trainer_session=trainer_session,
            update_context_prefix="env_step",
            failure_message="update failed",
        )
        is False
    )
    assert trainer_session.update_calls == []

    assert (
        tracker.advance(
            env_steps_after_chunk=6,
            trainer_session=trainer_session,
            update_context_prefix="env_step",
            failure_message="update failed",
        )
        is True
    )
    assert trainer_session.update_calls == [("env_step_5", "update failed")]

    assert (
        tracker.advance(
            env_steps_after_chunk=8,
            trainer_session=trainer_session,
            update_context_prefix="env_step",
            failure_message="update failed",
        )
        is True
    )
    assert trainer_session.update_calls == [("env_step_5", "update failed")]


def test_env_step_cadence_tracker_handles_multiple_updates_in_one_chunk() -> None:
    trainer_session = _FakeTrainerSession()
    tracker = EnvStepCadenceTracker(steps_per_update=3, log_period=10)

    assert (
        tracker.advance(
            env_steps_after_chunk=10,
            trainer_session=trainer_session,
            update_context_prefix="processor_commit_step",
            failure_message="processor update failed",
        )
        is True
    )
    assert trainer_session.update_calls == [
        ("processor_commit_step_3", "processor update failed"),
        ("processor_commit_step_6", "processor update failed"),
        ("processor_commit_step_9", "processor update failed"),
    ]


def test_env_step_cadence_tracker_prefers_blocking_update_when_available() -> None:
    trainer_session = _BlockingTrainerSession()
    tracker = EnvStepCadenceTracker(steps_per_update=5, log_period=10)

    assert (
        tracker.advance(
            env_steps_after_chunk=6,
            trainer_session=trainer_session,
            update_context_prefix="env_step",
            failure_message="legacy failure message",
        )
        is False
    )
    assert trainer_session.update_until_success_calls == ["env_step_5"]
    assert trainer_session.update_calls == []


def test_env_step_cadence_tracker_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="steps_per_update must be positive"):
        EnvStepCadenceTracker(steps_per_update=0, log_period=10)

    with pytest.raises(ValueError, match="log_period must be positive"):
        EnvStepCadenceTracker(steps_per_update=10, log_period=0)


def test_env_step_cadence_tracker_does_not_advance_boundary_when_update_raises() -> None:
    trainer_session = _FailingTrainerSession()
    tracker = EnvStepCadenceTracker(steps_per_update=5, log_period=4)

    with pytest.raises(RuntimeError, match="boom"):
        tracker.advance(
            env_steps_after_chunk=6,
            trainer_session=trainer_session,
            update_context_prefix="env_step",
            failure_message="update failed",
        )

    assert trainer_session.update_calls == [("env_step_5", "update failed")]
    assert tracker.next_update_step == 5
    assert tracker.next_log_step == 4
