from __future__ import annotations

from pathlib import Path
import sys

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.runtime.learner_shutdown import (
    actor_done_data_committed,
)


def _transport_status(*, accepted: int, committed: int) -> dict[str, int]:
    return {
        "accepted_update_id": int(accepted),
        "committed_update_id": int(committed),
    }


def test_actor_not_done_does_not_stop() -> None:
    assert not actor_done_data_committed(
        actor_done=False,
        target_env_steps=100,
        latest_data_id=100,
        transport_status=_transport_status(accepted=99, committed=99),
    )


def test_actor_done_requires_replay_to_receive_final_step() -> None:
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=99,
        transport_status=_transport_status(accepted=99, committed=99),
    )


def test_actor_done_requires_transport_accept_and_commit_to_cover_final_step() -> None:
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=100,
        transport_status=_transport_status(accepted=98, committed=98),
    )
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=100,
        transport_status=_transport_status(accepted=99, committed=98),
    )
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=100,
        transport_status=_transport_status(accepted=100, committed=99),
    )


def test_actor_done_stops_without_update_step_dependency() -> None:
    assert actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=100,
        transport_status=_transport_status(accepted=99, committed=99),
    )


def test_legacy_direct_datastore_can_stop_without_transport_ids() -> None:
    assert actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        latest_data_id=100,
        transport_status={"transport_mode": "legacy_agentlace"},
        require_transport_commit=False,
    )


def test_invalid_target_env_steps_do_not_stop() -> None:
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=0,
        latest_data_id=0,
        transport_status=_transport_status(accepted=-1, committed=-1),
    )


def test_actor_done_uses_explicit_target_data_steps_for_gated_replay() -> None:
    assert actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        target_data_steps=30,
        latest_data_id=30,
        transport_status=_transport_status(accepted=29, committed=29),
    )
    assert not actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        target_data_steps=30,
        latest_data_id=29,
        transport_status=_transport_status(accepted=29, committed=29),
    )


def test_actor_done_with_zero_target_data_steps_can_stop() -> None:
    assert actor_done_data_committed(
        actor_done=True,
        target_env_steps=100,
        target_data_steps=0,
        latest_data_id=0,
        transport_status=_transport_status(accepted=-1, committed=-1),
    )
