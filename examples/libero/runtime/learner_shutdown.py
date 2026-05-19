from __future__ import annotations

"""Learner shutdown predicates for LIBERO residual training."""

from typing import Any
from typing import Mapping


ACTOR_DONE_DATA_COMMITTED_STOP_REASON = "actor_done_data_committed"


def actor_done_data_committed(
    *,
    actor_done: bool,
    target_env_steps: int,
    target_data_steps: int | None = None,
    latest_data_id: int,
    transport_status: Mapping[str, Any] | None = None,
    require_transport_commit: bool = True,
) -> bool:
    """Return True when actor rollout is done and all actor data is visible."""

    if not bool(actor_done):
        return False

    if target_data_steps is None:
        target_env_steps = int(target_env_steps)
        if target_env_steps <= 0:
            return False
        target_data_steps = int(target_env_steps)
    else:
        target_data_steps = int(target_data_steps)
        if target_data_steps < 0:
            return False
        if target_data_steps == 0:
            return True

    if int(latest_data_id) < target_data_steps:
        return False

    if not bool(require_transport_commit):
        return True

    status = {} if transport_status is None else dict(transport_status)
    accepted = int(status.get("accepted_update_id", -1))
    committed = int(status.get("committed_update_id", -1))
    target_last_data_id = max(0, target_data_steps - 1)
    if accepted < target_last_data_id or committed < target_last_data_id:
        return False
    if accepted >= 0 and committed >= 0 and accepted > committed:
        return False
    return True


__all__ = [
    "ACTOR_DONE_DATA_COMMITTED_STOP_REASON",
    "actor_done_data_committed",
]
