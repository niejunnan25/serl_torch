from __future__ import annotations

import logging
from typing import Any

import pytest

from serl_launcher.common.trainer_session import TrainerClientSession


class _FakeTrainerClient:
    def __init__(self) -> None:
        self.update_results: list[bool] = []
        self.request_results: dict[str, list[dict[str, Any] | None]] = {}
        self.status_payload: dict[str, Any] = {"transport_mode": "async_commit"}
        self.raise_status = False
        self.wait_until_committed_result = True
        self.wait_until_committed_calls = 0

    def update(self) -> bool:
        if self.update_results:
            return bool(self.update_results.pop(0))
        return True

    def request(self, type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        del payload
        queue = self.request_results.get(str(type), [])
        if queue:
            return queue.pop(0)
        return {}

    def get_transport_status(self, store_name: str | None = None) -> dict[str, Any]:
        del store_name
        if self.raise_status:
            raise RuntimeError("status unavailable")
        return dict(self.status_payload)

    def wait_until_committed(self) -> bool:
        self.wait_until_committed_calls += 1
        return bool(self.wait_until_committed_result)


def test_trainer_session_status_uses_fallback() -> None:
    client = _FakeTrainerClient()
    client.raise_status = True
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
        store_name="actor_env",
        status_fallback=lambda: {"transport_mode": "fallback"},
    )
    assert session.status() == {"transport_mode": "fallback"}


def test_trainer_session_update_raises_after_threshold() -> None:
    client = _FakeTrainerClient()
    client.update_results = [False, False]
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    assert session.update(context="step_1", max_failures=2) is False
    with pytest.raises(RuntimeError, match="trainer transport update failed repeatedly"):
        session.update(context="step_2", max_failures=2)


def test_trainer_session_request_optional_does_not_raise() -> None:
    client = _FakeTrainerClient()
    client.request_results = {"send-stats": [None, {"ok": True}]}
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    assert (
        session.request(
            "send-stats",
            {"value": 1},
            raise_on_exhaustion=False,
        )
        is None
    )
    assert session.request("send-stats", {"value": 2}) == {"ok": True}


def test_trainer_session_request_raises_after_threshold() -> None:
    client = _FakeTrainerClient()
    client.request_results = {"send-stats": [None, None]}
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    assert session.request("send-stats", {"value": 1}, max_failures=2) is None
    with pytest.raises(RuntimeError, match="trainer transport send-stats failed repeatedly"):
        session.request("send-stats", {"value": 2}, max_failures=2)


def test_trainer_session_request_default_threshold_raises_after_five_failures() -> None:
    client = _FakeTrainerClient()
    client.request_results = {"send-stats": [None, None, None, None, None]}
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    for attempt in range(4):
        assert session.request("send-stats", {"value": attempt}) is None
    with pytest.raises(RuntimeError, match="trainer transport send-stats failed repeatedly"):
        session.request("send-stats", {"value": 4})


def test_trainer_session_request_retries_until_success_when_requested() -> None:
    client = _FakeTrainerClient()
    client.request_results = {"actor-progress": [None, None, {"ok": True}]}
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    assert session.request(
        "actor-progress",
        {"env_steps": 10},
        retry_until_exhausted=True,
        max_failures=3,
        retry_sleep_s=0.0,
    ) == {"ok": True}


def test_trainer_session_flush_waits_for_commit() -> None:
    client = _FakeTrainerClient()
    client.update_results = [True]
    client.wait_until_committed_result = False
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    with pytest.raises(RuntimeError, match="wait_until_committed timed out"):
        session.flush(context="shutdown", wait_until_committed=True)


def test_trainer_session_flush_raises_when_update_did_not_succeed() -> None:
    client = _FakeTrainerClient()
    client.update_results = [False]
    session = TrainerClientSession(
        client=client,
        logger=logging.getLogger(__name__),
    )
    with pytest.raises(RuntimeError, match="update did not succeed during flush"):
        session.flush(context="episode_end", wait_until_committed=True)
    assert client.wait_until_committed_calls == 0
