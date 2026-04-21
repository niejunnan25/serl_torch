from __future__ import annotations

import logging
from pathlib import Path
import sys
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.runtime.processor_dispatch import (  # noqa: E402
    QueuedProcessorSubmitter,
)


class _FakeProcessorClient:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._last_status: dict[str, object] = {}

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.calls.append(("wait_until_ready", timeout_s, poll_interval_s))

    def submit(self, *, payload: dict[str, object], context: str) -> dict[str, object]:
        self.calls.append(("submit", payload["chunk_seq"], context))
        self._last_status = {"accepted_chunk_seq": int(payload["chunk_seq"])}
        return dict(self._last_status)

    def finish(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
    ) -> dict[str, object]:
        self.calls.append(("finish", episode_id, last_chunk_seq))
        self._last_status = {
            "episode_id": int(episode_id),
            "processed_chunk_seq": int(-1 if last_chunk_seq is None else last_chunk_seq),
        }
        return dict(self._last_status)

    def mark_episode_end(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
        actor_progress: dict[str, object] | None = None,
        rollout_stats: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "mark_episode_end",
                episode_id,
                last_chunk_seq,
                actor_progress,
                rollout_stats,
            )
        )
        self._last_status = {
            "episode_id": int(episode_id),
            "pending_episode_flushes": 1,
        }
        return dict(self._last_status)

    def shutdown(self, *, last_chunk_seq: int | None) -> dict[str, object]:
        self.calls.append(("shutdown", last_chunk_seq))
        self._last_status = {
            "stop_requested": True,
            "processed_chunk_seq": int(-1 if last_chunk_seq is None else last_chunk_seq),
        }
        return dict(self._last_status)

    def last_status(self) -> dict[str, object]:
        return dict(self._last_status)

    def close(self) -> None:
        self.calls.append(("close",))


class QueuedProcessorSubmitterTest(unittest.TestCase):
    def test_submitter_processes_commands_in_order(self) -> None:
        client = _FakeProcessorClient()
        submitter = QueuedProcessorSubmitter(
            processor_client=client,
            logger=logging.getLogger(__name__),
        )

        submitter.wait_until_ready(timeout_s=1.5, poll_interval_s=0.05)
        submitter.submit_chunk(payload={"chunk_seq": 3}, context="chunk_3")
        submitter.mark_episode_end(
            episode_id=7,
            last_chunk_seq=3,
            actor_progress={"env_steps": 11},
            rollout_stats={"env_steps": 11, "rollout": {"episode_id": 7}},
        )
        submitter.shutdown(last_chunk_seq=3)
        submitter.close()

        self.assertEqual(
            client.calls,
            [
                ("wait_until_ready", 1.5, 0.05),
                ("submit", 3, "chunk_3"),
                (
                    "mark_episode_end",
                    7,
                    3,
                    {"env_steps": 11},
                    {"env_steps": 11, "rollout": {"episode_id": 7}},
                ),
                ("shutdown", 3),
                ("close",),
            ],
        )

    def test_status_snapshot_reports_queue_and_last_status(self) -> None:
        client = _FakeProcessorClient()
        submitter = QueuedProcessorSubmitter(
            processor_client=client,
            logger=logging.getLogger(__name__),
        )
        submitter.submit_chunk(payload={"chunk_seq": 5}, context="chunk_5")
        submitter.close()

        snapshot = submitter.status_snapshot()
        self.assertEqual(snapshot["outbox_depth"], 0)
        self.assertFalse(snapshot["sender_failed"])
        self.assertEqual(snapshot["processor"]["accepted_chunk_seq"], 5)


if __name__ == "__main__":
    unittest.main()
