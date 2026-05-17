from __future__ import annotations

import logging
import sys
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.rollout.processor_dispatch import build_processor_submission_payload
from serl_launcher.rollout.processor_dispatch import QueuedProcessorSubmitter


class _FakeProcessorClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def wait_until_ready(self, *, timeout_s: float, poll_interval_s: float) -> None:
        self.requests.append(
            (
                "ready",
                {
                    "timeout_s": float(timeout_s),
                    "poll_interval_s": float(poll_interval_s),
                },
            )
        )

    def submit(self, *, payload: dict[str, object], context: str) -> dict[str, object]:
        self.requests.append(("submit", {"payload": dict(payload), "context": context}))
        return {}

    def finish(self, *, episode_id: int, last_chunk_seq: int | None) -> dict[str, object]:
        self.requests.append(
            (
                "finish",
                {
                    "episode_id": int(episode_id),
                    "last_chunk_seq": last_chunk_seq,
                },
            )
        )
        return {}

    def mark_episode_end(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
        actor_progress: dict[str, object] | None = None,
        rollout_stats: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append(
            (
                "mark_episode_end",
                {
                    "episode_id": int(episode_id),
                    "last_chunk_seq": last_chunk_seq,
                    "actor_progress": actor_progress,
                    "rollout_stats": rollout_stats,
                },
            )
        )
        return {}

    def shutdown(self, *, last_chunk_seq: int | None) -> dict[str, object]:
        self.requests.append(("shutdown", {"last_chunk_seq": last_chunk_seq}))
        return {}

    def last_status(self) -> dict[str, object]:
        return {"ok": True}

    def close(self) -> None:
        self.closed = True


class ProcessorDispatchTest(unittest.TestCase):
    def test_payload_builder_preserves_chunk_identity(self) -> None:
        payload = build_processor_submission_payload(
            chunk_seq=3,
            episode_id=4,
            episode_step_start=5,
            task_prompt="pick",
            chunk_result={"reward_sum": 1.0},
        )

        self.assertEqual(payload["chunk_seq"], 3)
        self.assertEqual(payload["episode_id"], 4)
        self.assertEqual(payload["episode_step_start"], 5)
        self.assertEqual(payload["task_prompt"], "pick")

    def test_queued_submitter_sends_commands_in_order(self) -> None:
        client = _FakeProcessorClient()
        submitter = QueuedProcessorSubmitter(
            processor_client=client,
            logger=logging.getLogger(__name__),
            thread_name="test-agibot-processor-submit",
        )

        submitter.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)
        submitter.submit_rollout_chunk(
            episode_id=1,
            chunk_seq=0,
            episode_step_start=0,
            task_prompt="task",
            chunk_result={"reward_sum": 1.0},
        )
        submitter.mark_episode_end(
            episode_id=1,
            last_chunk_seq=0,
            rollout_stats={"env_steps": 1},
        )
        submitter.shutdown(last_chunk_seq=0)
        submitter.close(wait=True)

        kinds = [kind for kind, _payload in client.requests]
        self.assertEqual(
            kinds,
            ["ready", "submit", "mark_episode_end", "shutdown"],
        )
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
