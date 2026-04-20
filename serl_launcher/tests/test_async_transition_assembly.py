from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_PARENT = Path(__file__).resolve().parents[2]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_launcher.rollout.async_transition_assembly import (
    AsyncTransitionAssemblyCoordinator,
)


class AsyncTransitionAssemblyCoordinatorTest(unittest.TestCase):
    def test_submit_chunk_backfills_full_observation_list(self) -> None:
        seen_observations: list[list[int]] = []
        coordinator = AsyncTransitionAssemblyCoordinator(
            backfill_fn=lambda observations, task_prompt: (
                seen_observations.append(list(observations)) or [
                    f"{task_prompt}:{value}" for value in observations
                ]
            ),
            build_result_fn=lambda raw, backfilled_values: {
                "raw_id": int(raw["id"]),
                "values": list(backfilled_values),
            },
            thread_name_prefix="test-full-backfill",
        )

        seq = coordinator.submit_chunk(
            raw={"id": 7},
            observations=[1, 2, 3],
            task_prompt="task",
        )

        assembled = coordinator.pop_committable(block_until_seq=seq)

        self.assertEqual(seen_observations, [[1, 2, 3]])
        self.assertEqual(
            assembled,
            [{"raw_id": 7, "values": ["task:1", "task:2", "task:3"]}],
        )
        coordinator.close()

    def test_blocking_commit_preserves_submit_order(self) -> None:
        coordinator = AsyncTransitionAssemblyCoordinator(
            backfill_fn=lambda observations, task_prompt: [
                f"{task_prompt}:{value}" for value in observations
            ],
            build_result_fn=lambda raw, backfilled_values: {
                "raw_id": int(raw["id"]),
                "values": list(backfilled_values),
            },
            thread_name_prefix="test-order",
        )

        first_seq = coordinator.submit_chunk(
            raw={"id": 1},
            observations=[10],
            task_prompt="task",
        )
        second_seq = coordinator.submit_chunk(
            raw={"id": 2},
            observations=[20],
            task_prompt="task",
        )

        assembled = coordinator.pop_committable(block_until_seq=second_seq)

        self.assertEqual(first_seq, 0)
        self.assertEqual(second_seq, 1)
        self.assertEqual(
            assembled,
            [
                {"raw_id": 1, "values": ["task:10"]},
                {"raw_id": 2, "values": ["task:20"]},
            ],
        )
        coordinator.close()

    def test_close_invokes_close_fn(self) -> None:
        close_calls = {"count": 0}

        coordinator = AsyncTransitionAssemblyCoordinator(
            backfill_fn=lambda observations, task_prompt: [],
            build_result_fn=lambda raw, backfilled_values: None,
            thread_name_prefix="test-close",
            close_fn=lambda: close_calls.__setitem__("count", close_calls["count"] + 1),
        )

        coordinator.close()

        self.assertEqual(close_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
