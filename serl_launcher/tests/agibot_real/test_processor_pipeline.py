from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.agibot_real.runtime.processor_pipeline import (
        AgiBotRolloutProcessor,
    )
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        AssemblyResult,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    AgiBotRolloutProcessor = object  # type: ignore[assignment]
    AssemblyResult = object  # type: ignore[assignment]


class _FakeDataStore:
    def __init__(self) -> None:
        self.inserted: list[dict[str, object]] = []

    def insert(self, transition: dict[str, object]) -> None:
        self.inserted.append(dict(transition))


class _FakeAssembler:
    async_transition_assembly_enabled = False

    def __init__(self) -> None:
        self.closed = False

    def handle_chunk(self, *, raw, task_prompt: str):
        del task_prompt
        return [
            AssemblyResult(
                transitions=[
                    {
                        "episode_id": int(raw.episode_id),
                        "episode_step": int(raw.episode_step_start),
                        "observations": dict(raw.residual_obs_before_chunk),
                        "actions": np.asarray(raw.action_chunk[0], dtype=np.float32),
                        "next_observations": {"state": np.asarray([1.0])},
                        "rewards": float(raw.reward_sum),
                        "masks": 0.0,
                        "dones": True,
                    }
                ],
                prefetched=None,
                next_obs=dict(raw.final_obs),
                episode_done=True,
                env_steps_delta=int(raw.executed_steps),
                episode_steps_delta=int(raw.executed_steps),
                episode_return_delta=float(raw.reward_sum),
                episode_success=True,
                last_info=dict(raw.chunk_info),
            )
        ]

    def drain_ready(self):
        return []

    def finish_episode(self, *, block: bool = True):
        del block
        return []

    def close(self) -> None:
        self.closed = True


class _FakePendingAssembler(_FakeAssembler):
    def handle_chunk(self, *, raw, task_prompt: str):
        del task_prompt
        return [
            AssemblyResult(
                transitions=[
                    {
                        "episode_id": int(raw.episode_id),
                        "episode_step": int(raw.episode_step_start),
                        "observations": dict(raw.residual_obs_before_chunk),
                        "actions": np.asarray(raw.action_chunk[0], dtype=np.float32),
                        "next_observations": {"state": np.asarray([1.0])},
                        "rewards": float(raw.reward_sum),
                        "masks": 1.0,
                        "dones": False,
                    }
                ],
                prefetched=None,
                next_obs=dict(raw.final_obs),
                episode_done=False,
                env_steps_delta=int(raw.executed_steps),
                episode_steps_delta=int(raw.executed_steps),
                episode_return_delta=float(raw.reward_sum),
                episode_success=False,
                last_info=dict(raw.chunk_info),
            )
        ]


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing dependency: {_IMPORT_ERROR}")
class AgiBotRolloutProcessorTest(unittest.TestCase):
    def test_processor_assembles_and_commits_replay_transition(self) -> None:
        data_store = _FakeDataStore()
        update_contexts: list[str] = []
        assembler = _FakeAssembler()
        processor = AgiBotRolloutProcessor(
            transition_assembler=assembler,
            data_store=data_store,
            trainer_update_fn=update_contexts.append,
            steps_per_update=1,
        )

        processed = processor.process_step_chunk(
            episode_id=3,
            episode_step_start=7,
            residual_obs_before_chunk={
                "state": np.asarray([0.0], dtype=np.float32)
            },
            action_chunk=np.asarray([[0.25]], dtype=np.float32),
            chunk_result={
                "observations": [{"state": np.asarray([1.0], dtype=np.float32)}],
                "rewards": [1.0],
                "dones": [True],
                "infos": [
                    {
                        "controller_action_executed": True,
                        "success": True,
                    }
                ],
                "obs": {"state": np.asarray([1.0], dtype=np.float32)},
                "done": True,
                "truncated": False,
                "reward_sum": 1.0,
                "info": {"success": True},
            },
            task_prompt="test",
        )

        self.assertEqual(processed.raw.executed_steps, 1)
        self.assertEqual(len(processed.assembled_chunks), 1)
        self.assertEqual(len(data_store.inserted), 1)
        self.assertEqual(data_store.inserted[0]["episode_id"], 3)
        self.assertEqual(update_contexts, ["commit_step_1"])

        processor.close()
        self.assertTrue(assembler.closed)

    def test_zero_step_terminal_updates_pending_transition_before_flush(self) -> None:
        data_store = _FakeDataStore()
        assembler = _FakePendingAssembler()
        processor = AgiBotRolloutProcessor(
            transition_assembler=assembler,
            data_store=data_store,
            trainer_update_fn=lambda _context: None,
            steps_per_update=10,
        )

        processor.process_step_chunk(
            episode_id=3,
            episode_step_start=7,
            residual_obs_before_chunk={
                "state": np.asarray([0.0], dtype=np.float32)
            },
            action_chunk=np.asarray([[0.25]], dtype=np.float32),
            chunk_result={
                "observations": [{"state": np.asarray([1.0], dtype=np.float32)}],
                "rewards": [1.0],
                "dones": [False],
                "infos": [
                    {
                        "controller_action_executed": True,
                        "success": False,
                    }
                ],
                "obs": {"state": np.asarray([1.0], dtype=np.float32)},
                "done": False,
                "truncated": False,
                "reward_sum": 1.0,
                "info": {"success": False},
            },
            task_prompt="test",
        )
        self.assertEqual(data_store.inserted, [])

        processor.finalize_zero_step_terminal(
            terminal_reward=2.0,
            boundary_flag=True,
            wait_for_episode_commit=True,
        )

        self.assertEqual(len(data_store.inserted), 1)
        self.assertEqual(data_store.inserted[0]["rewards"], 3.0)
        self.assertEqual(data_store.inserted[0]["dones"], True)
        self.assertEqual(data_store.inserted[0]["masks"], 0.0)


if __name__ == "__main__":
    unittest.main()
