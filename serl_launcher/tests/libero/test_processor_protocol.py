from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.libero.runtime.processor_protocol import (
        build_processor_submission_payload,
    )
    from serl_torch.examples.libero.runtime.processor_protocol import (
        extract_actor_rollout_chunk_summary,
    )
    from serl_torch.examples.libero.runtime.processor_protocol import (
        normalize_chunk_result,
    )
    from serl_torch.examples.libero.runtime.processor_protocol import (
        reconstruct_chunk_execution_record,
    )
    from serl_torch.examples.libero.runtime.transition_assembly import (
        PrefetchedDecisionObs,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc

    def build_processor_submission_payload(*args: object, **kwargs: object) -> object:
        raise _IMPORT_ERROR

    def extract_actor_rollout_chunk_summary(
        *args: object, **kwargs: object
    ) -> object:
        raise _IMPORT_ERROR

    def normalize_chunk_result(*args: object, **kwargs: object) -> object:
        raise _IMPORT_ERROR

    def reconstruct_chunk_execution_record(*args: object, **kwargs: object) -> object:
        raise _IMPORT_ERROR

    PrefetchedDecisionObs = object  # type: ignore[assignment]


def _fake_obs(seed: float) -> dict[str, object]:
    pixel = np.full((4, 4, 3), int(seed), dtype=np.uint8)
    return {
        "robot0_eef_pos": np.asarray([seed, seed + 0.1, seed + 0.2], dtype=np.float32),
        "robot0_eef_axis_angle": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
        "agentview_image": pixel,
        "robot0_eye_in_hand_image": pixel,
    }


def _fake_chunk_result() -> dict[str, object]:
    obs0 = _fake_obs(1.0)
    obs1 = _fake_obs(2.0)
    obs2 = _fake_obs(3.0)
    steps = [
        {
            "obs": obs0,
            "action": np.asarray([0.1, 0.2], dtype=np.float32),
            "reward": 1.25,
            "done": False,
            "info": {"env_done": False},
            "next_obs": obs1,
        },
        {
            "obs": obs1,
            "action": np.asarray([0.3, 0.4], dtype=np.float32),
            "reward": 2.75,
            "done": True,
            "truncated": False,
            "info": {"env_done": True},
            "next_obs": obs2,
        },
    ]
    return {
        "steps": steps,
        "num_steps": 2,
        "reward_sum": 4.0,
        "obs": obs2,
        "done": True,
        "truncated": False,
        "info": {"env_done": True},
    }


class _FakeAssembler:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], str]] = []

    def infer_decision_obs(
        self,
        *,
        obs: dict[str, object],
        task_prompt: str,
    ) -> PrefetchedDecisionObs:
        self.calls.append((dict(obs), str(task_prompt)))
        return PrefetchedDecisionObs(
            base_actions=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            residual_obs={
                "state": np.asarray([9.0, 8.0], dtype=np.float32),
            },
        )


@unittest.skipIf(_IMPORT_ERROR is not None, str(_IMPORT_ERROR))
class ProcessorProtocolTest(unittest.TestCase):
    def test_normalize_chunk_result_returns_consistent_summary(self) -> None:
        normalized = normalize_chunk_result(_fake_chunk_result())

        self.assertEqual(normalized.executed_steps, 2)
        self.assertAlmostEqual(normalized.reward_sum, 4.0)
        self.assertTrue(normalized.chunk_done)
        self.assertFalse(normalized.chunk_truncated)
        self.assertEqual(normalized.infos[-1]["env_done"], True)
        self.assertEqual(len(normalized.post_step_observations), 2)

    def test_normalize_chunk_result_rejects_mismatched_declared_steps(self) -> None:
        chunk_result = _fake_chunk_result()
        chunk_result["num_steps"] = 3

        with self.assertRaisesRegex(
            ValueError,
            "chunk_result.num_steps does not match chunk_result.steps",
        ):
            normalize_chunk_result(chunk_result)

    def test_actor_rollout_summary_uses_same_validation_as_processor(self) -> None:
        chunk_result = _fake_chunk_result()
        chunk_result["reward_sum"] = 999.0

        with self.assertRaisesRegex(
            ValueError,
            "chunk_result.reward_sum does not match summed step rewards",
        ):
            extract_actor_rollout_chunk_summary(chunk_result)

    def test_reconstruct_chunk_execution_record_uses_payload_schema(self) -> None:
        payload = build_processor_submission_payload(
            chunk_seq=7,
            episode_id=5,
            episode_step_start=11,
            task_prompt="pick up the block",
            chunk_result=_fake_chunk_result(),
        )
        assembler = _FakeAssembler()

        record = reconstruct_chunk_execution_record(
            payload=payload,
            assembler=assembler,  # type: ignore[arg-type]
        )

        self.assertEqual(record.episode_id, 5)
        self.assertEqual(record.episode_step_start, 11)
        self.assertEqual(record.executed_steps, 2)
        self.assertEqual(record.reward_sum, 4.0)
        self.assertEqual(record.action_chunk.shape, (2, 2))
        self.assertTrue(np.allclose(record.action_chunk[0], np.asarray([0.1, 0.2])))
        self.assertEqual(record.residual_obs_before_chunk, {})
        self.assertIsNotNone(record.start_obs)
        assert record.start_obs is not None
        self.assertTrue(
            np.array_equal(
                record.start_obs["robot0_eef_pos"],
                np.asarray([1.0, 1.1, 1.2], dtype=np.float32),
            )
        )
        self.assertEqual(len(assembler.calls), 0)


if __name__ == "__main__":
    unittest.main()
