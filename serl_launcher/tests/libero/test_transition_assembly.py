from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.libero.transition_assembly import (
        BatchAwareLiberoTransitionAssembler,
    )
    from serl_torch.examples.libero.transition_assembly import (
        LiberoTransitionAssembler,
    )
    from serl_torch.examples.libero.transition_assembly import (
        RawChunkRecord,
    )
    from serl_torch.examples.libero.transition_assembly import (
        assemble_chunk_step_transitions,
    )
    from serl_torch.examples.libero.transition_assembly import (
        backfill_post_step_residual_obs_batch_aware,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    BatchAwareLiberoTransitionAssembler = object  # type: ignore[assignment]
    LiberoTransitionAssembler = object  # type: ignore[assignment]
    RawChunkRecord = object  # type: ignore[assignment]

    def assemble_chunk_step_transitions(*args: object, **kwargs: object) -> object:
        raise _IMPORT_ERROR

    def backfill_post_step_residual_obs_batch_aware(
        *args: object,
        **kwargs: object,
    ) -> object:
        raise _IMPORT_ERROR


class _FakeAssembler(LiberoTransitionAssembler):
    def __init__(
        self,
        *,
        base_actions: list[np.ndarray],
        residual_observations: list[dict[str, np.ndarray]],
    ) -> None:
        self._base_actions = base_actions
        self._residual_observations = residual_observations

    def backfill_post_step_residual_obs(
        self,
        *,
        observations: list[dict[str, object]],
        task_prompt: str,
    ) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        del observations, task_prompt
        return self._base_actions, self._residual_observations


class _BatchOnlyPolicyClient:
    def __init__(self) -> None:
        self.infer_many_calls = 0
        self.infer_calls = 0

    def infer_many(
        self,
        policy_inputs: list[object],
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        self.infer_many_calls += 1
        actions: list[np.ndarray] = []
        for index, _policy_input in enumerate(policy_inputs, start=1):
            actions.append(
                np.asarray(
                    [[float(index)], [float(index) + 0.5]],
                    dtype=np.float32,
                )
            )
        return actions, {"backend": "fake_batch"}

    def infer(self, policy_input: object) -> tuple[np.ndarray, dict[str, object]]:
        del policy_input
        self.infer_calls += 1
        raise AssertionError("batch-capable path should not fall back to infer()")


class _SerialOnlyPolicyClient:
    def __init__(self) -> None:
        self.infer_calls = 0

    def infer(self, policy_input: object) -> tuple[np.ndarray, dict[str, object]]:
        del policy_input
        self.infer_calls += 1
        base = float(self.infer_calls)
        return (
            np.asarray([[base], [base + 0.25]], dtype=np.float32),
            {"backend": "fake_serial"},
        )


def _fake_chunk_result(
    *,
    observations: list[dict[str, object]],
    rewards: list[float],
    dones: list[bool],
    infos: list[dict[str, object]],
    done: bool,
    truncated: bool,
) -> dict[str, object]:
    return {
        "observations": observations,
        "rewards": rewards,
        "dones": dones,
        "infos": infos,
        "obs": observations[-1],
        "done": bool(done),
        "truncated": bool(truncated),
        "reward_sum": float(sum(rewards)),
        "info": dict(infos[-1]),
        "num_steps": len(rewards),
    }


def _fake_libero_obs(value: float) -> dict[str, object]:
    pixel = np.full((8, 8, 3), fill_value=int(value), dtype=np.uint8)
    return {
        "robot0_eef_pos": np.asarray([value, value + 0.1, value + 0.2], dtype=np.float32),
        "robot0_eef_axis_angle": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
        "agentview_image": pixel,
        "robot0_eye_in_hand_image": pixel,
    }


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing dependency: {_IMPORT_ERROR}")
class LiberoTransitionAssemblyTest(unittest.TestCase):
    def test_assemble_chunk_step_transitions_chains_observations(self) -> None:
        obs0 = {"state": np.asarray([0.0], dtype=np.float32)}
        obs1 = {"state": np.asarray([1.0], dtype=np.float32)}
        obs2 = {"state": np.asarray([2.0], dtype=np.float32)}
        obs3 = {"state": np.asarray([3.0], dtype=np.float32)}

        transitions = assemble_chunk_step_transitions(
            episode_id=7,
            episode_step_start=3,
            residual_obs_before_chunk=obs0,
            executed_actions=np.asarray(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=np.float32,
            ),
            rewards=[0.5, 1.5, 2.5],
            dones=[False, False, True],
            infos=[
                {"env_done": False},
                {"env_done": False, "success": False},
                {"env_done": True, "success": True},
            ],
            next_residual_observations=[obs1, obs2, obs3],
        )

        self.assertEqual(len(transitions), 3)
        self.assertEqual([t["episode_step"] for t in transitions], [3, 4, 5])
        self.assertIs(transitions[0]["observations"], obs0)
        self.assertIs(transitions[0]["next_observations"], obs1)
        self.assertIs(transitions[1]["observations"], obs1)
        self.assertIs(transitions[1]["next_observations"], obs2)
        self.assertIs(transitions[2]["observations"], obs2)
        self.assertIs(transitions[2]["next_observations"], obs3)
        self.assertTrue(np.allclose(transitions[1]["actions"], [3.0, 4.0]))
        self.assertEqual([t["rewards"] for t in transitions], [0.5, 1.5, 2.5])
        self.assertEqual([t["masks"] for t in transitions], [1.0, 1.0, 0.0])
        self.assertEqual([t["dones"] for t in transitions], [False, False, True])

    def test_assemble_chunk_step_transitions_validates_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            assemble_chunk_step_transitions(
                episode_id=1,
                episode_step_start=0,
                residual_obs_before_chunk={"state": np.asarray([0.0])},
                executed_actions=np.asarray([[1.0], [2.0]], dtype=np.float32),
                rewards=[1.0],
                dones=[False, True],
                infos=[{}, {}],
                next_residual_observations=[
                    {"state": np.asarray([1.0])},
                    {"state": np.asarray([2.0])},
                ],
            )

    def test_raw_chunk_record_rejects_empty_chunk(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no executed steps"):
            RawChunkRecord.from_step_chunk_result(
                episode_id=1,
                episode_step_start=0,
                residual_obs_before_chunk={"state": np.asarray([0.0])},
                action_chunk=np.asarray([], dtype=np.float32).reshape(0, 1),
                chunk_result={
                    "observations": [],
                    "rewards": [],
                    "dones": [],
                    "infos": [],
                    "obs": {},
                    "done": False,
                    "truncated": False,
                    "reward_sum": 0.0,
                    "info": {},
                    "num_steps": 0,
                },
            )

    def test_raw_chunk_record_rejects_num_steps_past_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "shorter than executed_steps"):
            RawChunkRecord.from_step_chunk_result(
                episode_id=1,
                episode_step_start=0,
                residual_obs_before_chunk={"state": np.asarray([0.0])},
                action_chunk=np.asarray([[0.1], [0.2]], dtype=np.float32),
                chunk_result={
                    "observations": [{"raw": 1}],
                    "rewards": [1.0, 2.0],
                    "dones": [False, False],
                    "infos": [{}, {}],
                    "obs": {"raw": 1},
                    "done": False,
                    "truncated": False,
                    "reward_sum": 3.0,
                    "info": {},
                    "num_steps": 2,
                },
            )

    def test_raw_chunk_record_rejects_short_action_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_chunk is shorter"):
            RawChunkRecord.from_step_chunk_result(
                episode_id=1,
                episode_step_start=0,
                residual_obs_before_chunk={"state": np.asarray([0.0])},
                action_chunk=np.asarray([[0.1]], dtype=np.float32),
                chunk_result={
                    "observations": [{"raw": 1}, {"raw": 2}],
                    "rewards": [1.0, 2.0],
                    "dones": [False, False],
                    "infos": [{}, {}],
                    "obs": {"raw": 2},
                    "done": False,
                    "truncated": False,
                    "reward_sum": 3.0,
                    "info": {},
                    "num_steps": 2,
                },
            )

    def test_process_chunk_non_terminal_returns_prefetch(self) -> None:
        obs0 = {"state": np.asarray([0.0], dtype=np.float32)}
        obs1 = {"state": np.asarray([1.0], dtype=np.float32)}
        obs2 = {"state": np.asarray([2.0], dtype=np.float32)}
        base1 = np.asarray([[1.0], [1.1]], dtype=np.float32)
        base2 = np.asarray([[2.0], [2.1]], dtype=np.float32)
        assembler = _FakeAssembler(
            base_actions=[base1, base2],
            residual_observations=[obs1, obs2],
        )
        raw = RawChunkRecord.from_step_chunk_result(
            episode_id=3,
            episode_step_start=4,
            residual_obs_before_chunk=obs0,
            action_chunk=np.asarray([[0.1], [0.2]], dtype=np.float32),
            chunk_result=_fake_chunk_result(
                observations=[{"raw": 1}, {"raw": 2}],
                rewards=[1.0, 2.0],
                dones=[False, False],
                infos=[{"env_done": False}, {"env_done": False}],
                done=False,
                truncated=False,
            ),
        )

        result = assembler.process_chunk(raw=raw, task_prompt="task")

        self.assertFalse(result.episode_done)
        self.assertEqual(result.env_steps_delta, 2)
        self.assertEqual(result.episode_steps_delta, 2)
        self.assertEqual(result.episode_return_delta, 3.0)
        self.assertFalse(result.episode_success)
        self.assertIsNotNone(result.prefetched)
        assert result.prefetched is not None
        self.assertIs(result.prefetched.base_actions, base2)
        self.assertIs(result.prefetched.residual_obs, obs2)
        self.assertEqual(len(result.transitions), 2)
        self.assertIs(result.transitions[1]["observations"], obs1)
        self.assertIs(result.transitions[1]["next_observations"], obs2)

    def test_process_chunk_terminal_has_no_prefetch_and_zero_mask(self) -> None:
        obs0 = {"state": np.asarray([0.0], dtype=np.float32)}
        obs1 = {"state": np.asarray([1.0], dtype=np.float32)}
        assembler = _FakeAssembler(
            base_actions=[np.asarray([[1.0]], dtype=np.float32)],
            residual_observations=[obs1],
        )
        raw = RawChunkRecord.from_step_chunk_result(
            episode_id=1,
            episode_step_start=0,
            residual_obs_before_chunk=obs0,
            action_chunk=np.asarray([[0.1]], dtype=np.float32),
            chunk_result=_fake_chunk_result(
                observations=[{"raw": 1}],
                rewards=[1.0],
                dones=[True],
                infos=[{"env_done": True, "success": True}],
                done=True,
                truncated=False,
            ),
        )

        result = assembler.process_chunk(raw=raw, task_prompt="task")

        self.assertTrue(result.episode_done)
        self.assertTrue(result.episode_success)
        self.assertIsNone(result.prefetched)
        self.assertEqual(result.transitions[0]["masks"], 0.0)
        self.assertEqual(result.transitions[0]["dones"], True)

    def test_process_chunk_truncated_has_no_prefetch_but_keeps_bootstrap_mask(
        self,
    ) -> None:
        obs0 = {"state": np.asarray([0.0], dtype=np.float32)}
        obs1 = {"state": np.asarray([1.0], dtype=np.float32)}
        assembler = _FakeAssembler(
            base_actions=[np.asarray([[1.0]], dtype=np.float32)],
            residual_observations=[obs1],
        )
        raw = RawChunkRecord.from_step_chunk_result(
            episode_id=1,
            episode_step_start=0,
            residual_obs_before_chunk=obs0,
            action_chunk=np.asarray([[0.1]], dtype=np.float32),
            chunk_result=_fake_chunk_result(
                observations=[{"raw": 1}],
                rewards=[1.0],
                dones=[True],
                infos=[{"env_done": False, "step_limit_reached": True}],
                done=True,
                truncated=True,
            ),
        )

        result = assembler.process_chunk(raw=raw, task_prompt="task")

        self.assertTrue(result.episode_done)
        self.assertFalse(result.episode_success)
        self.assertIsNone(result.prefetched)
        self.assertEqual(result.transitions[0]["masks"], 1.0)
        self.assertEqual(result.transitions[0]["dones"], True)

    def test_batch_aware_backfill_uses_infer_many_and_preserves_order(self) -> None:
        policy_client = _BatchOnlyPolicyClient()
        observations = [
            _fake_libero_obs(1.0),
            _fake_libero_obs(2.0),
            _fake_libero_obs(3.0),
        ]

        base_actions, residual_observations = (
            backfill_post_step_residual_obs_batch_aware(
                observations=observations,
                task_prompt="pick up the block",
                policy_client=policy_client,
                chunk_horizon=2,
                image_keys=("image_rgb_0", "image_rgb_1"),
                residual_alpha=0.1,
            )
        )

        self.assertEqual(policy_client.infer_many_calls, 1)
        self.assertEqual(policy_client.infer_calls, 0)
        self.assertEqual(len(base_actions), 3)
        self.assertEqual(len(residual_observations), 3)
        self.assertTrue(np.allclose(base_actions[0][:, 0], [1.0, 1.5]))
        self.assertTrue(np.allclose(base_actions[1][:, 0], [2.0, 2.5]))
        self.assertTrue(np.allclose(base_actions[2][:, 0], [3.0, 3.5]))

    def test_batch_aware_backfill_falls_back_to_serial_infer(self) -> None:
        policy_client = _SerialOnlyPolicyClient()
        observations = [
            _fake_libero_obs(1.0),
            _fake_libero_obs(2.0),
        ]

        base_actions, residual_observations = (
            backfill_post_step_residual_obs_batch_aware(
                observations=observations,
                task_prompt="stack the blocks",
                policy_client=policy_client,
                chunk_horizon=2,
                image_keys=("image_rgb_0", "image_rgb_1"),
                residual_alpha=0.2,
            )
        )

        self.assertEqual(policy_client.infer_calls, 2)
        self.assertEqual(len(base_actions), 2)
        self.assertEqual(len(residual_observations), 2)
        self.assertTrue(np.allclose(base_actions[0][:, 0], [1.0, 1.25]))
        self.assertTrue(np.allclose(base_actions[1][:, 0], [2.0, 2.25]))

    def test_batch_aware_assembler_process_chunk_keeps_prefetch_contract(self) -> None:
        assembler = BatchAwareLiberoTransitionAssembler(
            policy_client=_BatchOnlyPolicyClient(),
            chunk_horizon=2,
            image_keys=("image_rgb_0", "image_rgb_1"),
            residual_alpha=0.1,
        )
        raw = RawChunkRecord.from_step_chunk_result(
            episode_id=5,
            episode_step_start=0,
            residual_obs_before_chunk={"state": np.asarray([0.0], dtype=np.float32)},
            action_chunk=np.asarray([[0.1], [0.2]], dtype=np.float32),
            chunk_result=_fake_chunk_result(
                observations=[_fake_libero_obs(1.0), _fake_libero_obs(2.0)],
                rewards=[0.0, 1.0],
                dones=[False, False],
                infos=[{"env_done": False}, {"env_done": False}],
                done=False,
                truncated=False,
            ),
        )

        result = assembler.process_chunk(raw=raw, task_prompt="move")

        self.assertFalse(result.episode_done)
        self.assertIsNotNone(result.prefetched)
        assert result.prefetched is not None
        self.assertTrue(np.allclose(result.prefetched.base_actions[:, 0], [2.0, 2.5]))


if __name__ == "__main__":
    unittest.main()
