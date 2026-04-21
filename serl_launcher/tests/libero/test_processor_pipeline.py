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
    from serl_torch.examples.libero.runtime.processor_pipeline import (
        ProcessorChunkContext,
    )
    from serl_torch.examples.libero.runtime.processor_pipeline import (
        ProcessorTransitionStage,
    )
    from serl_torch.examples.libero.runtime.processor_pipeline import (
        RolloutProcessorPipeline,
    )
    from serl_torch.examples.libero.runtime.transition_assembly import AssemblyResult
    from serl_torch.examples.libero.runtime.transition_assembly import (
        PrefetchedDecisionObs,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    ProcessorChunkContext = object  # type: ignore[assignment]
    ProcessorTransitionStage = object  # type: ignore[assignment]
    RolloutProcessorPipeline = object  # type: ignore[assignment]
    AssemblyResult = object  # type: ignore[assignment]
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


def _fake_chunk_payload() -> dict[str, object]:
    obs0 = _fake_obs(1.0)
    obs1 = _fake_obs(2.0)
    obs2 = _fake_obs(3.0)
    return {
        "chunk_seq": 2,
        "episode_id": 4,
        "episode_step_start": 7,
        "task_prompt": "stack blocks",
        "chunk_result": {
            "steps": [
                {
                    "obs": obs0,
                    "action": np.asarray([0.1, 0.2], dtype=np.float32),
                    "reward": 1.0,
                    "done": False,
                    "info": {"env_done": False},
                    "next_obs": obs1,
                },
                {
                    "obs": obs1,
                    "action": np.asarray([0.3, 0.4], dtype=np.float32),
                    "reward": 2.0,
                    "done": True,
                    "info": {"env_done": True},
                    "next_obs": obs2,
                },
            ],
            "num_steps": 2,
            "reward_sum": 3.0,
            "obs": obs2,
            "done": True,
            "truncated": False,
            "info": {"env_done": True},
        },
    }


class _FakeAssembler:
    def infer_decision_obs(
        self,
        *,
        obs: dict[str, object],
        task_prompt: str,
    ) -> PrefetchedDecisionObs:
        del obs, task_prompt
        return PrefetchedDecisionObs(
            base_actions=np.asarray([[1.0], [2.0]], dtype=np.float32),
            residual_obs={"state": np.asarray([5.0], dtype=np.float32)},
        )

    def process_chunk(
        self,
        *,
        raw: object,
        task_prompt: str,
    ) -> AssemblyResult:
        del raw, task_prompt
        return AssemblyResult(
            transitions=[{"reward": 3.0}],
            prefetched=None,
            next_obs={"state": np.asarray([9.0], dtype=np.float32)},
            episode_done=True,
            env_steps_delta=2,
            episode_steps_delta=2,
            episode_return_delta=3.0,
            episode_success=True,
            last_info={"env_done": True},
        )


class _TagTransitionStage(ProcessorTransitionStage):
    name = "tag_transition_batch"

    def run(
        self,
        assembled_chunk: AssemblyResult,
        *,
        context: ProcessorChunkContext,
    ) -> AssemblyResult:
        del context
        tagged_transitions = [dict(item, tagged=True) for item in assembled_chunk.transitions]
        return AssemblyResult(
            transitions=tagged_transitions,
            prefetched=assembled_chunk.prefetched,
            next_obs=assembled_chunk.next_obs,
            episode_done=assembled_chunk.episode_done,
            env_steps_delta=assembled_chunk.env_steps_delta,
            episode_steps_delta=assembled_chunk.episode_steps_delta,
            episode_return_delta=assembled_chunk.episode_return_delta,
            episode_success=assembled_chunk.episode_success,
            last_info=assembled_chunk.last_info,
        )


@unittest.skipIf(_IMPORT_ERROR is not None, str(_IMPORT_ERROR))
class RolloutProcessorPipelineTest(unittest.TestCase):
    def test_pipeline_runs_default_and_custom_stages(self) -> None:
        pipeline = RolloutProcessorPipeline.for_libero(
            assembler=_FakeAssembler(),  # type: ignore[arg-type]
            transition_stages=(_TagTransitionStage(),),
        )

        assembled = pipeline.process_payload(_fake_chunk_payload())

        self.assertEqual(assembled.env_steps_delta, 2)
        self.assertTrue(assembled.episode_done)
        self.assertEqual(assembled.transitions[0]["reward"], 3.0)
        self.assertEqual(assembled.transitions[0]["tagged"], True)


if __name__ == "__main__":
    unittest.main()
