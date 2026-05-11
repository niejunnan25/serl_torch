from __future__ import annotations

from collections import deque
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import types
import unittest
from unittest import mock

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

if "gym" not in sys.modules and "gymnasium" not in sys.modules:
    fake_gym = types.ModuleType("gym")

    class _FakeBox:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    class _FakeDict:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    fake_gym.spaces = types.SimpleNamespace(Box=_FakeBox, Dict=_FakeDict)
    sys.modules["gym"] = fake_gym

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        AgiBotTransitionAssembler,
    )
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        AssemblyResult,
    )
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        PrefetchedDecisionObs,
    )
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        RawChunkRecord,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    AgiBotTransitionAssembler = object  # type: ignore[assignment]
    AssemblyResult = object  # type: ignore[assignment]
    PrefetchedDecisionObs = object  # type: ignore[assignment]
    RawChunkRecord = object  # type: ignore[assignment]


def _fake_agibot_cfg(*, async_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        obs=SimpleNamespace(
            image_keys=("image_rgb_0", "image_rgb_1", "image_rgb_2"),
        ),
        residual=SimpleNamespace(alpha=0.2),
        env=SimpleNamespace(arm_layout="dual_arm"),
        backfill_policy=SimpleNamespace(
            enabled=bool(async_enabled),
            mode="thread",
            host="127.0.0.1",
            port=9100,
            max_pending_chunks=8,
        ),
    )


def _fake_prefetched_decision(seed: float) -> PrefetchedDecisionObs:
    return PrefetchedDecisionObs(
        base_actions=np.asarray(
            [[float(seed)], [float(seed) + 0.5]],
            dtype=np.float32,
        ),
        residual_obs={
            "state": np.asarray([float(seed)], dtype=np.float32),
        },
    )


def _fake_actor_result(
    *,
    prefetched: PrefetchedDecisionObs | None,
    next_obs_seed: float,
    episode_done: bool = False,
) -> AssemblyResult:
    return AssemblyResult(
        transitions=[
            {
                "observations": {"state": np.asarray([0.0], dtype=np.float32)},
                "actions": np.asarray([0.1], dtype=np.float32),
                "next_observations": {
                    "state": np.asarray([next_obs_seed], dtype=np.float32)
                },
                "rewards": 1.0,
                "masks": 1.0,
                "dones": bool(episode_done),
            }
        ],
        prefetched=prefetched,
        next_obs={"marker": float(next_obs_seed)},
        episode_done=bool(episode_done),
        env_steps_delta=1,
        episode_steps_delta=1,
        episode_return_delta=1.0,
        episode_success=False,
        last_info={"marker": float(next_obs_seed)},
    )


def _fake_agibot_raw_chunk(
    *,
    obs_seeds: tuple[float, ...] = (10.0, 20.0),
    done: bool = False,
    truncated: bool = False,
) -> RawChunkRecord:
    executed_steps = len(obs_seeds)
    return RawChunkRecord(
        episode_id=1,
        episode_step_start=0,
        residual_obs_before_chunk={
            "state": np.asarray([-1.0], dtype=np.float32)
        },
        action_chunk=np.asarray(
            [[float(index) + 0.1] for index in range(executed_steps)],
            dtype=np.float32,
        ),
        post_step_observations=[
            {"seed": float(value)} for value in obs_seeds
        ],
        rewards=[1.0 for _ in obs_seeds],
        dones=[False for _ in obs_seeds],
        infos=[{"success": False} for _ in obs_seeds],
        final_obs={"seed": float(obs_seeds[-1])},
        chunk_done=bool(done),
        chunk_truncated=bool(truncated),
        reward_sum=float(executed_steps),
        chunk_info={"seed": float(obs_seeds[-1])},
        executed_steps=executed_steps,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing dependency: {_IMPORT_ERROR}")
class AgiBotTransitionAssemblerRuntimeTest(unittest.TestCase):
    def test_sync_path_reuses_prefetched_decision_obs(self) -> None:
        assembler = AgiBotTransitionAssembler(
            cfg=_fake_agibot_cfg(async_enabled=False),
            base_policy=object(),
            logger=logging.getLogger(__name__),
        )
        decision_obses = deque(
            [_fake_prefetched_decision(1.0)]
        )
        prefetched = _fake_prefetched_decision(2.0)

        assembler.infer_decision_obs = lambda *, obs, task_prompt: decision_obses.popleft()
        assembler.process_chunk = lambda *, raw, task_prompt: _fake_actor_result(
            prefetched=prefetched,
            next_obs_seed=7.0,
        )

        first = assembler.infer_decision_obs(
            obs={"seed": 0.0},
            task_prompt="task",
        )
        assembled = assembler.handle_chunk(
            raw=_fake_agibot_raw_chunk(obs_seeds=(10.0,)),
            task_prompt="task",
        )
        reused = assembler.pop_prefetched_decision_obs()

        self.assertTrue(np.allclose(first.base_actions[:, 0], [1.0, 1.5]))
        self.assertEqual(len(assembled), 1)
        self.assertIs(assembled[0].prefetched, prefetched)
        self.assertIs(reused, prefetched)
        self.assertIsNone(assembler.pop_prefetched_decision_obs())

    def test_async_path_backfills_full_chunk_without_next_decision_handoff(self) -> None:
        fake_backfill_policy = SimpleNamespace(close=lambda: None)
        with mock.patch(
            "serl_torch.examples.agibot_real.runtime.transition_assembly.build_agibot_base_policy",
            return_value=fake_backfill_policy,
        ):
            assembler = AgiBotTransitionAssembler(
                cfg=_fake_agibot_cfg(async_enabled=True),
                base_policy=object(),
                logger=logging.getLogger(__name__),
            )

        decision_obses = deque([_fake_prefetched_decision(1.0)])
        assembler.infer_decision_obs = lambda *, obs, task_prompt: decision_obses.popleft()
        assembler._backfill_residual_observations = (
            lambda *, observations, task_prompt, base_policy: [
                {"state": np.asarray([float(obs["seed"])], dtype=np.float32)}
                for obs in observations
            ]
        )

        assembler.infer_decision_obs(
            obs={"seed": 0.0},
            task_prompt="task",
        )
        assembled = assembler.handle_chunk(
            raw=_fake_agibot_raw_chunk(),
            task_prompt="task",
        )
        drained = list(assembled)
        for _ in range(20):
            if drained:
                break
            drained = assembler.drain_ready()
            time.sleep(0.01)

        self.assertEqual(len(drained), 1)
        self.assertTrue(
            np.allclose(
                drained[0].transitions[-1]["next_observations"]["state"],
                [20.0],
            )
        )

    def test_finish_episode_blocks_until_pending_chunk_commits(self) -> None:
        fake_backfill_policy = SimpleNamespace(close=lambda: None)
        with mock.patch(
            "serl_torch.examples.agibot_real.runtime.transition_assembly.build_agibot_base_policy",
            return_value=fake_backfill_policy,
        ):
            assembler = AgiBotTransitionAssembler(
                cfg=_fake_agibot_cfg(async_enabled=True),
                base_policy=object(),
                logger=logging.getLogger(__name__),
            )

        decision_obses = deque([_fake_prefetched_decision(1.0)])
        assembler.infer_decision_obs = lambda *, obs, task_prompt: decision_obses.popleft()
        def _slow_backfill(*, observations, task_prompt, base_policy):
            del task_prompt, base_policy
            time.sleep(0.05)
            return [
                {"state": np.asarray([float(obs["seed"])], dtype=np.float32)}
                for obs in observations
            ]

        assembler._backfill_residual_observations = _slow_backfill

        assembler.infer_decision_obs(
            obs={"seed": 0.0},
            task_prompt="task",
        )
        assembler.handle_chunk(
            raw=_fake_agibot_raw_chunk(),
            task_prompt="task",
        )

        finished = assembler.finish_episode(
            block=True,
        )

        self.assertEqual(len(finished), 1)
        self.assertTrue(
            np.allclose(
                finished[0].transitions[-1]["next_observations"]["state"],
                [20.0],
            )
        )
        self.assertEqual(assembler.finish_episode(block=True), [])


if __name__ == "__main__":
    unittest.main()
