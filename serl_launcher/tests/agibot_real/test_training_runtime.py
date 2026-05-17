from __future__ import annotations

import logging
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.common.training_payloads import build_rollout_payload
from serl_launcher.common.training_payloads import build_rollout_stats_payload
from serl_launcher.utils.serialization import to_jsonable

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.agibot_real.scripts import run_residual_training
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    run_residual_training = None  # type: ignore[assignment]


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing dependency: {_IMPORT_ERROR}")
class AgiBotTrainingRuntimeTest(unittest.TestCase):
    def test_prepared_chunk_wrapper_respects_config(self) -> None:
        class _FakePrepared:
            prepare_profile = {
                "prepared_windows": 3.0,
                "prepare_action_cache_sec": 0.1,
                "prepare_scalar_cache_sec": 0.2,
            }
            num_windows = 3

            def __init__(self, replay_buffer, *, name: str):
                self.replay_buffer = replay_buffer
                self.name = name

            def __len__(self) -> int:
                return len(self.replay_buffer)

        cfg = SimpleNamespace(
            replay=SimpleNamespace(
                prepared_chunk=SimpleNamespace(
                    offline_enabled=True,
                    online_enabled=False,
                )
            )
        )
        replay_buffer = [object(), object(), object()]

        with mock.patch.object(
            run_residual_training,
            "PreparedStepWindowReplayBufferSampler",
            _FakePrepared,
        ):
            wrapped, profile = (
                run_residual_training._maybe_wrap_offline_replay_for_prepared_chunk(
                    cfg=cfg,
                    offline_replay_buffer=replay_buffer,
                    logger=logging.getLogger(__name__),
                )
            )

        self.assertIsInstance(wrapped, _FakePrepared)
        self.assertIs(wrapped.replay_buffer, replay_buffer)
        self.assertEqual(wrapped.name, "offline")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["prepared_windows"], 3.0)

    def test_prepared_chunk_wrapper_can_stay_disabled(self) -> None:
        cfg = SimpleNamespace(
            replay=SimpleNamespace(
                prepared_chunk=SimpleNamespace(
                    offline_enabled=False,
                    online_enabled=False,
                )
            )
        )
        replay_buffer = [object()]

        wrapped, profile = (
            run_residual_training._maybe_wrap_offline_replay_for_prepared_chunk(
                cfg=cfg,
                offline_replay_buffer=replay_buffer,
                logger=logging.getLogger(__name__),
            )
        )

        self.assertIs(wrapped, replay_buffer)
        self.assertIsNone(profile)


class AgiBotTrainingSourceShapeTest(unittest.TestCase):
    def test_actor_delegates_transition_processing_to_processor(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "scripts"
            / "run_residual_training.py"
        ).read_text(encoding="utf-8")

        self.assertIn("AgiBotRolloutProcessor(", source)
        self.assertIn("rollout_processor.process_step_chunk(", source)
        self.assertNotIn("RawChunkRecord.from_step_chunk_result(", source)
        self.assertNotIn("def _commit_assembled_chunks", source)

    def test_actor_logs_rollout_on_episode_and_env_step_axes(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "scripts"
            / "run_residual_training.py"
        ).read_text(encoding="utf-8")

        self.assertIn("step=rollout_step", source)
        self.assertIn("build_rollout_env_step_wandb_metrics", source)
        self.assertIn("speed/actor_env_steps_per_sec", source)

    def test_rollout_metrics_payload_is_jsonable_with_episode_and_env_axes(self) -> None:
        payload = build_rollout_stats_payload(
            env_steps=123,
            rollout=build_rollout_payload(
                episode_id=7,
                episode_steps=30,
                episode_return=1.0,
                success=True,
                cumulative_success_rate=0.5,
                recent_success_rate_20=0.4,
            ),
            env_info={"success": True},
        )

        encoded = to_jsonable(payload)
        json.dumps(encoded)
        self.assertEqual(encoded["env_steps"], 123)
        self.assertEqual(encoded["rollout"]["episode_id"], 7)

    def test_eval_runner_records_standalone_eval_axes(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "runtime"
            / "eval_runner.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"eval_episode_id"', source)
        self.assertIn('"eval_env_steps"', source)
        self.assertIn('"policy_requests_per_env_step"', source)

    def test_learner_does_not_queue_agibot_async_eval(self) -> None:
        source = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "scripts"
            / "run_residual_training.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("start_async_eval_worker", source)
        self.assertNotIn("append_async_eval", source)
        self.assertNotIn("save_async_eval", source)


if __name__ == "__main__":
    unittest.main()
