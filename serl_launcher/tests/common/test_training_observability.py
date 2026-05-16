from __future__ import annotations

import unittest

from serl_launcher.common.training_observability import build_learner_runtime_wandb_metrics
from serl_launcher.common.training_observability import build_rollout_env_step_wandb_metrics
from serl_launcher.common.training_observability import configure_rollout_wandb_metrics
from serl_launcher.common.training_observability import extract_learner_wandb_metrics
from serl_launcher.common.training_observability import extract_rollout_wandb_metrics


class _FakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.calls.append((name, dict(kwargs)))


class _FakeWandbLogger:
    def __init__(self) -> None:
        self.run = _FakeRun()


class _LegacyFakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def define_metric(self, name: str, **kwargs: object) -> None:
        if "hidden" in kwargs:
            raise TypeError("hidden is not supported")
        self.calls.append((name, dict(kwargs)))


class _LegacyFakeWandbLogger:
    def __init__(self) -> None:
        self.run = _LegacyFakeRun()


class TrainingObservabilityTest(unittest.TestCase):
    def test_extract_rollout_metrics_keep_episode_axis(self) -> None:
        payload = {
            "env_steps": 123,
            "rollout": {
                "episode_id": 7,
                "episode_return": 4.5,
                "cumulative_success_rate": 0.4,
                "recent_success_rate_20": 0.6,
            },
        }

        metrics = extract_rollout_wandb_metrics(payload)

        self.assertEqual(metrics["rollout/episode_id"], 7.0)
        self.assertEqual(metrics["rollout/episode_return"], 4.5)
        self.assertEqual(metrics["rollout_progress/env_steps"], 123.0)

    def test_extract_rollout_metrics_includes_residual_stats(self) -> None:
        payload = {
            "env_steps": 123,
            "rollout": {
                "episode_id": 7,
                "episode_return": 4.5,
                "cumulative_success_rate": 0.4,
                "recent_success_rate_20": 0.6,
            },
            "residual": {
                "mean_abs": 0.1,
                "max_abs": 0.9,
                "std": 0.2,
                "saturation_rate": 0.05,
                "action_delta_mean_abs": 0.01,
                "action_delta_max_abs": 0.03,
            },
        }

        metrics = extract_rollout_wandb_metrics(payload)

        self.assertEqual(metrics["residual/mean_abs"], 0.1)
        self.assertEqual(metrics["residual/max_abs"], 0.9)
        self.assertEqual(metrics["residual/std"], 0.2)
        self.assertEqual(metrics["residual/saturation_rate"], 0.05)
        self.assertEqual(metrics["action/mean_abs_delta_from_base"], 0.01)
        self.assertEqual(metrics["action/max_abs_delta_from_base"], 0.03)

    def test_build_rollout_env_step_metrics(self) -> None:
        metrics = {
            "rollout/episode_id": 7.0,
            "rollout/episode_return": 4.5,
            "rollout/cumulative_success_rate": 0.4,
            "rollout/recent_success_rate_20": 0.6,
            "rollout_progress/env_steps": 123.0,
            "speed/actor_env_steps_per_sec": 8.0,
            "residual/mean_abs": 0.1,
            "action/mean_abs_delta_from_base": 0.01,
        }

        env_step_metrics = build_rollout_env_step_wandb_metrics(metrics)

        self.assertEqual(env_step_metrics["rollout_by_env_steps/episode_id"], 7.0)
        self.assertEqual(
            env_step_metrics["rollout_by_env_steps/episode_return"],
            4.5,
        )
        self.assertEqual(
            env_step_metrics["rollout_by_env_steps/env_steps"],
            123.0,
        )
        self.assertEqual(
            env_step_metrics["speed_by_env_steps/actor_env_steps_per_sec"],
            8.0,
        )
        self.assertEqual(env_step_metrics["residual_by_env_steps/mean_abs"], 0.1)
        self.assertEqual(
            env_step_metrics["action_by_env_steps/mean_abs_delta_from_base"],
            0.01,
        )

    def test_configure_rollout_metrics_hides_axis_metric(self) -> None:
        logger = _FakeWandbLogger()

        configure_rollout_wandb_metrics(wandb_logger=logger)

        self.assertGreaterEqual(len(logger.run.calls), 1)
        axis_name, axis_kwargs = logger.run.calls[0]
        self.assertEqual(axis_name, "rollout/episode_id")
        self.assertEqual(axis_kwargs, {"hidden": True})

    def test_configure_rollout_metrics_fallback_when_hidden_unsupported(self) -> None:
        logger = _LegacyFakeWandbLogger()

        configure_rollout_wandb_metrics(wandb_logger=logger)

        self.assertGreaterEqual(len(logger.run.calls), 1)
        axis_name, axis_kwargs = logger.run.calls[0]
        self.assertEqual(axis_name, "rollout/episode_id")
        self.assertEqual(axis_kwargs, {})

    def test_extract_learner_metrics_share_single_prefix(self) -> None:
        update_info = {
            "critic_loss": 1.0,
            "critic_td_loss": 2.0,
            "actor_loss": 3.0,
            "predicted_qs": 4.0,
            "target_qs": 5.0,
            "actor_predicted_q": 6.0,
            "predicted_q_gap": 7.0,
            "temperature": 8.0,
            "entropy": 9.0,
        }

        metrics = extract_learner_wandb_metrics(update_info)

        expected_keys = {
            "learner/loss_critic",
            "learner/loss_critic_td",
            "learner/loss_actor",
            "learner/q_predicted_mean",
            "learner/q_target_mean",
            "learner/q_actor_predicted_mean",
            "learner/q_predicted_gap",
            "learner/temperature",
            "learner/entropy",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_build_learner_runtime_metrics(self) -> None:
        metrics = build_learner_runtime_wandb_metrics(
            update_steps=10,
            env_steps=200,
            replay_size=128,
            updates_per_sec=3.5,
            eval_queue_backlog=2,
            offline_replay_size=64,
            batch_mix={"online_batch_size": 96, "offline_batch_size": 32},
            timer_metrics={
                "sample_replay_buffer": 0.01,
                "train": 0.2,
                "publish_network": 0.03,
            },
            sample_profile_metrics={"sample_online": 0.004},
            update_profile_metrics={"update_actor": 0.02},
        )

        self.assertEqual(metrics["progress/env_steps"], 200.0)
        self.assertEqual(metrics["progress/update_steps"], 10.0)
        self.assertEqual(metrics["progress/replay_size"], 128.0)
        self.assertEqual(metrics["speed/learner_updates_per_sec"], 3.5)
        self.assertEqual(metrics["eval_queue/backlog"], 2.0)
        self.assertEqual(metrics["replay/online_size"], 128.0)
        self.assertEqual(metrics["replay/offline_size"], 64.0)
        self.assertEqual(metrics["replay/sample_online_count"], 96.0)
        self.assertEqual(metrics["replay/sample_offline_count"], 32.0)
        self.assertAlmostEqual(metrics["replay/offline_ratio_actual"], 0.25)
        self.assertEqual(metrics["learner_time/sample_replay_buffer_sec"], 0.01)
        self.assertEqual(metrics["learner_time/train_sec"], 0.2)
        self.assertEqual(metrics["learner_time/publish_network_sec"], 0.03)
        self.assertEqual(metrics["learner_time/sample_profile/sample_online"], 0.004)
        self.assertEqual(metrics["learner_time/update_profile/update_actor"], 0.02)


if __name__ == "__main__":
    unittest.main()
