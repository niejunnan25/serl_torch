from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
