from __future__ import annotations

import logging
import unittest

from serl_launcher.common.training_reporting import sync_eval_results_to_wandb


class _FakeWandbLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int | None]] = []

    def log(self, data: dict[str, object], step: int | None = None) -> None:
        self.calls.append((data, step))


class TrainingReportingTest(unittest.TestCase):
    def test_sync_eval_results_uses_train_episode_id_as_step(self) -> None:
        wandb_logger = _FakeWandbLogger()

        sync_eval_results_to_wandb(
            records=[
                {
                    "status": "ok",
                    "eval_index": 2,
                    "train_episode_id": 150,
                    "train_update_step": 1234,
                    "train_env_step": 45678,
                    "summary": {"success_rate": 0.8, "mean_return": 0.7},
                }
            ],
            wandb_logger=wandb_logger,
            logger=logging.getLogger(__name__),
        )

        self.assertEqual(len(wandb_logger.calls), 1)
        metrics, step = wandb_logger.calls[0]
        self.assertEqual(step, 150)
        self.assertEqual(metrics["eval/train_episode_id"], 150.0)
        self.assertEqual(metrics["eval/success_rate"], 0.8)
        self.assertEqual(metrics["eval/mean_return"], 0.7)


if __name__ == "__main__":
    unittest.main()
