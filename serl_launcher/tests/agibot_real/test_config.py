from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

if OmegaConf is not None:
    from serl_torch.examples.agibot_real.config import parse_eval_cfg
    from serl_torch.examples.agibot_real.config import parse_train_cfg


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class AgiBotConfigTest(unittest.TestCase):
    def test_parse_train_cfg_with_offline_block(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )

        parsed = parse_train_cfg(cfg)
        self.assertEqual(parsed.wandb.project, "agibot_real")
        self.assertEqual(parsed.wandb.mode, "online")
        self.assertFalse(parsed.wandb.debug)
        self.assertFalse(parsed.offline.enabled)
        self.assertEqual(parsed.offline.prepared_path, None)
        self.assertEqual(parsed.offline.ratio, 0.5)
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")
        self.assertFalse(parsed.action_filter.enabled)
        self.assertEqual(parsed.action_filter.alpha, 0.25)
        self.assertIsNone(parsed.action_filter.max_delta)
        self.assertEqual(parsed.action_filter.warmup_steps, 0)
        self.assertTrue(parsed.action_filter.reset_each_episode)

    def test_parse_eval_cfg_from_yaml(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "eval_residual.yaml"
        )

        parsed = parse_eval_cfg(cfg)
        self.assertEqual(parsed.eval.episodes, 10)
        self.assertTrue(parsed.eval.deterministic)
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")

    def test_parse_train_cfg_keeps_defaults_without_offline_block(self) -> None:
        cfg = OmegaConf.create(
            {
                "global_seed": 0,
                "task": {
                    "name": "agibot_real_default",
                    "prompt": "test prompt",
                    "control_mode": "camera_position",
                },
                "runtime": {"role": "learner"},
                "controller": {"enabled": True},
                "policy": {"type": "openpi", "host": "127.0.0.1", "port": 30001},
                "backfill_policy": {
                    "enabled": False,
                    "host": "${policy.host}",
                    "port": "${policy.port}",
                    "max_pending_chunks": 2,
                    "mode": "thread",
                },
                "env": {"action_dim": 14, "backend": "local"},
                "obs": {"image_keys": ["image_rgb_0"]},
                "residual": {
                    "alpha": 0.2,
                    "action_mask": [True] * 14,
                    "action_limits": [1.0] * 14,
                    "chunk_horizon": 50,
                },
                "encoder": {"use_proprio": False},
            }
        )

        parsed = parse_train_cfg(cfg)
        self.assertEqual(parsed.wandb.mode, "online")
        self.assertFalse(parsed.offline.enabled)
        self.assertEqual(parsed.offline.pretrain_steps, 0)
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")
        self.assertFalse(parsed.backfill_policy.enabled)
        self.assertEqual(parsed.backfill_policy.port, 30001)
        self.assertFalse(parsed.action_filter.enabled)
        self.assertEqual(parsed.action_filter.alpha, 0.25)

    def test_parse_train_cfg_uses_wandb_entity_env_default(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )

        with mock.patch.dict("os.environ", {"WANDB_ENTITY": "robotics"}, clear=False):
            parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.entity, "robotics")

    def test_parse_train_cfg_accepts_right_arm_policy_action_layout(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.policy.type = "openpi"
        cfg.policy.action_layout = "right_arm"

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.policy.action_layout, "right_arm")

    def test_parse_train_cfg_debug_disables_wandb_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.wandb.mode = "offline"
        cfg.wandb.debug = True

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.mode, "disabled")
        self.assertTrue(parsed.wandb.debug)

    def test_parse_train_cfg_rejects_invalid_wandb_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.wandb.mode = "cloud"

        with self.assertRaisesRegex(ValueError, "wandb.mode must be one of"):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_action_filter_alpha_above_one(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.action_filter.alpha = 1.5

        with self.assertRaisesRegex(
            ValueError,
            "action_filter.alpha must be <= 1.0",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_requires_explicit_backfill_policy_block(self) -> None:
        cfg = OmegaConf.create(
            {
                "global_seed": 0,
                "task": {
                    "name": "agibot_real_default",
                    "prompt": "test prompt",
                    "control_mode": "camera_position",
                },
                "runtime": {"role": "learner"},
                "controller": {"enabled": True},
                "policy": {"type": "openpi", "host": "127.0.0.1", "port": 30001},
                "env": {"action_dim": 14, "backend": "local"},
                "obs": {"image_keys": ["image_rgb_0"]},
                "residual": {
                    "alpha": 0.2,
                    "action_mask": [True] * 14,
                    "action_limits": [1.0] * 14,
                    "chunk_horizon": 50,
                },
                "encoder": {"use_proprio": False},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "backfill_policy must be declared explicitly in the train yaml",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_controller_disabled(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.controller.enabled = False

        with self.assertRaisesRegex(
            ValueError,
            "controller.enabled=true",
        ):
            parse_train_cfg(cfg)

    def test_parse_eval_cfg_rejects_controller_disabled(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "eval_residual.yaml"
        )
        cfg.controller.enabled = False

        with self.assertRaisesRegex(
            ValueError,
            "controller.enabled=true",
        ):
            parse_eval_cfg(cfg)

    def test_parse_eval_cfg_rejects_removed_start_episode_idx(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "eval_residual.yaml"
        )
        cfg.eval.start_episode_idx = 0

        with self.assertRaisesRegex(
            ValueError,
            "start_episode_idx has been removed",
        ):
            parse_eval_cfg(cfg)


if __name__ == "__main__":
    unittest.main()
