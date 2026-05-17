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

_IMPORT_ERROR: ModuleNotFoundError | None = None
if OmegaConf is not None:
    try:
        from serl_torch.examples.agibot_real.config import parse_eval_cfg
        from serl_torch.examples.agibot_real.config import parse_train_cfg
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        _IMPORT_ERROR = exc


@unittest.skipIf(
    OmegaConf is None or _IMPORT_ERROR is not None,
    "omegaconf or AgiBot config dependencies are not installed",
)
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
        self.assertTrue(parsed.offline.enabled)
        self.assertEqual(
            parsed.offline.prepared_path,
            "/home/hello/codebase/serl_torch/examples/agibot_real/outputs/offline_data/office_setting/joyra_joyra_office_setting_chunk30_alpha0p005",
        )
        self.assertEqual(parsed.offline.ratio, 0.5)
        self.assertEqual(
            parsed.offline.prepare.raw_dataset_path,
            "/home/hello/codebase/datasets/task_3463_mouse",
        )
        self.assertEqual(
            parsed.offline.prepare.output_root,
            "/home/hello/codebase/serl_torch/examples/agibot_real/outputs/offline_data",
        )
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")
        self.assertFalse(parsed.action_filter.enabled)
        self.assertEqual(parsed.env.arm_layout, "dual_arm")
        self.assertEqual(parsed.env.action_dim, 14)
        self.assertEqual(parsed.env.robot_action_dim, 14)
        self.assertEqual(parsed.action_filter.alpha, 0.25)
        self.assertIsNone(parsed.action_filter.max_delta)
        self.assertEqual(parsed.action_filter.warmup_steps, 0)
        self.assertTrue(parsed.action_filter.reset_each_episode)
        self.assertFalse(parsed.replay.prepared_chunk.offline_enabled)
        self.assertFalse(parsed.replay.prepared_chunk.online_enabled)
        self.assertFalse(parsed.training.async_eval.enabled)
        self.assertEqual(parsed.training.async_eval.every_episodes, 20)
        self.assertEqual(parsed.processor.mode, "in_process")
        self.assertTrue(parsed.processor_batching.enabled)
        self.assertEqual(parsed.processor_batching.max_batch_chunks, 4)
        self.assertEqual(parsed.runtime.processor_transport.port, 5491)
        self.assertFalse(parsed.recycle.enabled)

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
        self.assertFalse(parsed.action_filter.enabled)
        self.assertEqual(parsed.action_filter.alpha, 0.25)
        self.assertFalse(parsed.eval.logging.enabled)
        self.assertEqual(parsed.eval.logging.backend, "swanlab")
        self.assertEqual(parsed.eval.logging.project, "agibot_real")

    def test_parse_train_cfg_rejects_async_eval_enabled(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.training.async_eval.enabled = True

        with self.assertRaisesRegex(
            ValueError,
            "AgiBot real robot training currently does not support async eval",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_accepts_standalone_processor_role(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.runtime.role = "processor"
        cfg.processor.mode = "standalone"

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.runtime.role, "processor")
        self.assertEqual(parsed.processor.mode, "standalone")

    def test_parse_train_cfg_rejects_processor_role_without_standalone_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.runtime.role = "processor"
        cfg.processor.mode = "in_process"

        with self.assertRaisesRegex(
            ValueError,
            "runtime.role=processor requires processor.mode=standalone",
        ):
            parse_train_cfg(cfg)

    def test_agibot_does_not_define_async_eval_worker(self) -> None:
        worker_path = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "runtime"
            / "async_eval_worker.py"
        )

        self.assertFalse(worker_path.exists())

    def test_parse_train_cfg_accepts_fake_backend(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.env.backend = "fake"

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.env.backend, "fake")

    def test_parse_train_cfg_accepts_fake_right_arm_backend(self) -> None:
        base_cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        override_cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual_openpi_right_arm.yaml"
        )
        override_cfg.pop("defaults", None)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
        cfg.env.backend = "fake"

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.env.backend, "fake")
        self.assertEqual(parsed.env.arm_layout, "right_arm")
        self.assertEqual(parsed.env.action_dim, 7)

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

    def test_parse_train_cfg_accepts_right_arm_layout(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.policy.type = "openpi"
        cfg.policy.action_layout = "right_arm"
        cfg.env.arm_layout = "right_arm"
        cfg.env.action_dim = 7
        cfg.residual.action_mask = [True] * 7
        cfg.residual.action_limits = [1.0] * 7

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.env.arm_layout, "right_arm")
        self.assertEqual(parsed.env.action_dim, 7)
        self.assertEqual(parsed.policy.action_layout, "right_arm")

    def test_parse_train_cfg_rejects_policy_env_layout_conflict(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.policy.action_layout = "right_arm"
        cfg.env.arm_layout = "dual_arm"

        with self.assertRaisesRegex(ValueError, "policy.action_layout must match"):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_single_arm_14d_action_dim(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.policy.action_layout = "right_arm"
        cfg.env.arm_layout = "right_arm"

        with self.assertRaisesRegex(ValueError, "requires env.action_dim=7"):
            parse_train_cfg(cfg)

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


class AgiBotAsyncEvalStaticSafetyTest(unittest.TestCase):
    def test_agibot_runtime_has_no_async_eval_worker(self) -> None:
        worker_path = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "agibot_real"
            / "runtime"
            / "async_eval_worker.py"
        )

        self.assertFalse(worker_path.exists())


if __name__ == "__main__":
    unittest.main()
