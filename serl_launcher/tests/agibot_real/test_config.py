from __future__ import annotations

import sys
from pathlib import Path
import unittest

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
        self.assertFalse(parsed.offline.enabled)
        self.assertEqual(parsed.offline.prepared_path, None)
        self.assertEqual(parsed.offline.ratio, 0.5)
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")

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
                "policy": {"type": "openpi", "host": "127.0.0.1", "port": 30001},
                "env": {"action_dim": 14, "backend": "local"},
                "residual": {
                    "alpha": 0.2,
                    "action_mask": [True] * 14,
                    "action_limits": [1.0] * 14,
                    "chunk_horizon": 50,
                },
            }
        )

        parsed = parse_train_cfg(cfg)
        self.assertFalse(parsed.offline.enabled)
        self.assertEqual(parsed.offline.pretrain_steps, 0)
        self.assertEqual(parsed.logging.episode_log_file, "episode_logs.jsonl")

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
