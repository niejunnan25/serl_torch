from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

if OmegaConf is not None:
    from serl_torch.examples.libero.rlpd.config import parse_eval_cfg
    from serl_torch.examples.libero.rlpd.config import parse_train_cfg


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class LiberoRLPDConfigTest(unittest.TestCase):
    def test_parse_train_cfg_reads_direct_yaml(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_rlpd.yaml"
        )

        parsed = parse_train_cfg(cfg)
        self.assertEqual(parsed.offline.prepare.output_root, "data/rlpd/offline_data")
        self.assertEqual(parsed.obs.vector_obs_keys, ("robot_proprio",))
        self.assertFalse(hasattr(parsed, "policy"))
        self.assertFalse(hasattr(parsed, "residual"))
        self.assertEqual(parsed.runtime.trainer_transport.mode, "sync_commit")
        self.assertTrue(parsed.offline.enabled)
        self.assertEqual(parsed.training.training_starts, 200)
        self.assertEqual(parsed.training.random_steps, 300)

    def test_parse_train_cfg_defaults_missing_vector_obs_to_robot_proprio(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_rlpd.yaml"
        )
        cfg.obs.vector_obs_keys = None

        parsed = parse_train_cfg(cfg)
        self.assertEqual(parsed.obs.vector_obs_keys, ("robot_proprio",))

    def test_parse_train_cfg_rejects_non_robot_proprio_vector_obs(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_rlpd.yaml"
        )
        cfg.obs.vector_obs_keys = ["robot_proprio", "base_action"]

        with self.assertRaisesRegex(
            ValueError,
            "obs.vector_obs_keys must be exactly",
        ):
            parse_train_cfg(cfg)

    def test_parse_eval_cfg_defaults_to_checkpoint_required(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "eval_rlpd.yaml"
        )

        parsed = parse_eval_cfg(cfg)
        self.assertFalse(parsed.eval.allow_random_policy)
        self.assertIsNone(parsed.eval.checkpoint_path)


if __name__ == "__main__":
    unittest.main()
