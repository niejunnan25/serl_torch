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
    from serl_torch.examples.libero.config import parse_train_cfg


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class LiberoConfigTest(unittest.TestCase):
    def test_parse_train_cfg_reads_backfill_policy_from_yaml(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual.yaml"
        )

        parsed = parse_train_cfg(cfg)
        self.assertFalse(parsed.backfill_policy.enabled)
        self.assertEqual(parsed.backfill_policy.host, "localhost")
        self.assertEqual(parsed.backfill_policy.port, 30001)
        self.assertEqual(parsed.backfill_policy.max_pending_chunks, 2)
        self.assertEqual(parsed.backfill_policy.mode, "thread")

    def test_parse_train_cfg_requires_explicit_backfill_policy_block(self) -> None:
        cfg = OmegaConf.create(
            {
                "global_seed": 0,
                "task": {"suite_name": "libero_spatial", "task_id": 4},
                "runtime": {"role": "learner"},
                "wandb": {"project": "libero", "exp_name": "test"},
                "policy": {"type": "openpi", "host": "127.0.0.1", "port": 30001},
                "env": {"action_dim": 7, "backend": "remote", "seed": 7},
                "obs": {"image_keys": ["image", "wrist_image"], "stack_horizon": 1},
                "residual": {
                    "alpha": 0.1,
                    "action_mask": [True] * 7,
                    "action_limits": [1.0] * 7,
                    "chunk_horizon": 5,
                },
                "encoder": {"type": "resnet", "use_proprio": False},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "backfill_policy must be declared explicitly in the train yaml",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_invalid_backfill_policy_values(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual.yaml"
        )
        cfg.backfill_policy.max_pending_chunks = 0

        with self.assertRaisesRegex(
            ValueError,
            "backfill_policy.max_pending_chunks must be positive",
        ):
            parse_train_cfg(cfg)


if __name__ == "__main__":
    unittest.main()
