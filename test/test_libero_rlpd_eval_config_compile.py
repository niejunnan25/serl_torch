from __future__ import annotations

import unittest
from pathlib import Path
import sys

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

REPO_PARENT = Path(__file__).resolve().parents[2]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[1] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

if OmegaConf is not None:
    from serl_torch.examples.libero.config import EvalConfig
    from serl_torch.examples.libero.rlpd.config import parse_train_cfg
    from serl_torch.examples.libero.rlpd.config import train_cfg_to_eval_cfg


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class LiberoRLPDEvalConfigCompileTest(unittest.TestCase):
    def test_train_cfg_to_eval_cfg_preserves_torch_compile(self) -> None:
        cfg = OmegaConf.load("examples/libero/configs/train_rlpd.yaml")
        cfg.training.torch_compile.enabled = True
        cfg.training.torch_compile.target = "actor_critic"
        cfg.training.torch_compile.backend = "inductor"
        cfg.training.torch_compile.mode = "default"
        cfg.training.torch_compile.fullgraph = True
        cfg.training.torch_compile.dynamic = False

        train_cfg = parse_train_cfg(cfg)
        eval_cfg = train_cfg_to_eval_cfg(
            train_cfg,
            eval_cfg=EvalConfig(
                episodes=1,
                start_episode_idx=0,
                max_env_steps_per_episode=None,
                deterministic=True,
                checkpoint_path=None,
                checkpoint_step=None,
                allow_random_policy=False,
            ),
        )

        self.assertTrue(eval_cfg.training.torch_compile.enabled)
        self.assertEqual(eval_cfg.training.torch_compile.target, "actor_critic")
        self.assertEqual(eval_cfg.training.torch_compile.backend, "inductor")
        self.assertEqual(eval_cfg.training.torch_compile.mode, "default")
        self.assertTrue(eval_cfg.training.torch_compile.fullgraph)
        self.assertFalse(eval_cfg.training.torch_compile.dynamic)


if __name__ == "__main__":
    unittest.main()
