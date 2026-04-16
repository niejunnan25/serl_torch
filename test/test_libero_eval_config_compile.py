from __future__ import annotations

import unittest

from omegaconf import OmegaConf

from serl_torch.examples.libero.config import EvalConfig
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.config import train_cfg_to_eval_cfg


class LiberoEvalConfigCompileTest(unittest.TestCase):
    def test_train_cfg_to_eval_cfg_preserves_torch_compile(self) -> None:
        cfg = OmegaConf.load("examples/libero/configs/train_residual.yaml")
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
