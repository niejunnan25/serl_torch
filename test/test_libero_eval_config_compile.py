from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from omegaconf import OmegaConf

REPO_PARENT = Path(__file__).resolve().parents[2]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[1] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _install_optional_import_stubs() -> None:
    if importlib.util.find_spec("agentlace") is None:
        transport_mod = types.ModuleType("serl_launcher.common.trainer_transport")

        @dataclass(frozen=True, slots=True)
        class _TrainerTransportConfig:
            mode: str
            data_port: int
            control_timeout_ms: int
            data_queue_capacity: int
            data_socket_hwm: int
            commit_poll_ms: int
            wait_committed_on_episode_end: bool
            wait_committed_on_shutdown: bool

        def _validate_transport_mode(mode: object) -> str:
            return str(mode)

        transport_mod.SUPPORTED_TRANSPORT_MODES = ("sync_commit", "async_commit")
        transport_mod.TrainerTransportConfig = _TrainerTransportConfig
        transport_mod.validate_transport_mode = _validate_transport_mode
        sys.modules["serl_launcher.common.trainer_transport"] = transport_mod


_install_optional_import_stubs()

from serl_torch.examples.libero.config import EvalConfig
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.config import train_cfg_to_eval_cfg


class LiberoEvalConfigCompileTest(unittest.TestCase):
    def test_train_cfg_to_eval_cfg_preserves_torch_compile(self) -> None:
        cfg = OmegaConf.load("examples/libero/configs/train_residual_step.yaml")
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
