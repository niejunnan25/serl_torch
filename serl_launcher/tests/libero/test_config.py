from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
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

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

_IMPORT_ERROR: ModuleNotFoundError | None = None
if OmegaConf is not None:
    try:
        from serl_torch.examples.libero.config import parse_train_cfg_allow_processor
        from serl_torch.examples.libero.config import parse_train_cfg
        from serl_torch.examples.libero.config import parse_eval_cfg
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = ModuleNotFoundError("omegaconf is not installed")


@unittest.skipIf(_IMPORT_ERROR is not None, str(_IMPORT_ERROR))
class LiberoConfigTest(unittest.TestCase):
    def test_parse_train_cfg_reads_backfill_policy_from_yaml(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )

        parsed = parse_train_cfg(cfg)
        self.assertFalse(parsed.backfill_policy.enabled)
        self.assertEqual(parsed.backfill_policy.host, "localhost")
        self.assertEqual(parsed.backfill_policy.port, 30001)
        self.assertEqual(parsed.backfill_policy.max_pending_chunks, 2)
        self.assertEqual(parsed.backfill_policy.mode, "thread")
        self.assertIsNone(parsed.wandb.entity)
        self.assertEqual(parsed.wandb.mode, "online")
        self.assertFalse(parsed.processor_batching.enabled)
        self.assertEqual(parsed.processor_batching.max_batch_chunks, 4)
        self.assertEqual(parsed.processor_batching.max_batch_obs, 24)
        self.assertEqual(parsed.processor_batching.max_wait_ms, 3)
        self.assertFalse(parsed.recycle.enabled)
        self.assertEqual(parsed.recycle.output_root, "raw_rollout_recycle")
        self.assertEqual(
            parsed.runtime.processor_transport.port,
            int(parsed.runtime.trainer_transport.data_port) + 10,
        )
        self.assertEqual(
            parsed.runtime.processor_transport.timeout_ms,
            int(parsed.runtime.trainer_transport.control_timeout_ms),
        )
        self.assertEqual(parsed.runtime.processor_transport.queue_capacity, 4)
        self.assertEqual(parsed.training.async_eval.parallel_envs, 1)
        self.assertEqual(parsed.training.async_eval.policy_batch_size, 1)
        self.assertIsNone(parsed.training.async_eval.env.remote.ports)
        self.assertFalse(parsed.key_rl.enabled)
        self.assertEqual(parsed.key_rl.mode, "fixed_step")
        self.assertEqual(parsed.key_rl.start_step, 0)
        self.assertEqual(parsed.key_rl.replay_mode, "active_only")
        self.assertTrue(parsed.key_rl.require_chunk_boundary)
        for stage in (parsed.key_rl.stage1, parsed.key_rl.stage2, parsed.key_rl.stage3):
            self.assertFalse(stage.enabled)
            self.assertEqual(stage.start_step, 0)
            self.assertEqual(stage.end_step, 0)
        self.assertFalse(parsed.reward_model.enabled)
        self.assertIsNone(parsed.reward_model.checkpoint_path)
        self.assertEqual(parsed.reward_model.views, ("wrist",))
        self.assertEqual(parsed.reward_model.threshold, 0.8)
        self.assertEqual(parsed.reward_model.bonus, 0.5)
        self.assertEqual(parsed.reward_model.stage, "stage1")
        self.assertTrue(parsed.reward_model.one_shot)
        self.assertTrue(parsed.reward_model.apply_only_when_key_rl_active)
        self.assertIsNone(parsed.reward_model.device)

    def test_parse_train_cfg_reads_key_rl_override(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.key_rl.enabled = True
        cfg.key_rl.start_step = 30

        parsed = parse_train_cfg(cfg)

        self.assertTrue(parsed.key_rl.enabled)
        self.assertEqual(parsed.key_rl.mode, "fixed_step")
        self.assertEqual(parsed.key_rl.start_step, 30)
        self.assertEqual(parsed.key_rl.replay_mode, "active_only")
        self.assertTrue(parsed.key_rl.require_chunk_boundary)
        self.assertFalse(parsed.key_rl.stage1.enabled)

    def test_parse_train_cfg_reads_reward_model_override(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.reward_model.enabled = True
        cfg.reward_model.checkpoint_path = "/tmp/stage1_reward_model.pt"
        cfg.reward_model.views = ["wrist"]
        cfg.reward_model.threshold = 0.75
        cfg.reward_model.bonus = 0.25
        cfg.reward_model.device = "cuda:0"

        parsed = parse_train_cfg(cfg)

        self.assertTrue(parsed.reward_model.enabled)
        self.assertEqual(
            parsed.reward_model.checkpoint_path,
            "/tmp/stage1_reward_model.pt",
        )
        self.assertEqual(parsed.reward_model.views, ("wrist",))
        self.assertEqual(parsed.reward_model.threshold, 0.75)
        self.assertEqual(parsed.reward_model.bonus, 0.25)
        self.assertEqual(parsed.reward_model.stage, "stage1")
        self.assertEqual(parsed.reward_model.device, "cuda:0")

    def test_parse_train_cfg_rejects_enabled_reward_model_without_checkpoint(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.reward_model.enabled = True
        cfg.reward_model.checkpoint_path = None

        with self.assertRaisesRegex(
            ValueError,
            "reward_model.checkpoint_path must be set",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_reads_key_rl_fixed_stages_override(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.key_rl.enabled = True
        cfg.key_rl.mode = "fixed_stages"
        cfg.key_rl.stage1.enabled = True
        cfg.key_rl.stage1.start_step = 30
        cfg.key_rl.stage1.end_step = 75
        cfg.key_rl.stage2.enabled = True
        cfg.key_rl.stage2.start_step = 110
        cfg.key_rl.stage2.end_step = 160

        parsed = parse_train_cfg(cfg)

        self.assertTrue(parsed.key_rl.enabled)
        self.assertEqual(parsed.key_rl.mode, "fixed_stages")
        self.assertEqual(parsed.key_rl.stage1.start_step, 30)
        self.assertEqual(parsed.key_rl.stage1.end_step, 75)
        self.assertEqual(parsed.key_rl.stage2.start_step, 110)
        self.assertEqual(parsed.key_rl.stage2.end_step, 160)
        self.assertFalse(parsed.key_rl.stage3.enabled)

    def test_parse_train_cfg_rejects_unsupported_key_rl_stage(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.key_rl.stage4 = {
            "enabled": True,
            "start_step": 0,
            "end_step": 10,
        }

        with self.assertRaisesRegex(
            ValueError,
            "key_rl supports only stage1/stage2/stage3",
        ):
            parse_train_cfg(cfg)

    def test_spatial_chunk_config_exposes_key_rl_override(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "spatial_4_0514_runtime"
            / "spatial4_chunk_alpha0p2_unfiltered_offline_noent_std1p0_ports53300.yaml"
        )
        cfg.key_rl.enabled = True
        cfg.key_rl.start_step = 30

        parsed = parse_train_cfg(cfg)

        self.assertTrue(parsed.key_rl.enabled)
        self.assertEqual(parsed.key_rl.start_step, 30)

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
            / "train_residual_step.yaml"
        )
        cfg.backfill_policy.max_pending_chunks = 0

        with self.assertRaisesRegex(
            ValueError,
            "backfill_policy.max_pending_chunks must be positive",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_allow_processor_preserves_processor_role(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.runtime.role = "processor"

        parsed = parse_train_cfg_allow_processor(cfg)

        self.assertEqual(parsed.runtime.role, "processor")

    def test_parse_train_cfg_rejects_invalid_processor_transport_values(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.processor_transport = {"queue_capacity": 0}

        with self.assertRaisesRegex(
            ValueError,
            "processor_transport.queue_capacity must be positive",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_invalid_processor_batching_values(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_processor.yaml"
        )
        cfg.processor_batching.max_batch_obs = 0

        with self.assertRaisesRegex(
            ValueError,
            "processor_batching.max_batch_obs must be positive",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_blank_recycle_output_root(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )
        cfg.recycle = {"enabled": True, "output_root": "   "}

        with self.assertRaisesRegex(
            ValueError,
            "recycle.output_root must be a non-empty string",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_reads_wandb_entity_from_env(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )

        with patch.dict(os.environ, {"WANDB_ENTITY": "niejunnan"}, clear=False):
            parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.entity, "niejunnan")

    def test_parse_train_cfg_prefers_explicit_wandb_entity(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )
        cfg.wandb.entity = "explicit-team"

        with patch.dict(os.environ, {"WANDB_ENTITY": "niejunnan"}, clear=False):
            parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.entity, "explicit-team")

    def test_parse_train_cfg_reads_explicit_wandb_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.wandb.mode = "offline"
        cfg.wandb.debug = False

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.mode, "offline")

    def test_parse_train_cfg_falls_back_to_debug_for_wandb_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )
        del cfg.wandb["mode"]
        cfg.wandb.debug = True

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.mode, "disabled")

    def test_parse_train_cfg_debug_overrides_explicit_wandb_mode(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_step.yaml"
        )
        cfg.wandb.mode = "online"
        cfg.wandb.debug = True

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.wandb.mode, "disabled")

    def test_parse_train_cfg_rejects_parallel_async_eval_without_ports(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.parallel_envs = 2

        with self.assertRaisesRegex(
            ValueError,
            "training.async_eval.parallel_envs > 1 requires "
            "training.async_eval.env.remote.ports",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_parallel_async_eval_port_count_mismatch(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.parallel_envs = 3
        cfg.training.async_eval.env.remote.ports = [30110, 30111]

        with self.assertRaisesRegex(
            ValueError,
            "training.async_eval.env.remote.ports length must equal "
            "training.async_eval.parallel_envs",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_parallel_async_eval_duplicate_ports(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.parallel_envs = 2
        cfg.training.async_eval.env.remote.ports = [30110, 30110]

        with self.assertRaisesRegex(
            ValueError,
            "training.async_eval.env.remote.ports must not contain duplicate ports",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_parallel_async_eval_train_env_port(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.env.remote.port = 30110
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.parallel_envs = 2
        cfg.training.async_eval.env.remote.ports = [30110, 30111]

        with self.assertRaisesRegex(
            ValueError,
            "async eval requires dedicated eval env server ports",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_rejects_single_async_eval_train_env_port(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.env.remote.port = 30110
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.env.remote.port = 30110

        with self.assertRaisesRegex(
            ValueError,
            "async eval requires dedicated eval env server ports",
        ):
            parse_train_cfg(cfg)

    def test_parse_train_cfg_accepts_parallel_async_eval_ports(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "train_residual_chunk.yaml"
        )
        cfg.training.async_eval.enabled = True
        cfg.training.async_eval.parallel_envs = 3
        cfg.training.async_eval.policy_batch_size = 2
        cfg.training.async_eval.env.remote.ports = [30110, 30111, 30112]

        parsed = parse_train_cfg(cfg)

        self.assertEqual(parsed.training.async_eval.parallel_envs, 3)
        self.assertEqual(parsed.training.async_eval.policy_batch_size, 2)
        self.assertEqual(
            parsed.training.async_eval.env.remote.ports,
            (30110, 30111, 30112),
        )

    def test_parse_eval_cfg_rejects_parallel_eval_without_ports(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "eval_residual.yaml"
        )
        cfg.eval.parallel_envs = 2

        with self.assertRaisesRegex(
            ValueError,
            "eval.parallel_envs > 1 requires env.remote.ports",
        ):
            parse_eval_cfg(cfg)

    def test_parse_eval_cfg_rejects_parallel_duplicate_ports(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "eval_residual.yaml"
        )
        cfg.eval.parallel_envs = 2
        cfg.env.remote.ports = [30110, 30110]

        with self.assertRaisesRegex(
            ValueError,
            "env.remote.ports must not contain duplicate ports",
        ):
            parse_eval_cfg(cfg)

    def test_parse_eval_cfg_reads_key_rl_defaults(self) -> None:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "libero"
            / "configs"
            / "eval_residual.yaml"
        )

        parsed = parse_eval_cfg(cfg)

        self.assertFalse(parsed.key_rl.enabled)
        self.assertEqual(parsed.key_rl.start_step, 0)
        self.assertFalse(parsed.key_rl.stage1.enabled)
        self.assertFalse(parsed.key_rl.stage2.enabled)
        self.assertFalse(parsed.key_rl.stage3.enabled)


if __name__ == "__main__":
    unittest.main()
