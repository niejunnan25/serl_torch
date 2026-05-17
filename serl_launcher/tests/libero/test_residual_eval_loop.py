from __future__ import annotations

import logging
import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import types
import unittest
from unittest import mock

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _install_optional_import_stubs() -> None:
    if importlib.util.find_spec("hydra") is None:
        hydra_mod = types.ModuleType("hydra")
        hydra_mod.main = lambda **_kwargs: (lambda fn: fn)
        hydra_core_mod = types.ModuleType("hydra.core")
        hydra_config_mod = types.ModuleType("hydra.core.hydra_config")

        class _HydraConfig:
            @staticmethod
            def get() -> SimpleNamespace:
                return SimpleNamespace(
                    runtime=SimpleNamespace(output_dir="."),
                )

        hydra_config_mod.HydraConfig = _HydraConfig
        hydra_utils_mod = types.ModuleType("hydra.utils")
        hydra_utils_mod.get_original_cwd = lambda: "."
        sys.modules["hydra"] = hydra_mod
        sys.modules["hydra.core"] = hydra_core_mod
        sys.modules["hydra.core.hydra_config"] = hydra_config_mod
        sys.modules["hydra.utils"] = hydra_utils_mod

    if importlib.util.find_spec("omegaconf") is None:
        omegaconf_mod = types.ModuleType("omegaconf")
        omegaconf_mod.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_mod

    if importlib.util.find_spec("einops") is None:
        drq_mod = types.ModuleType(
            "serl_launcher.agents.continuous.drq_typed_config"
        )

        def _unused_create_drq_agent_from_typed_cfg(
            *args: object,
            **kwargs: object,
        ) -> object:
            del args, kwargs
            raise AssertionError("base-policy-only eval should not create a DRQ agent")

        drq_mod.create_drq_agent_from_typed_cfg = _unused_create_drq_agent_from_typed_cfg
        sys.modules[
            "serl_launcher.agents.continuous.drq_typed_config"
        ] = drq_mod

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

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_torch.examples.libero.scripts import run_residual_eval as eval_mod
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    eval_mod = None  # type: ignore[assignment]


class _FakeChunkEnv:
    def __init__(self, *, done_after: int) -> None:
        self.done_after = int(done_after)
        self.task_description = "fake task"
        self.current_init_state_idx = 0
        self.step_count = 0
        self.chunk_lengths: list[int] = []
        self.actions: list[np.ndarray] = []
        self.close_calls = 0

    def reset(self, *, seed: int, init_episode_idx: int) -> dict[str, object]:
        del seed
        self.current_init_state_idx = int(init_episode_idx)
        self.step_count = 0
        return {"step": 0}

    def step_chunk(self, actions: np.ndarray) -> dict[str, object]:
        action_chunk = np.asarray(actions, dtype=np.float32)
        self.chunk_lengths.append(int(action_chunk.shape[0]))
        observations: list[dict[str, object]] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, object]] = []
        for action in action_chunk:
            self.actions.append(np.array(action, copy=True))
            self.step_count += 1
            done = self.step_count >= self.done_after
            observations.append({"step": int(self.step_count)})
            rewards.append(1.0)
            dones.append(bool(done))
            infos.append(
                {
                    "env_done": bool(done),
                    "success": bool(done),
                    "init_state_idx": int(self.current_init_state_idx),
                }
            )
            if done:
                break
        if not observations:
            raise RuntimeError("empty action chunk")
        return {
            "obs": observations[-1],
            "observations": observations,
            "rewards": rewards,
            "dones": dones,
            "done": bool(dones[-1]),
            "truncated": False,
            "infos": infos,
            "info": dict(infos[-1]),
            "reward_sum": float(sum(rewards)),
            "num_steps": int(len(rewards)),
        }

    def close(self, *, clear_cache: bool = False) -> None:
        del clear_cache
        self.close_calls += 1


class _CountingPolicyClient:
    def __init__(self, *, chunk_horizon: int, action_dim: int) -> None:
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.infer_calls = 0
        self.infer_many_calls = 0
        self.batch_sizes: list[int] = []
        self.close_calls = 0

    def infer(self, policy_input: object) -> tuple[np.ndarray, dict[str, object]]:
        del policy_input
        self.infer_calls += 1
        return (
            np.full(
                (self.chunk_horizon, self.action_dim),
                0.25,
                dtype=np.float32,
            ),
            {},
        )

    def infer_many(
        self,
        policy_inputs: list[object],
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        self.infer_many_calls += 1
        self.batch_sizes.append(len(policy_inputs))
        return (
            [
                np.full(
                    (self.chunk_horizon, self.action_dim),
                    0.25,
                    dtype=np.float32,
                )
                for _ in policy_inputs
            ],
            {},
        )

    def close(self) -> None:
        self.close_calls += 1


class _SerialOnlyPolicyClient:
    def __init__(self, *, chunk_horizon: int, action_dim: int) -> None:
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.infer_calls = 0
        self.close_calls = 0

    def infer(self, policy_input: object) -> tuple[np.ndarray, dict[str, object]]:
        del policy_input
        self.infer_calls += 1
        return (
            np.full(
                (self.chunk_horizon, self.action_dim),
                0.25,
                dtype=np.float32,
            ),
            {},
        )

    def close(self) -> None:
        self.close_calls += 1


def _fake_cfg(
    *,
    chunk_horizon: int = 5,
    action_dim: int = 1,
    max_env_steps_per_episode: int | None = None,
    episodes: int = 1,
    parallel_envs: int = 1,
    policy_batch_size: int | None = None,
    eval_ports: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        global_seed=0,
        libero_root=None,
        libero_config_dir=None,
        libero_datasets_root=None,
        task=SimpleNamespace(suite_name="libero_spatial", task_id=4),
        policy=SimpleNamespace(type="fake", host="127.0.0.1", port=1, id="fake"),
        env=SimpleNamespace(
            action_dim=int(action_dim),
            seed=7,
            backend="remote",
            remote=SimpleNamespace(
                host="127.0.0.1",
                port=30010,
                timeout_sec=1.0,
                ports=eval_ports,
            ),
        ),
        obs=SimpleNamespace(image_keys=("image",)),
        residual=SimpleNamespace(
            alpha=0.1,
            action_mask=[True] * int(action_dim),
            action_limits=[1.0] * int(action_dim),
            clip_gripper=True,
            chunk_horizon=int(chunk_horizon),
        ),
        encoder=SimpleNamespace(),
        network=SimpleNamespace(),
        sac=SimpleNamespace(),
        training=SimpleNamespace(),
        logging=SimpleNamespace(
            episode_log_file="episode_logs.jsonl",
            summary_file="summary.json",
        ),
        eval=SimpleNamespace(
            episodes=int(episodes),
            start_episode_idx=0,
            max_env_steps_per_episode=max_env_steps_per_episode,
            deterministic=True,
            checkpoint_path=None,
            checkpoint_step=None,
            parallel_envs=int(parallel_envs),
            policy_batch_size=(
                int(policy_batch_size)
                if policy_batch_size is not None
                else int(parallel_envs)
            ),
        ),
    )


@unittest.skipIf(_IMPORT_ERROR is not None, str(_IMPORT_ERROR))
class LiberoResidualEvalLoopTest(unittest.TestCase):
    def _run_fake_eval(
        self,
        *,
        done_after: int,
        chunk_horizon: int = 5,
        max_env_steps_per_episode: int | None = None,
    ) -> tuple[dict[str, object], _FakeChunkEnv, _CountingPolicyClient]:
        assert eval_mod is not None
        cfg = _fake_cfg(
            chunk_horizon=chunk_horizon,
            max_env_steps_per_episode=max_env_steps_per_episode,
        )
        env = _FakeChunkEnv(done_after=done_after)
        policy_client = _CountingPolicyClient(
            chunk_horizon=chunk_horizon,
            action_dim=int(cfg.env.action_dim),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(eval_mod, "cfg_to_log_payload", return_value={}),
                mock.patch.object(eval_mod, "set_global_seeds"),
                mock.patch.object(eval_mod, "create_env", return_value=env),
                mock.patch.object(
                    eval_mod,
                    "build_policy_client",
                    return_value=policy_client,
                ),
                mock.patch.object(
                    eval_mod,
                    "describe_policy_backend",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_type",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_id",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "build_libero_state",
                    side_effect=lambda obs: np.asarray(
                        [float(obs["step"])],
                        dtype=np.float32,
                    ),
                ),
                mock.patch.object(
                    eval_mod,
                    "extract_libero_images",
                    return_value={
                        "image": np.zeros((2, 2, 3), dtype=np.uint8),
                    },
                ),
                mock.patch.object(
                    eval_mod,
                    "build_libero_policy_input",
                    side_effect=lambda **kwargs: dict(kwargs),
                ),
            ):
                summary = eval_mod.run_residual_eval(
                    cfg,
                    run_dir=Path(tmpdir),
                    logger=logging.getLogger(__name__),
                )

        return summary, env, policy_client

    def _run_parallel_fake_eval(
        self,
        *,
        policy_client: object,
        episodes: int = 7,
        parallel_envs: int = 3,
        policy_batch_size: int = 3,
        done_after: int = 5,
    ) -> tuple[dict[str, object], list[_FakeChunkEnv]]:
        assert eval_mod is not None
        cfg = _fake_cfg(
            chunk_horizon=5,
            episodes=episodes,
            parallel_envs=parallel_envs,
            policy_batch_size=policy_batch_size,
            eval_ports=tuple(range(30100, 30100 + parallel_envs)),
        )
        envs = [_FakeChunkEnv(done_after=done_after) for _ in range(parallel_envs)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(eval_mod, "cfg_to_log_payload", return_value={}),
                mock.patch.object(eval_mod, "set_global_seeds"),
                mock.patch.object(eval_mod, "create_env", side_effect=envs),
                mock.patch.object(
                    eval_mod,
                    "build_policy_client",
                    return_value=policy_client,
                ),
                mock.patch.object(
                    eval_mod,
                    "describe_policy_backend",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_type",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_id",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "build_libero_state",
                    side_effect=lambda obs: np.asarray(
                        [float(obs["step"])],
                        dtype=np.float32,
                    ),
                ),
                mock.patch.object(
                    eval_mod,
                    "extract_libero_images",
                    return_value={
                        "image": np.zeros((2, 2, 3), dtype=np.uint8),
                    },
                ),
                mock.patch.object(
                    eval_mod,
                    "build_libero_policy_input",
                    side_effect=lambda **kwargs: dict(kwargs),
                ),
            ):
                summary = eval_mod.run_residual_eval(
                    cfg,
                    run_dir=Path(tmpdir),
                    logger=logging.getLogger(__name__),
                )

        return summary, envs

    def test_eval_builds_policy_decision_once_per_chunk(self) -> None:
        summary, env, policy_client = self._run_fake_eval(done_after=25)

        self.assertEqual(summary["env_steps"], 25)
        self.assertEqual(summary["policy_requests"], 5)
        self.assertEqual(policy_client.infer_calls, 5)
        self.assertEqual(env.chunk_lengths, [5, 5, 5, 5, 5])
        self.assertAlmostEqual(summary["policy_requests_per_env_step"], 0.2)

    def test_eval_does_not_prefetch_after_mid_chunk_done(self) -> None:
        summary, env, policy_client = self._run_fake_eval(done_after=3)

        self.assertEqual(summary["env_steps"], 3)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["policy_requests"], 1)
        self.assertEqual(policy_client.infer_calls, 1)
        self.assertEqual(env.chunk_lengths, [5])

    def test_eval_respects_manual_episode_cap_inside_chunk(self) -> None:
        summary, env, policy_client = self._run_fake_eval(
            done_after=25,
            max_env_steps_per_episode=3,
        )

        self.assertEqual(summary["env_steps"], 3)
        self.assertEqual(summary["successes"], 0)
        self.assertEqual(summary["policy_requests"], 1)
        self.assertEqual(policy_client.infer_calls, 1)
        self.assertEqual(env.chunk_lengths, [3])

    def test_base_policy_only_eval_executes_base_actions(self) -> None:
        summary, env, policy_client = self._run_fake_eval(done_after=5)

        self.assertFalse(summary["checkpoint_loaded"])
        self.assertEqual(summary["policy_requests"], 1)
        self.assertEqual(policy_client.infer_calls, 1)
        self.assertEqual(len(env.actions), 5)
        for action in env.actions:
            self.assertTrue(
                np.allclose(action, np.asarray([0.25], dtype=np.float32))
            )

    def test_parallel_eval_batches_policy_requests(self) -> None:
        policy_client = _CountingPolicyClient(chunk_horizon=5, action_dim=1)

        summary, envs = self._run_parallel_fake_eval(policy_client=policy_client)

        self.assertEqual(summary["episodes_completed"], 7)
        self.assertEqual(summary["env_steps"], 35)
        self.assertEqual(summary["parallel_envs"], 3)
        self.assertEqual(summary["policy_batch_size"], 3)
        self.assertEqual(summary["policy_requests"], 3)
        self.assertEqual(summary["policy_batch_requests"], 3)
        self.assertEqual(summary["policy_samples"], 7)
        self.assertEqual(policy_client.infer_calls, 0)
        self.assertEqual(policy_client.infer_many_calls, 3)
        self.assertEqual(policy_client.batch_sizes, [3, 3, 1])
        self.assertEqual([env.close_calls for env in envs], [1, 1, 1])
        self.assertAlmostEqual(summary["policy_samples_per_env_step"], 0.2)

    def test_parallel_eval_falls_back_to_serial_policy_infer(self) -> None:
        policy_client = _SerialOnlyPolicyClient(chunk_horizon=5, action_dim=1)

        summary, envs = self._run_parallel_fake_eval(policy_client=policy_client)

        self.assertEqual(summary["episodes_completed"], 7)
        self.assertEqual(summary["env_steps"], 35)
        self.assertEqual(summary["policy_requests"], 7)
        self.assertEqual(summary["policy_batch_requests"], 0)
        self.assertEqual(summary["policy_samples"], 7)
        self.assertEqual(policy_client.infer_calls, 7)
        self.assertEqual([env.close_calls for env in envs], [1, 1, 1])

    def test_eval_closes_created_env_when_later_env_creation_fails(self) -> None:
        assert eval_mod is not None
        cfg = _fake_cfg(
            episodes=2,
            parallel_envs=2,
            policy_batch_size=2,
            eval_ports=(30100, 30101),
        )
        first_env = _FakeChunkEnv(done_after=5)
        created_first_env = False

        def _create_env(_cfg: object, _logger: logging.Logger) -> _FakeChunkEnv:
            nonlocal created_first_env
            del _cfg, _logger
            if not created_first_env:
                created_first_env = True
                return first_env
            raise RuntimeError("second env failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(eval_mod, "cfg_to_log_payload", return_value={}),
                mock.patch.object(eval_mod, "set_global_seeds"),
                mock.patch.object(eval_mod, "create_env", side_effect=_create_env),
            ):
                with self.assertRaisesRegex(RuntimeError, "second env failed"):
                    eval_mod.run_residual_eval(
                        cfg,
                        run_dir=Path(tmpdir),
                        logger=logging.getLogger(__name__),
                    )

        self.assertEqual(first_env.close_calls, 1)

    def test_eval_closes_envs_when_policy_client_creation_fails(self) -> None:
        assert eval_mod is not None
        cfg = _fake_cfg(
            episodes=2,
            parallel_envs=2,
            policy_batch_size=2,
            eval_ports=(30100, 30101),
        )
        envs = [_FakeChunkEnv(done_after=5) for _ in range(2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(eval_mod, "cfg_to_log_payload", return_value={}),
                mock.patch.object(eval_mod, "set_global_seeds"),
                mock.patch.object(eval_mod, "create_env", side_effect=envs),
                mock.patch.object(
                    eval_mod,
                    "build_policy_client",
                    side_effect=RuntimeError("policy client failed"),
                ),
                mock.patch.object(
                    eval_mod,
                    "describe_policy_backend",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_type",
                    return_value="fake",
                ),
                mock.patch.object(
                    eval_mod,
                    "resolve_policy_backend_id",
                    return_value="fake",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "policy client failed"):
                    eval_mod.run_residual_eval(
                        cfg,
                        run_dir=Path(tmpdir),
                        logger=logging.getLogger(__name__),
                    )

        self.assertEqual([env.close_calls for env in envs], [1, 1])


if __name__ == "__main__":
    unittest.main()
