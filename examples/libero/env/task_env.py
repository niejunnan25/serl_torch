"""Local LIBERO task environment wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .setup import resolve_libero_config_dir
from .setup import resolve_libero_datasets_root
from .setup import resolve_libero_root
from .setup import resolve_max_episode_steps
from .setup import setup_libero_pythonpath


def _clone_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_obs_tree(v) for k, v in value.items()}
    return np.array(value, copy=True)


def _shape_to_dim(shape: Any) -> Optional[int]:
    if shape is None:
        return None
    try:
        dim = int(np.prod(shape))
    except Exception:  # noqa: BLE001
        return None
    if dim <= 0:
        return None
    return dim


def _action_spec_to_dim(action_spec: Any) -> Optional[int]:
    if action_spec is None:
        return None
    if isinstance(action_spec, tuple) and len(action_spec) == 2:
        low, high = action_spec
        low_dim = _shape_to_dim(getattr(low, "shape", None))
        if low_dim is not None:
            return low_dim
        high_dim = _shape_to_dim(getattr(high, "shape", None))
        if high_dim is not None:
            return high_dim
    return _shape_to_dim(getattr(action_spec, "shape", None))


def _infer_runtime_action_dim(env: Any) -> Tuple[int, str]:
    sources: List[Tuple[str, Any]] = [("offscreen", env)]
    inner = getattr(env, "env", None)
    if inner is not None:
        sources.append(("inner", inner))

    for source_name, source_env in sources:
        action_space = getattr(source_env, "action_space", None)
        dim = _shape_to_dim(getattr(action_space, "shape", None))
        if dim is not None:
            return dim, f"{source_name}.action_space.shape"

    for source_name, source_env in sources:
        dim = _action_spec_to_dim(getattr(source_env, "action_spec", None))
        if dim is not None:
            return dim, f"{source_name}.action_spec"

    robots = getattr(inner, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        for idx, robot in enumerate(robots):
            try:
                dim = int(getattr(robot, "action_dim"))
            except Exception:  # noqa: BLE001
                continue
            if dim > 0:
                return dim, f"inner.robots[{idx}].action_dim"

    raise ValueError(
        "Unable to infer LIBERO action dim: expected action_space/action_spec/robot.action_dim"
    )


class LiberoTaskEnv:
    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        action_dim: Optional[int] = None,
        resolution: int = 256,
        num_steps_wait: int = 10,
        max_episode_steps: Optional[int] = None,
        libero_root: Optional[str] = None,
        libero_config_dir: Optional[str] = None,
        libero_datasets_root: Optional[str] = None,
        env_seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.suite_name = str(suite_name)
        self.task_id = int(task_id)
        self.resolution = int(resolution)
        self.num_steps_wait = int(num_steps_wait)
        if env_seed is None:
            raise ValueError("env.seed must be explicitly set")
        self.env_seed = int(env_seed)
        self.libero_root = resolve_libero_root(libero_root)
        self.libero_config_dir = resolve_libero_config_dir(libero_config_dir)
        self.libero_datasets_root = resolve_libero_datasets_root(
            libero_datasets_root,
            libero_root=self.libero_root,
        )
        setup_libero_pythonpath(
            self.libero_root,
            self.libero_config_dir,
            datasets_root=self.libero_datasets_root,
        )

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.suite_name]()
        self.task = self.task_suite.get_task(self.task_id)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        self._current_instruction = str(self.task.language)
        self._task_description = str(self.task.language)

        task_bddl_file = (
            Path(get_libero_path("bddl_files"))
            / self.task.problem_folder
            / self.task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.resolution,
            "camera_widths": self.resolution,
        }
        self.env = OffScreenRenderEnv(**env_args)
        runtime_action_dim, runtime_action_dim_source = _infer_runtime_action_dim(
            self.env
        )
        requested_action_dim = (
            int(action_dim) if action_dim is not None else runtime_action_dim
        )
        if requested_action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {requested_action_dim}")
        if requested_action_dim != runtime_action_dim:
            raise ValueError(
                f"Configured env.action_dim ({requested_action_dim}) does not match env action space "
                f"dim ({runtime_action_dim}, source={runtime_action_dim_source})"
            )
        self._action_dim = int(requested_action_dim)
        self.logger.info(
            "LIBERO action dim=%d (source=%s)",
            self._action_dim,
            runtime_action_dim_source,
        )
        self.env.seed(self.env_seed)

        self._step_limit = int(
            max_episode_steps
            if max_episode_steps is not None
            else resolve_max_episode_steps(self.suite_name)
        )
        self._take_action_cnt = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self.last_seed: Optional[int] = None
        self.current_init_state_idx: Optional[int] = None

    @property
    def current_instruction(self) -> str:
        return self._current_instruction

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def step_limit(self) -> int:
        return int(self._step_limit)

    @property
    def take_action_cnt(self) -> int:
        return int(self._take_action_cnt)

    @property
    def action_dim(self) -> int:
        return int(self._action_dim)

    def reset(
        self,
        seed: int,
        init_episode_idx: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del episode_info, seed
        episode_index = int(init_episode_idx)
        if episode_index < 0:
            raise ValueError(
                f"init_episode_idx must be >= 0, got {init_episode_idx}"
            )

        # Mainline semantics: keep env RNG fixed across episodes while
        # cycling LIBERO init states deterministically by episode id.
        applied_seed = int(self.env_seed)
        self.last_seed = int(applied_seed)
        self.current_init_state_idx = int(episode_index) % len(self.initial_states)
        self._take_action_cnt = 0

        self.env.seed(applied_seed)
        dummy_action = np.zeros((self._action_dim,), dtype=np.float32)
        if self._action_dim > 0:
            dummy_action[-1] = -1.0

        # Some init states can accidentally terminate during warmup dummy steps.
        # Retry a few times; if it still happens, skip warmup for this reset so the
        # first real training step does not hit "executing action in terminated episode".
        warmup_terminated = False
        obs = None
        max_warmup_retries = 3
        for warmup_retry in range(max_warmup_retries):
            self.env.reset()
            obs = self.env.set_init_state(self.initial_states[self.current_init_state_idx])
            warmup_terminated = False
            for wait_step in range(self.num_steps_wait):
                obs, _, done, _ = self.env.step(dummy_action.tolist())
                if bool(done):
                    warmup_terminated = True
                    self.logger.warning(
                        "LIBERO warmup terminated early: init_state_idx=%s wait_step=%s/%s retry=%s/%s",
                        self.current_init_state_idx,
                        wait_step + 1,
                        self.num_steps_wait,
                        warmup_retry + 1,
                        max_warmup_retries,
                    )
                    break
            if not warmup_terminated:
                break

        if warmup_terminated:
            self.logger.warning(
                "LIBERO warmup keeps terminating for init_state_idx=%s; skip warmup this reset",
                self.current_init_state_idx,
            )
            self.env.reset()
            obs = self.env.set_init_state(self.initial_states[self.current_init_state_idx])

        assert obs is not None
        self.logger.info(
            "LIBERO reset: suite=%s task_id=%s init_state_idx=%s seed=%s",
            self.suite_name,
            self.task_id,
            self.current_init_state_idx,
            applied_seed,
        )
        self._last_obs = _clone_obs_tree(obs)
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, reward, env_done, info = self.env.step(
            np.asarray(action, dtype=np.float32).tolist()
        )
        self._take_action_cnt += 1
        env_done = bool(env_done)
        truncated = bool((not env_done) and self._take_action_cnt >= self._step_limit)
        done = bool(env_done or truncated)
        success = bool(env_done)
        info_dict = dict(info) if isinstance(info, dict) else {}
        info_dict.update(
            {
                "success": success,
                "env_done": bool(env_done),
                "episode_done": bool(done),
                "step_limit_reached": bool(truncated),
                "take_action_cnt": int(self._take_action_cnt),
                "step_lim": int(self._step_limit),
                "task_description": self._task_description,
                "init_state_idx": self.current_init_state_idx,
            }
        )
        self._last_obs = _clone_obs_tree(obs)
        return obs, float(reward), bool(done), bool(truncated), info_dict

    def step_chunk(self, actions: np.ndarray) -> Dict[str, Any]:
        action_chunk = np.asarray(actions, dtype=np.float32)
        if action_chunk.ndim == 1:
            if action_chunk.size % self._action_dim != 0:
                raise ValueError(
                    "Flat action chunk size must be divisible by action_dim="
                    f"{self._action_dim}, got {action_chunk.shape}"
                )
            action_chunk = action_chunk.reshape(-1, self._action_dim)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self._action_dim:
            raise ValueError(f"Unexpected action chunk shape: {action_chunk.shape}")

        if self._last_obs is None:
            raise RuntimeError("step_chunk called before reset")

        observations: List[Dict[str, Any]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []

        truncated = False
        for step_action in action_chunk:
            prev_obs = _clone_obs_tree(self._last_obs)
            obs, reward, done, truncated, info = self.step(step_action)
            # OffScreen envs may reuse internal observation buffers.
            # Clone per-step observations so chunk history is immutable.
            next_obs = _clone_obs_tree(obs)
            observations.append(next_obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            info_dict = dict(info)
            infos.append(info_dict)
            steps.append(
                {
                    "obs": prev_obs,
                    "action": np.array(step_action, copy=True),
                    "next_obs": next_obs,
                    "reward": float(reward),
                    "env_done": bool(info_dict.get("env_done", False)),
                    "truncated": bool(truncated),
                    "done": bool(done),
                    "info": info_dict,
                }
            )
            if done or truncated:
                break

        if not observations:
            raise RuntimeError("step_chunk received an empty action chunk")

        return {
            "steps": steps,
            "obs": observations[-1],
            "observations": observations,
            "reward_sum": float(sum(rewards)),
            "rewards": rewards,
            "dones": dones,
            "done": bool(dones[-1]),
            "truncated": bool(truncated),
            "infos": infos,
            "info": dict(infos[-1]),
            "num_steps": int(len(rewards)),
        }

    def close(self, clear_cache: bool = False) -> None:
        del clear_cache
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
