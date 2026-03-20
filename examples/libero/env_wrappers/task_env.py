"""Local LIBERO task environment wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .setup import (
    resolve_libero_config_dir,
    resolve_libero_datasets_root,
    resolve_libero_root,
    resolve_max_episode_steps,
    setup_libero_pythonpath,
)


class LiberoTaskEnv:
    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        resolution: int = 256,
        num_steps_wait: int = 10,
        max_episode_steps: Optional[int] = None,
        libero_root: Optional[str] = None,
        openpi_root: Optional[str] = None,
        libero_config_dir: Optional[str] = None,
        libero_datasets_root: Optional[str] = None,
        env_seed_mode: str = "per_episode",
        fixed_env_seed: Optional[int] = None,
        init_state_index_mode: str = "seed",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.suite_name = str(suite_name)
        self.task_id = int(task_id)
        self.resolution = int(resolution)
        self.num_steps_wait = int(num_steps_wait)
        self.env_seed_mode = str(env_seed_mode).lower()
        self.fixed_env_seed = None if fixed_env_seed is None else int(fixed_env_seed)
        self.init_state_index_mode = str(init_state_index_mode).lower()
        if self.env_seed_mode not in {"per_episode", "fixed"}:
            raise ValueError(f"Unsupported env_seed_mode: {env_seed_mode}")
        if self.init_state_index_mode not in {"seed", "episode_id"}:
            raise ValueError(f"Unsupported init_state_index_mode: {init_state_index_mode}")
        self.libero_root = resolve_libero_root(libero_root, openpi_root=openpi_root)
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

        task_bddl_file = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.resolution,
            "camera_widths": self.resolution,
        }
        self.env = OffScreenRenderEnv(**env_args)
        if self.env_seed_mode == "fixed":
            if self.fixed_env_seed is None:
                raise ValueError("fixed_env_seed must be provided when env_seed_mode='fixed'")
            self.env.seed(self.fixed_env_seed)

        self._step_limit = int(
            max_episode_steps if max_episode_steps is not None else resolve_max_episode_steps(self.suite_name)
        )
        self._take_action_cnt = 0
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

    def expert_precheck(self, seed: int, episode_id: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        del seed, episode_id
        return True, None

    def reset(
        self,
        seed: int,
        episode_id: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del episode_info
        applied_seed = int(seed)
        if self.env_seed_mode == "fixed":
            applied_seed = int(self.fixed_env_seed)
        self.last_seed = int(applied_seed)
        if self.init_state_index_mode == "episode_id" and int(episode_id) >= 0:
            self.current_init_state_idx = int(episode_id) % len(self.initial_states)
        else:
            self.current_init_state_idx = int(seed) % len(self.initial_states)
        self._take_action_cnt = 0

        if self.env_seed_mode != "fixed":
            self.env.seed(int(seed))
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[self.current_init_state_idx])
        dummy_action = [0.0] * 6 + [-1.0]
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(dummy_action)
        self.logger.info(
            "LIBERO reset: suite=%s task_id=%s episode_id=%s init_state_idx=%s seed=%s",
            self.suite_name,
            self.task_id,
            episode_id,
            self.current_init_state_idx,
            applied_seed,
        )
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32).tolist())
        self._take_action_cnt += 1
        success = bool(done)
        info_dict = dict(info) if isinstance(info, dict) else {}
        info_dict.update(
            {
                "success": success,
                "take_action_cnt": int(self._take_action_cnt),
                "step_lim": int(self._step_limit),
                "task_description": self._task_description,
                "init_state_idx": self.current_init_state_idx,
            }
        )
        return obs, float(reward), bool(done), False, info_dict

    def close(self, clear_cache: bool = False) -> None:
        del clear_cache
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
