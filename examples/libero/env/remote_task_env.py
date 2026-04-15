"""Remote LIBERO environment wrapper via HTTP RPC."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from serl_launcher.envs.remote_http import RemoteHttpRpcClient


class RemoteLiberoTaskEnv:
    def __init__(
        self,
        *,
        host: str,
        port: int,
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
        timeout_sec: float = 120.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.logger = logger or logging.getLogger(__name__)

        self.suite_name = str(suite_name)
        self.task_id = int(task_id)
        self._action_dim = int(action_dim) if action_dim is not None else 0
        self.resolution = int(resolution)
        self.num_steps_wait = int(num_steps_wait)
        self.max_episode_steps = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.libero_root = None if libero_root is None else str(libero_root)
        self.libero_config_dir = (
            None if libero_config_dir is None else str(libero_config_dir)
        )
        self.libero_datasets_root = (
            None if libero_datasets_root is None else str(libero_datasets_root)
        )
        self.env_seed = None if env_seed is None else int(env_seed)

        self.last_seed: Optional[int] = None
        self.current_init_state_idx: Optional[int] = None
        self._current_instruction: str = ""
        self._task_description: str = ""
        self._step_limit: int = 0
        self._take_action_cnt: int = 0
        self._rpc_client = RemoteHttpRpcClient(
            host=self.host,
            port=self.port,
            timeout_sec=self.timeout_sec,
            keep_alive=True,
            logger=self.logger,
        )

        try:
            self._rpc(
                "create_env",
                suite_name=self.suite_name,
                task_id=self.task_id,
                action_dim=(None if action_dim is None else int(action_dim)),
                resolution=self.resolution,
                num_steps_wait=self.num_steps_wait,
                max_episode_steps=self.max_episode_steps,
                libero_root=self.libero_root,
                libero_config_dir=self.libero_config_dir,
                libero_datasets_root=self.libero_datasets_root,
                env_seed=self.env_seed,
            )
            self._apply_meta(self._rpc("get_meta"))
        except Exception:
            self._rpc_client.close()
            raise

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

    def _rpc(self, method: str, **kwargs: Any) -> Any:
        return self._rpc_client.call(method, **kwargs)

    def _apply_meta(self, meta: Any) -> None:
        if not isinstance(meta, dict):
            return
        self._current_instruction = str(
            meta.get("current_instruction", self._current_instruction)
        )
        self._task_description = str(
            meta.get("task_description", self._task_description)
        )
        self._step_limit = int(meta.get("step_limit", self._step_limit))
        self._take_action_cnt = int(meta.get("take_action_cnt", self._take_action_cnt))
        if meta.get("action_dim", None) is not None:
            self._action_dim = int(meta.get("action_dim"))
        self.current_init_state_idx = meta.get(
            "current_init_state_idx", self.current_init_state_idx
        )
        last_seed = meta.get("last_seed", None)
        self.last_seed = int(last_seed) if last_seed is not None else None

    def reset(
        self,
        seed: int,
        init_episode_idx: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._rpc(
            "reset",
            seed=int(seed),
            init_episode_idx=int(init_episode_idx),
            episode_info=episode_info,
        )
        if not isinstance(result, dict) or "obs" not in result:
            raise RuntimeError("remote reset returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return result["obs"]

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        result = self._rpc("step", action=np.asarray(action, dtype=np.float32))
        if not isinstance(result, dict):
            raise RuntimeError("remote step returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return (
            result["obs"],
            float(result["reward"]),
            bool(result["done"]),
            bool(result["truncated"]),
            dict(result["info"]),
        )

    def step_chunk(self, actions: np.ndarray) -> Dict[str, Any]:
        result = self._rpc("step_chunk", actions=np.asarray(actions, dtype=np.float32))
        if not isinstance(result, dict):
            raise RuntimeError("remote step_chunk returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return result

    def close(self, clear_cache: bool = False) -> None:
        try:
            self._rpc("close", clear_cache=bool(clear_cache))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("remote env close failed: %s", exc)
        finally:
            self._rpc_client.close()
