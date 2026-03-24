"""Remote LIBERO environment wrapper via HTTP RPC."""
from __future__ import annotations

import http.client
import logging
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _sanitize_for_pickle(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _sanitize_for_pickle(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_pickle(val) for val in value]
    return value


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
        openpi_root: Optional[str] = None,
        libero_config_dir: Optional[str] = None,
        libero_datasets_root: Optional[str] = None,
        env_seed_mode: str = "per_episode",
        fixed_env_seed: Optional[int] = None,
        init_state_index_mode: str = "seed",
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
        self.openpi_root = None if openpi_root is None else str(openpi_root)
        self.libero_config_dir = (
            None if libero_config_dir is None else str(libero_config_dir)
        )
        self.libero_datasets_root = (
            None if libero_datasets_root is None else str(libero_datasets_root)
        )
        self.env_seed_mode = str(env_seed_mode)
        self.fixed_env_seed = None if fixed_env_seed is None else int(fixed_env_seed)
        self.init_state_index_mode = str(init_state_index_mode)

        self.last_seed: Optional[int] = None
        self.current_init_state_idx: Optional[int] = None
        self._current_instruction: str = ""
        self._task_description: str = ""
        self._step_limit: int = 0
        self._take_action_cnt: int = 0
        self._conn: Optional[http.client.HTTPConnection] = None

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
                openpi_root=self.openpi_root,
                libero_config_dir=self.libero_config_dir,
                libero_datasets_root=self.libero_datasets_root,
                env_seed_mode=self.env_seed_mode,
                fixed_env_seed=self.fixed_env_seed,
                init_state_index_mode=self.init_state_index_mode,
            )
            self._apply_meta(self._rpc("get_meta"))
        except Exception:
            self._close_conn()
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

    def _close_conn(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=self.timeout_sec,
            )
        return self._conn

    def _reconnect(self) -> http.client.HTTPConnection:
        self._close_conn()
        return self._ensure_conn()

    def _rpc_once(self, method: str, payload: bytes) -> Any:
        conn = self._ensure_conn()
        conn.request(
            "POST",
            "/rpc",
            body=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Connection": "keep-alive",
            },
        )
        resp = conn.getresponse()
        try:
            resp_bytes = resp.read()
        finally:
            if getattr(resp, "will_close", False):
                self._close_conn()

        if resp.status != 200:
            self._close_conn()
            raise RuntimeError(
                f"remote env rpc failed method={method} status={resp.status} reason={resp.reason}"
            )
        data = pickle.loads(resp_bytes)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"remote env rpc invalid response type for method={method}"
            )
        if not bool(data.get("ok", False)):
            err = str(data.get("error", "unknown remote error"))
            raise RuntimeError(f"remote env rpc method={method} failed: {err}")
        return data.get("result", None)

    def _rpc(self, method: str, **kwargs: Any) -> Any:
        payload = pickle.dumps(
            {"method": method, "kwargs": _sanitize_for_pickle(kwargs)},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            try:
                return self._rpc_once(method, payload)
            except (
                OSError,
                EOFError,
                http.client.HTTPException,
                pickle.PickleError,
            ) as exc:
                last_exc = exc
                self._close_conn()
                if attempt == 0:
                    self.logger.debug(
                        "remote env rpc reconnect: method=%s error=%s",
                        method,
                        exc,
                    )
                    self._reconnect()
                    continue
                break

        assert last_exc is not None
        raise RuntimeError(
            f"remote env rpc method={method} transport error: {last_exc}"
        ) from last_exc

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

    def expert_precheck(
        self, seed: int, init_episode_idx: int
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        result = self._rpc(
            "expert_precheck",
            seed=int(seed),
            init_episode_idx=int(init_episode_idx),
        )
        if not isinstance(result, dict):
            raise RuntimeError("remote expert_precheck returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        passed = bool(result.get("passed", False))
        episode_info = result.get("episode_info", None)
        if episode_info is not None and (not isinstance(episode_info, dict)):
            episode_info = None
        return passed, episode_info

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
            self._close_conn()
