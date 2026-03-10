"""RoboTwin 远程环境客户端：在训练进程中通过 HTTP RPC 调用环境服务。"""
from __future__ import annotations

import http.client
import logging
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np


class RemoteRoboTwinTaskEnv:
    """
    RoboTwin 远程环境封装，接口与本地 `RoboTwinTaskEnv` 尽量保持一致。

    训练进程在 `serl_torch` 环境中运行；
    环境服务在 `robotwin2` 环境中运行。
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        task_name: str,
        task_args: Dict[str, Any],
        prompt: str,
        max_setup_retries: int = 5,
        instruction_type: str = "seen",
        robo_root: Optional[str] = None,
        timeout_sec: float = 120.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.logger = logger or logging.getLogger(__name__)

        self.task_name = str(task_name)
        self.task_args = dict(task_args)
        self.prompt = str(prompt)
        self.max_setup_retries = int(max_setup_retries)
        self.instruction_type = str(instruction_type)
        self.robo_root = None if robo_root is None else str(robo_root)

        self.last_seed: Optional[int] = None
        self._current_instruction: str = self.prompt
        self._step_limit: int = 0
        self._take_action_cnt: int = 0

        self._rpc(
            "create_env",
            task_name=self.task_name,
            task_args=self.task_args,
            prompt=self.prompt,
            max_setup_retries=self.max_setup_retries,
            instruction_type=self.instruction_type,
            robo_root=self.robo_root,
        )
        meta = self._rpc("get_meta")
        self._apply_meta(meta)

    @property
    def current_instruction(self) -> str:
        return self._current_instruction

    @property
    def step_limit(self) -> int:
        return int(self._step_limit)

    @property
    def take_action_cnt(self) -> int:
        return int(self._take_action_cnt)

    def _rpc(self, method: str, **kwargs: Any) -> Any:
        payload = pickle.dumps({"method": method, "kwargs": kwargs}, protocol=pickle.HIGHEST_PROTOCOL)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        try:
            conn.request(
                "POST",
                "/rpc",
                body=payload,
                headers={"Content-Type": "application/octet-stream"},
            )
            resp = conn.getresponse()
            resp_bytes = resp.read()
        finally:
            conn.close()

        if resp.status != 200:
            raise RuntimeError(
                f"remote env rpc failed method={method} status={resp.status} reason={resp.reason}"
            )
        data = pickle.loads(resp_bytes)
        if not isinstance(data, dict):
            raise RuntimeError(f"remote env rpc invalid response type for method={method}")
        if not bool(data.get("ok", False)):
            err = str(data.get("error", "unknown remote error"))
            raise RuntimeError(f"remote env rpc method={method} failed: {err}")
        return data.get("result", None)

    def _apply_meta(self, meta: Any) -> None:
        if not isinstance(meta, dict):
            return
        self._current_instruction = str(meta.get("current_instruction", self._current_instruction))
        self._step_limit = int(meta.get("step_limit", self._step_limit))
        self._take_action_cnt = int(meta.get("take_action_cnt", self._take_action_cnt))
        last_seed = meta.get("last_seed", None)
        self.last_seed = int(last_seed) if last_seed is not None else None

    def reset(
        self,
        seed: int,
        episode_id: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._rpc(
            "reset",
            seed=int(seed),
            episode_id=int(episode_id),
            episode_info=episode_info,
        )
        if not isinstance(result, dict) or "obs" not in result:
            raise RuntimeError("remote reset returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return result["obs"]

    def expert_precheck(self, seed: int, episode_id: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        result = self._rpc(
            "expert_precheck",
            seed=int(seed),
            episode_id=int(episode_id),
        )
        if not isinstance(result, dict):
            raise RuntimeError("remote expert_precheck returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        passed = bool(result.get("passed", False))
        episode_info = result.get("episode_info", None)
        if episode_info is not None and (not isinstance(episode_info, dict)):
            episode_info = None
        return passed, episode_info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        result = self._rpc(
            "step",
            action=np.asarray(action, dtype=np.float32),
        )
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

    def close(self, clear_cache: bool = False) -> None:
        try:
            self._rpc("close", clear_cache=bool(clear_cache))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("remote env close failed: %s", exc)
