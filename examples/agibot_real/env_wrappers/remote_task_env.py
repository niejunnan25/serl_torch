"""Remote AgiBot environment wrapper via HTTP RPC."""
from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

import numpy as np

from serl_launcher.envs.remote_http import RemoteHttpRpcClient


class RemoteAgiBotTaskEnv:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        task_name: str,
        prompt: str,
        action_dim: int,
        control_mode: str = "camera_position",
        hz: float = 20.0,
        use_smooth_trajectory: bool = False,
        trajectory_time: Optional[float] = None,
        max_episode_steps: Optional[int] = None,
        retargeter_urdf_path: Optional[str] = None,
        retargeter_camera_extrinsic_path: Optional[str] = None,
        controller: Optional[Mapping[str, Any]] = None,
        reset_hook: Optional[str] = None,
        success_hook: Optional[str] = None,
        expert_precheck_hook: Optional[str] = None,
        timeout_sec: float = 180.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.logger = logger or logging.getLogger(__name__)

        self.task_name = str(task_name)
        self.prompt = str(prompt)
        self._action_dim = int(action_dim)
        self.control_mode = str(control_mode)
        self.hz = float(hz)
        self.use_smooth_trajectory = bool(use_smooth_trajectory)
        self.trajectory_time = None if trajectory_time is None else float(trajectory_time)
        self.max_episode_steps = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.retargeter_urdf_path = (
            None if retargeter_urdf_path is None else str(retargeter_urdf_path)
        )
        self.retargeter_camera_extrinsic_path = (
            None
            if retargeter_camera_extrinsic_path is None
            else str(retargeter_camera_extrinsic_path)
        )
        self.controller = dict(controller or {})
        self.reset_hook = None if reset_hook is None else str(reset_hook)
        self.success_hook = None if success_hook is None else str(success_hook)
        self.expert_precheck_hook = (
            None if expert_precheck_hook is None else str(expert_precheck_hook)
        )

        self.last_seed: Optional[int] = None
        self.current_init_state_idx: Optional[int] = None
        self._current_instruction: str = self.prompt
        self._task_description: str = self.prompt
        self._step_limit: int = int(max_episode_steps or 0)
        self._take_action_cnt: int = 0
        self._controller_meta: Dict[str, Any] = {
            "enabled": bool(self.controller.get("enabled", False))
        }
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
                task_name=self.task_name,
                prompt=self.prompt,
                action_dim=self._action_dim,
                control_mode=self.control_mode,
                hz=self.hz,
                use_smooth_trajectory=self.use_smooth_trajectory,
                trajectory_time=self.trajectory_time,
                max_episode_steps=self.max_episode_steps,
                retargeter_urdf_path=self.retargeter_urdf_path,
                retargeter_camera_extrinsic_path=self.retargeter_camera_extrinsic_path,
                controller=self.controller,
                reset_hook=self.reset_hook,
                success_hook=self.success_hook,
                expert_precheck_hook=self.expert_precheck_hook,
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

    @property
    def controller_enabled(self) -> bool:
        return bool(self._controller_meta.get("enabled", False))

    def _rpc(self, method: str, **kwargs: Any) -> Any:
        return self._rpc_client.call(method, **kwargs)

    def _apply_meta(self, meta: Any) -> None:
        if not isinstance(meta, dict):
            return
        self._current_instruction = str(
            meta.get("current_instruction", self._current_instruction)
        )
        self._task_description = str(meta.get("task_description", self._task_description))
        self._step_limit = int(meta.get("step_limit", self._step_limit))
        self._take_action_cnt = int(meta.get("take_action_cnt", self._take_action_cnt))
        if meta.get("action_dim", None) is not None:
            self._action_dim = int(meta.get("action_dim"))
        self.current_init_state_idx = meta.get(
            "current_init_state_idx",
            self.current_init_state_idx,
        )
        last_seed = meta.get("last_seed", None)
        self.last_seed = int(last_seed) if last_seed is not None else None
        controller_meta = meta.get("controller", None)
        if isinstance(controller_meta, dict):
            self._controller_meta = dict(controller_meta)

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
        self,
        seed: int,
        init_episode_idx: int,
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
        self,
        action: np.ndarray,
    ) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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

    def get_controller_meta(self) -> Dict[str, Any]:
        result = self._rpc("get_controller_meta")
        if not isinstance(result, dict):
            raise RuntimeError("remote get_controller_meta returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def request_ready(self) -> Dict[str, Any]:
        result = self._rpc("request_ready")
        if not isinstance(result, dict):
            raise RuntimeError("remote request_ready returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def request_pause(self) -> Dict[str, Any]:
        result = self._rpc("request_pause")
        if not isinstance(result, dict):
            raise RuntimeError("remote request_pause returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def request_reset(self) -> Dict[str, Any]:
        result = self._rpc("request_reset")
        if not isinstance(result, dict):
            raise RuntimeError("remote request_reset returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def mark_success(self) -> Dict[str, Any]:
        result = self._rpc("mark_success")
        if not isinstance(result, dict):
            raise RuntimeError("remote mark_success returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def mark_fail(self) -> Dict[str, Any]:
        result = self._rpc("mark_fail")
        if not isinstance(result, dict):
            raise RuntimeError("remote mark_fail returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return dict(result.get("controller", {}))

    def get_latest_obs(self) -> Dict[str, Any]:
        result = self._rpc("get_latest_obs")
        if not isinstance(result, dict) or "obs" not in result:
            raise RuntimeError("remote get_latest_obs returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        return result["obs"]

    def enqueue_action_chunk(self, actions: np.ndarray) -> list[int]:
        result = self._rpc(
            "enqueue_action_chunk",
            actions=np.asarray(actions, dtype=np.float32),
        )
        if not isinstance(result, dict):
            raise RuntimeError("remote enqueue_action_chunk returned invalid payload")
        self._apply_meta(result.get("meta", {}))
        ids = result.get("sequence_ids", [])
        return [int(v) for v in ids]

    def poll_controller_transitions(self, *, max_items: int = 64) -> list[Dict[str, Any]]:
        result = self._rpc("poll_controller_transitions", max_items=int(max_items))
        if not isinstance(result, dict):
            raise RuntimeError(
                "remote poll_controller_transitions returned invalid payload"
            )
        self._apply_meta(result.get("meta", {}))
        payloads = result.get("transitions", [])
        if not isinstance(payloads, list):
            raise RuntimeError(
                "remote poll_controller_transitions payload must be a list"
            )
        return [dict(v) for v in payloads]
