"""Local AgiBot real-robot task environment."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple

import numpy as np

from .controller import ExecutedTransition
from .controller import ManualEpisodeController
from .controller import STATE_RUNNING
from .controller import TERMINAL_FAIL
from .controller import TERMINAL_HOOK
from .controller import TERMINAL_RESET
from .controller import TERMINAL_SUCCESS
from .controller import TERMINAL_TIMEOUT
from ..robot.hooks import call_optional_hook
from ..robot.hooks import resolve_hook
from ..robot.interface import AgiBotRobotNode
from ..robot.retargeter import BodyRetargeter


def _clone_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_obs_tree(v) for k, v in value.items()}
    return np.array(value, copy=True)


def _resolve_robot_asset_path(
    explicit_path: Optional[str],
    *,
    assets_root: Optional[str],
    default_name: str,
) -> str:
    if explicit_path is not None:
        return str(Path(str(explicit_path)).expanduser().resolve())
    if assets_root is not None:
        return str(
            (Path(str(assets_root)).expanduser() / "G1" / default_name).resolve()
        )
    return str(
        (Path(__file__).resolve().parents[1] / "assets" / "G1" / default_name).resolve()
    )


class AgiBotTaskEnv:
    def __init__(
        self,
        *,
        task_name: str,
        prompt: str,
        action_dim: int = 14,
        control_mode: str = "camera_position",
        hz: float = 20.0,
        use_smooth_trajectory: bool = False,
        trajectory_time: Optional[float] = None,
        max_episode_steps: Optional[int] = None,
        assets_root: Optional[str] = None,
        retargeter_urdf_path: Optional[str] = None,
        retargeter_camera_extrinsic_path: Optional[str] = None,
        controller: Optional[Mapping[str, Any]] = None,
        reset_hook: Optional[str] = None,
        success_hook: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = str(task_name)
        self._current_instruction = str(prompt)
        self._task_description = str(prompt)
        self.control_mode = str(control_mode)
        self.hz = float(hz)
        self.use_smooth_trajectory = bool(use_smooth_trajectory)
        self.trajectory_time = (
            float(trajectory_time)
            if trajectory_time is not None
            else (1.0 / self.hz) * 2.0
        )
        self._action_dim = int(action_dim)
        if self.control_mode != "camera_position":
            raise ValueError(
                f"AgiBot residual RL currently supports only camera_position mode, got {control_mode!r}"
            )
        if self._action_dim != 14:
            raise ValueError(
                f"AgiBot camera_position mode expects env.action_dim=14, got {self._action_dim}"
            )
        retargeter_urdf_path = _resolve_robot_asset_path(
            retargeter_urdf_path,
            assets_root=assets_root,
            default_name="model.urdf",
        )
        retargeter_camera_extrinsic_path = _resolve_robot_asset_path(
            retargeter_camera_extrinsic_path,
            assets_root=assets_root,
            default_name="head_extrinsic_ours.json",
        )

        self.robot_node = AgiBotRobotNode(hz=self.hz)
        self.retargeter = BodyRetargeter(
            urdf_path=retargeter_urdf_path,
            camera_extrinsic_path=retargeter_camera_extrinsic_path,
        )
        self._step_limit = (
            int(max_episode_steps) if max_episode_steps is not None else 200
        )
        self._take_action_cnt = 0
        self.episode_count = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self._controller_cfg = dict(controller or {})
        self._controller_enabled = bool(self._controller_cfg.get("enabled", False))

        self._reset_hook = resolve_hook(reset_hook)
        self._success_hook_spec = (
            None if success_hook is None else str(success_hook).strip() or None
        )
        if self._success_hook_spec is not None:
            self.logger.warning(
                "task.success_hook is currently retained for compatibility only and "
                "is not used by the canonical AgiBot residual train/eval flow: %s",
                self._success_hook_spec,
            )
        self._controller: Optional[ManualEpisodeController] = None
        self._control_thread: Optional[threading.Thread] = None
        self._control_stop = threading.Event()
        if self._controller_enabled:
            key_cfg = self._controller_cfg.get("keys", {})
            self._controller = ManualEpisodeController(
                enabled=True,
                interface=str(self._controller_cfg.get("interface", "terminal")),
                poll_interval_sec=float(
                    self._controller_cfg.get("poll_interval_sec", 0.05)
                ),
                keys=key_cfg if isinstance(key_cfg, Mapping) else None,
                logger=self.logger,
            )
            try:
                self._controller.set_latest_obs(self._get_obs())
            except Exception:  # noqa: BLE001
                pass
            self._controller.start_operator_interface()
            self._control_thread = threading.Thread(
                target=self._controller_loop,
                name="agibot-controller-loop",
                daemon=True,
            )
            self._control_thread.start()

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
        return bool(self._controller_enabled)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        img_head = self.robot_node.get_img_head()
        img_left = self.robot_node.get_img_left_wrist()
        img_right = self.robot_node.get_img_right_wrist()
        joint_state = self.robot_node.get_joint_state()
        if any(x is None for x in (img_head, img_left, img_right, joint_state)):
            if self._last_obs is not None:
                return self._last_obs
            raise RuntimeError("Unable to fetch AgiBot sensor data")

        obs = {
            "image/head": np.asarray(img_head, dtype=np.uint8),
            "image/left_wrist": np.asarray(img_left, dtype=np.uint8),
            "image/right_wrist": np.asarray(img_right, dtype=np.uint8),
            "state/joint": np.asarray(joint_state, dtype=np.float32),
        }
        obs.update(self._compute_pose_state(obs["state/joint"]))
        self._last_obs = obs
        return obs

    def _compute_pose_state(self, joint_state: np.ndarray) -> Dict[str, np.ndarray]:
        head_states = np.asarray(
            self.robot_node.get_head_joint_states(), dtype=np.float32
        )
        waist_states = np.asarray(
            self.robot_node.get_waist_joint_states(), dtype=np.float32
        )
        arm_states = np.asarray(
            self.robot_node.get_arm_joint_states(), dtype=np.float32
        )

        state_vec = np.zeros((1, 53), dtype=np.float32)
        state_vec[0, 28:35] = arm_states[:7]
        state_vec[0, 35:42] = arm_states[7:]
        state_vec[0, 42:43] = joint_state[7]
        state_vec[0, 43:44] = joint_state[15]
        state_vec[0, 51:53] = waist_states
        state_vec[0, 26:28] = head_states

        (left_pos, left_axisangle), (
            right_pos,
            right_axisangle,
        ) = self.retargeter.process_kinematics(state_vec)
        pose = np.concatenate(
            [
                np.asarray(left_pos, dtype=np.float32).reshape(3),
                np.asarray(left_axisangle, dtype=np.float32).reshape(3),
                np.asarray([joint_state[7]], dtype=np.float32),
                np.asarray(right_pos, dtype=np.float32).reshape(3),
                np.asarray(right_axisangle, dtype=np.float32).reshape(3),
                np.asarray([joint_state[15]], dtype=np.float32),
            ]
        )
        return {
            "state/pose": pose.astype(np.float32),
            "state/head": head_states.astype(np.float32),
            "state/waist": waist_states.astype(np.float32),
        }

    def _step_cartesian(self, action: np.ndarray) -> None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 14:
            raise ValueError(
                f"camera_position action must be 14D, got {action_arr.shape}"
            )
        hand_left, hand_right = float(action_arr[6]), float(action_arr[13])
        head_states = np.asarray(
            self.robot_node.get_head_joint_states(), dtype=np.float32
        )
        waist_states = np.asarray(
            self.robot_node.get_waist_joint_states(), dtype=np.float32
        )
        arm_states = np.asarray(
            self.robot_node.get_arm_joint_states(), dtype=np.float32
        )

        action_vec = np.zeros((1, 53), dtype=np.float32)
        action_vec[0, 51:53] = waist_states
        action_vec[0, 26:28] = head_states

        left_pos = action_arr[:3].reshape(1, 3)
        left_aa = action_arr[3:6].reshape(1, 3)
        right_pos = action_arr[7:10].reshape(1, 3)
        right_aa = action_arr[10:13].reshape(1, 3)

        (left_pos_base, left_euler), (
            right_pos_base,
            right_euler,
        ) = self.retargeter.inverse_kinematics_from_camera_axisangle(
            left_pos,
            left_aa,
            right_pos,
            right_aa,
            action_vec,
        )
        abs_action = np.concatenate(
            [
                left_pos_base[0],
                left_euler[0],
                right_pos_base[0],
                right_euler[0],
            ]
        )
        action_abs = {
            "observation_timestamp": int(time.time() * 1e9),
            "head_joint_states": np.rad2deg(head_states).tolist(),
            "waist_joint_states": waist_states.tolist(),
            "arm_joint_states": arm_states.tolist(),
            "arm_cmd": [abs_action.tolist()],
        }
        hand_action = np.asarray([hand_left, hand_right], dtype=np.float32)
        self.robot_node.publish_abs_pose_command_and_hand(
            action_abs,
            hand_action,
            trajectory_reference_time=self.trajectory_time,
        )

    def _controller_error_obs(self) -> Dict[str, Any]:
        obs = self._controller_cached_obs()
        if obs:
            return obs
        try:
            return self._get_obs()
        except Exception as obs_exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to refresh AgiBot observation after controller error: %s",
                obs_exc,
            )
        return {}

    def _controller_cached_obs(self) -> Dict[str, Any]:
        if self._controller is not None:
            obs = self._controller.get_latest_obs()
            if obs is not None:
                return obs
        if self._last_obs is not None:
            return _clone_obs_tree(self._last_obs)
        return {}

    def _handle_controller_step_exception(
        self,
        *,
        sequence_id: int,
        exc: Exception,
    ) -> None:
        assert self._controller is not None
        error_message = f"{type(exc).__name__}: {exc}"
        self.logger.exception(
            "Controller step failed for sequence_id=%s: %s",
            int(sequence_id),
            error_message,
        )
        self._controller.push_transition(
            ExecutedTransition(
                sequence_id=int(sequence_id),
                obs=self._controller_error_obs(),
                reward=0.0,
                done=True,
                truncated=False,
                info={
                    "success": False,
                    "controller_error": error_message,
                    "controller_sequence_id": int(sequence_id),
                    "take_action_cnt": int(self._take_action_cnt),
                    "step_lim": int(self._step_limit),
                    "task_description": self._task_description,
                    "task_name": self.task_name,
                },
            )
        )

    def _controller_loop(self) -> None:
        assert self._controller is not None
        period_sec = max(1.0 / max(float(self.hz), 1.0), 0.001)
        while not self._control_stop.is_set():
            loop_t0 = time.perf_counter()
            queued = self._controller.pop_next_action()
            if queued is not None:
                try:
                    self._step_cartesian(queued.action)
                    self._take_action_cnt += 1
                    obs = self._get_obs()
                    step_result = self._resolve_step_result(
                        obs=obs,
                        action=queued.action,
                        controller_mode=True,
                    )
                    if bool(step_result["truncated"]) and bool(
                        step_result["info"].get("time_limit_reached", False)
                    ):
                        self._controller.mark_timeout(info=step_result["info"])
                    self._controller.push_transition(
                        ExecutedTransition(
                            sequence_id=int(queued.sequence_id),
                            obs=obs,
                            reward=float(step_result["reward"]),
                            done=bool(step_result["done"]),
                            truncated=bool(step_result["truncated"]),
                            info=dict(step_result["info"]),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self._handle_controller_step_exception(
                        sequence_id=int(queued.sequence_id),
                        exc=exc,
                    )
            sleep_sec = max(0.0, period_sec - (time.perf_counter() - loop_t0))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    def _resolve_step_result(
        self,
        *,
        obs: Dict[str, Any],
        action: np.ndarray,
        controller_mode: bool,
    ) -> Dict[str, Any]:
        info_dict: Dict[str, Any] = {
            "success": False,
            "take_action_cnt": int(self._take_action_cnt),
            "step_lim": int(self._step_limit),
            "task_description": self._task_description,
            "task_name": self.task_name,
        }
        reward = 0.0
        done = False
        truncated = False
        success = False

        if controller_mode and self._controller is not None:
            ctrl_meta = self._controller.get_meta()
            terminal_signal = ctrl_meta.get("terminal_signal", None)
            terminal_info = ctrl_meta.get("terminal_info", {})
            if terminal_signal == TERMINAL_SUCCESS:
                reward = 1.0
                done = True
                truncated = False
                success = True
                info_dict.update(dict(terminal_info))
                info_dict["controller_terminal_signal"] = TERMINAL_SUCCESS
                info_dict["human_success"] = True
            elif terminal_signal == TERMINAL_FAIL:
                reward = 0.0
                done = True
                truncated = False
                success = False
                info_dict.update(dict(terminal_info))
                info_dict["controller_terminal_signal"] = TERMINAL_FAIL
                info_dict["human_fail"] = True
            elif terminal_signal == TERMINAL_RESET:
                reward = 0.0
                done = False
                truncated = True
                success = False
                info_dict.update(dict(terminal_info))
                info_dict["controller_terminal_signal"] = TERMINAL_RESET
                info_dict["human_reset"] = True

        info_dict["success"] = success
        if (not done) and (not truncated) and self._take_action_cnt >= self._step_limit:
            reward = 0.0
            truncated = True
            info_dict["controller_terminal_signal"] = TERMINAL_TIMEOUT
            info_dict["time_limit_reached"] = True
        return {
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "success": bool(success),
            "info": info_dict,
        }

    def get_controller_meta(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            return {
                "enabled": False,
                "state": None,
                "terminal_signal": None,
                "queue_depth": 0,
            }
        return self._controller.get_meta()

    def request_ready(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.request_ready()

    def request_pause(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.request_pause()

    def request_reset(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.request_reset()

    def mark_success(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.mark_success()

    def mark_fail(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.mark_fail()

    def get_latest_obs(self) -> Dict[str, Any]:
        if not self._controller_enabled or self._controller is None:
            return self._get_obs()
        obs = self._get_obs()
        self._controller.set_latest_obs(obs)
        return obs

    def enqueue_action_chunk(self, actions: np.ndarray) -> list[int]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        return self._controller.enqueue_action_chunk(actions)

    def poll_controller_transitions(
        self, *, max_items: int = 64
    ) -> list[Dict[str, Any]]:
        if not self._controller_enabled or self._controller is None:
            raise RuntimeError("controller mode is disabled")
        transitions = self._controller.poll_transitions(max_items=max_items)
        payloads: list[Dict[str, Any]] = []
        for transition in transitions:
            payloads.append(
                {
                    "sequence_id": int(transition.sequence_id),
                    "obs": _clone_obs_tree(transition.obs),
                    "reward": float(transition.reward),
                    "done": bool(transition.done),
                    "truncated": bool(transition.truncated),
                    "info": dict(transition.info),
                }
            )
        return payloads

    def _override_transition_for_terminal_meta(
        self,
        transition: Dict[str, Any],
        meta: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(transition)
        info = dict(payload.get("info", {}))
        terminal_signal = meta.get("terminal_signal", None)
        terminal_info = meta.get("terminal_info", {})
        if isinstance(terminal_info, Mapping):
            info.update(dict(terminal_info))
        if terminal_signal is not None:
            info["controller_terminal_signal"] = str(terminal_signal)
        if terminal_signal == TERMINAL_SUCCESS:
            payload["reward"] = 1.0
            payload["done"] = True
            payload["truncated"] = False
            info["success"] = True
            info["human_success"] = True
        elif terminal_signal == TERMINAL_FAIL:
            payload["reward"] = 0.0
            payload["done"] = True
            payload["truncated"] = False
            info["success"] = False
            info["human_fail"] = True
        elif terminal_signal == TERMINAL_RESET:
            payload["reward"] = 0.0
            payload["done"] = False
            payload["truncated"] = True
            info["success"] = False
            info["human_reset"] = True
        elif terminal_signal == TERMINAL_TIMEOUT:
            payload["reward"] = 0.0
            payload["done"] = False
            payload["truncated"] = True
            info["success"] = False
            info["time_limit_reached"] = True
        elif terminal_signal == TERMINAL_HOOK:
            info.setdefault("success", bool(info.get("success", False)))
        payload["info"] = info
        return payload

    def _is_completed_terminal_meta(self, meta: Mapping[str, Any]) -> bool:
        return (
            meta.get("terminal_signal", None)
            in {
                TERMINAL_SUCCESS,
                TERMINAL_FAIL,
                TERMINAL_RESET,
                TERMINAL_TIMEOUT,
            }
            and meta.get("state", None) != STATE_RUNNING
        )

    def _synthesize_terminal_transition(
        self,
        *,
        meta: Mapping[str, Any],
        sequence_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        terminal_signal = meta.get("terminal_signal", None)
        info: Dict[str, Any] = {
            "success": False,
            "take_action_cnt": int(self._take_action_cnt),
            "step_lim": int(self._step_limit),
            "task_description": self._task_description,
            "task_name": self.task_name,
            "controller_action_executed": False,
            "controller_terminated_before_execution": True,
        }
        if terminal_signal is not None:
            info["controller_terminal_signal"] = str(terminal_signal)
        if sequence_id is not None:
            info["controller_sequence_id"] = int(sequence_id)
        payload = {
            "sequence_id": int(sequence_id) if sequence_id is not None else -1,
            "obs": self._controller_cached_obs(),
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": info,
        }
        return self._override_transition_for_terminal_meta(payload, meta)

    def _controller_execute_chunk_blocking(
        self, actions: np.ndarray
    ) -> list[Dict[str, Any]]:
        assert self._controller is not None
        action_chunk = np.asarray(actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected 2-D action chunk, got {action_chunk.shape}")
        accepted_ids = self.enqueue_action_chunk(action_chunk)
        if not accepted_ids:
            meta = self.get_controller_meta()
            if self._is_completed_terminal_meta(meta):
                return [self._synthesize_terminal_transition(meta=meta)]
            return []
        accepted_cursor = 0
        transitions: list[Dict[str, Any]] = []
        pending: Optional[Dict[str, Any]] = None
        while True:
            polled = self.poll_controller_transitions(max_items=len(accepted_ids))
            for payload in polled:
                if accepted_cursor >= len(accepted_ids):
                    raise RuntimeError(
                        "controller returned more transitions than accepted actions"
                    )
                expected_sequence_id = int(accepted_ids[accepted_cursor])
                observed_sequence_id = int(payload["sequence_id"])
                if expected_sequence_id != observed_sequence_id:
                    raise RuntimeError(
                        "controller transition sequence mismatch: "
                        f"expected={expected_sequence_id} observed={observed_sequence_id}"
                    )
                accepted_cursor += 1
                if pending is not None:
                    transitions.append(pending)
                pending = payload
                if bool(payload["done"]) or bool(payload["truncated"]):
                    transitions.append(pending)
                    return transitions
            meta = self.get_controller_meta()
            next_expected_sequence_id = (
                int(accepted_ids[accepted_cursor])
                if accepted_cursor < len(accepted_ids)
                else None
            )
            inflight_sequence_id = meta.get("inflight_sequence_id", None)
            if inflight_sequence_id is not None:
                try:
                    inflight_sequence_id = int(inflight_sequence_id)
                except (TypeError, ValueError):
                    inflight_sequence_id = None
            if self._is_completed_terminal_meta(meta):
                if (
                    next_expected_sequence_id is not None
                    and inflight_sequence_id == next_expected_sequence_id
                ):
                    time.sleep(
                        float(
                            self._controller_cfg.get(
                                "poll_interval_sec",
                                0.05,
                            )
                        )
                    )
                    continue
                if pending is not None:
                    transitions.append(
                        self._override_transition_for_terminal_meta(pending, meta)
                    )
                else:
                    transitions.append(
                        self._synthesize_terminal_transition(
                            meta=meta,
                            sequence_id=next_expected_sequence_id,
                        )
                    )
                remaining_count = int(len(accepted_ids) - accepted_cursor)
                if remaining_count > 0:
                    self.logger.warning(
                        "controller terminal=%s canceled %s unexecuted queued actions",
                        meta.get("terminal_signal", None),
                        remaining_count,
                    )
                return transitions
            if accepted_cursor >= len(accepted_ids) and pending is not None:
                transitions.append(pending)
                return transitions
            time.sleep(
                float(
                    self._controller_cfg.get(
                        "poll_interval_sec",
                        0.05,
                    )
                )
            )

    def reset(
        self,
    ) -> Dict[str, Any]:
        self._take_action_cnt = 0
        self.episode_count += 1
        result = call_optional_hook(
            self._reset_hook,
            env=self,
            task_name=self.task_name,
            prompt=self._current_instruction,
        )
        if isinstance(result, dict):
            prompt = result.get("prompt", None)
            if prompt is not None:
                self._current_instruction = str(prompt)
                self._task_description = str(prompt)
        obs = self._get_obs()
        if self._controller_enabled and self._controller is not None:
            self._controller.start_episode()
            self._controller.set_latest_obs(obs)
            self._controller.transition_after_reset()
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._controller_enabled and self._controller is not None:
            results = self._controller_execute_chunk_blocking(
                np.asarray(action, dtype=np.float32).reshape(1, -1)
            )
            if not results:
                raise RuntimeError("controller step produced no executed transition")
            payload = results[-1]
            return (
                payload["obs"],
                float(payload["reward"]),
                bool(payload["done"]),
                bool(payload["truncated"]),
                dict(payload["info"]),
            )
        self._step_cartesian(action)
        self._take_action_cnt += 1
        obs = self._get_obs()
        step_result = self._resolve_step_result(
            obs=obs,
            action=np.asarray(action, dtype=np.float32),
            controller_mode=False,
        )
        return (
            obs,
            float(step_result["reward"]),
            bool(step_result["done"]),
            bool(step_result["truncated"]),
            dict(step_result["info"]),
        )

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
        if self._controller_enabled and self._controller is not None:
            transitions = self._controller_execute_chunk_blocking(action_chunk)
            if not transitions:
                raise RuntimeError(
                    "controller step_chunk produced no executed transitions"
                )
            observations = [_clone_obs_tree(v["obs"]) for v in transitions]
            rewards = [float(v["reward"]) for v in transitions]
            dones = [bool(v["done"]) for v in transitions]
            infos = [dict(v["info"]) for v in transitions]
            truncated = bool(transitions[-1]["truncated"])
            return {
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

        observations: List[Dict[str, Any]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        truncated = False
        for step_action in action_chunk:
            obs, reward, done, truncated, info = self.step(step_action)
            observations.append(_clone_obs_tree(obs))
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(dict(info))
            if done or truncated:
                break
        if not observations:
            raise RuntimeError("step_chunk received an empty action chunk")

        return {
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
        self._control_stop.set()
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        if self._controller is not None:
            self._controller.shutdown()
        try:
            self.robot_node.shutdown()
        except Exception:  # noqa: BLE001
            pass
