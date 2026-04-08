"""Local AgiBot real-robot task environment wrapper."""
from __future__ import annotations

import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from ..robot.hooks import call_optional_hook
from ..robot.hooks import coerce_precheck_result
from ..robot.hooks import coerce_success_result
from ..robot.hooks import resolve_hook
from ..robot.interface import AgiBotRobotNode
from ..robot.retargeter import BodyRetargeter


def _clone_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_obs_tree(v) for k, v in value.items()}
    return np.array(value, copy=True)


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
        retargeter_urdf_path: Optional[str] = None,
        retargeter_camera_extrinsic_path: Optional[str] = None,
        reset_hook: Optional[str] = None,
        success_hook: Optional[str] = None,
        expert_precheck_hook: Optional[str] = None,
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
            float(trajectory_time) if trajectory_time is not None else (1.0 / self.hz) * 2.0
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
        if retargeter_urdf_path is None or retargeter_camera_extrinsic_path is None:
            raise ValueError("retargeter asset paths must be provided")

        self.robot_node = AgiBotRobotNode(hz=self.hz)
        self.retargeter = BodyRetargeter(
            urdf_path=retargeter_urdf_path,
            camera_extrinsic_path=retargeter_camera_extrinsic_path,
        )
        self._step_limit = int(max_episode_steps) if max_episode_steps is not None else 200
        self._take_action_cnt = 0
        self.last_seed: Optional[int] = None
        self.current_init_state_idx: Optional[int] = None
        self.episode_count = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self._reset_hook = resolve_hook(reset_hook)
        self._success_hook = resolve_hook(success_hook)
        self._expert_precheck_hook = resolve_hook(expert_precheck_hook)

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
        head_states = np.asarray(self.robot_node.get_head_joint_states(), dtype=np.float32)
        waist_states = np.asarray(self.robot_node.get_waist_joint_states(), dtype=np.float32)
        arm_states = np.asarray(self.robot_node.get_arm_joint_states(), dtype=np.float32)

        state_vec = np.zeros((1, 53), dtype=np.float32)
        state_vec[0, 28:35] = arm_states[:7]
        state_vec[0, 35:42] = arm_states[7:]
        state_vec[0, 42:43] = joint_state[7]
        state_vec[0, 43:44] = joint_state[15]
        state_vec[0, 51:53] = waist_states
        state_vec[0, 26:28] = head_states

        (left_pos, left_axisangle), (right_pos, right_axisangle) = self.retargeter.process_kinematics(
            state_vec
        )
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
            raise ValueError(f"camera_position action must be 14D, got {action_arr.shape}")
        hand_left, hand_right = float(action_arr[6]), float(action_arr[13])
        head_states = np.asarray(self.robot_node.get_head_joint_states(), dtype=np.float32)
        waist_states = np.asarray(self.robot_node.get_waist_joint_states(), dtype=np.float32)
        arm_states = np.asarray(self.robot_node.get_arm_joint_states(), dtype=np.float32)

        action_vec = np.zeros((1, 53), dtype=np.float32)
        action_vec[0, 51:53] = waist_states
        action_vec[0, 26:28] = head_states

        left_pos = action_arr[:3].reshape(1, 3)
        left_aa = action_arr[3:6].reshape(1, 3)
        right_pos = action_arr[7:10].reshape(1, 3)
        right_aa = action_arr[10:13].reshape(1, 3)

        (left_pos_base, left_euler), (right_pos_base, right_euler) = (
            self.retargeter.inverse_kinematics_from_camera_axisangle(
                left_pos,
                left_aa,
                right_pos,
                right_aa,
                action_vec,
            )
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

    def expert_precheck(
        self,
        seed: int,
        init_episode_idx: int,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        result = call_optional_hook(
            self._expert_precheck_hook,
            env=self,
            seed=int(seed),
            init_episode_idx=int(init_episode_idx),
            task_name=self.task_name,
            prompt=self._current_instruction,
        )
        return coerce_precheck_result(result)

    def reset(
        self,
        seed: int,
        init_episode_idx: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_seed = int(seed)
        self.current_init_state_idx = int(init_episode_idx)
        self._take_action_cnt = 0
        self.episode_count += 1
        result = call_optional_hook(
            self._reset_hook,
            env=self,
            seed=int(seed),
            init_episode_idx=int(init_episode_idx),
            episode_info=episode_info,
            task_name=self.task_name,
            prompt=self._current_instruction,
        )
        if isinstance(result, dict):
            prompt = result.get("prompt", None)
            if prompt is not None:
                self._current_instruction = str(prompt)
                self._task_description = str(prompt)
        return self._get_obs()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_cartesian(action)
        self._take_action_cnt += 1
        obs = self._get_obs()
        info_dict: Dict[str, Any] = {
            "success": False,
            "take_action_cnt": int(self._take_action_cnt),
            "step_lim": int(self._step_limit),
            "task_description": self._task_description,
            "task_name": self.task_name,
            "init_state_idx": self.current_init_state_idx,
        }
        success_result = coerce_success_result(
            call_optional_hook(
                self._success_hook,
                env=self,
                obs=obs,
                action=np.asarray(action, dtype=np.float32),
                step_count=int(self._take_action_cnt),
                step_limit=int(self._step_limit),
                task_name=self.task_name,
                prompt=self._current_instruction,
            )
        )
        reward = float(success_result["reward"])
        done = bool(success_result["done"])
        truncated = bool(success_result["truncated"])
        success = bool(success_result["success"])
        info_dict.update(success_result["info"])
        info_dict["success"] = success
        if (not done) and (not truncated) and self._take_action_cnt >= self._step_limit:
            truncated = True
            info_dict["time_limit_reached"] = True
        return obs, reward, done, truncated, info_dict

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
        try:
            self.robot_node.shutdown()
        except Exception:  # noqa: BLE001
            pass

