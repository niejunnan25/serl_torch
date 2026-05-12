"""Keyboard-driven fake task env: same controller / step_chunk contract, no robot/SDK."""

from __future__ import annotations

import logging
import threading
import time
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


def _clone_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_obs_tree(v) for k, v in value.items()}
    return np.array(value, copy=True)


class AgiBotFakeTaskEnv:
    """Mirrors `AgiBotTaskEnv` contracts but synthesizes observations and ignores actions."""

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
        fake_seed: int = 0,
        fake_image_hw: tuple[int, int] = (480, 640),
    ) -> None:
        del assets_root, retargeter_urdf_path, retargeter_camera_extrinsic_path
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

        self._step_limit = (
            int(max_episode_steps) if max_episode_steps is not None else 200
        )
        self._take_action_cnt = 0
        self.episode_count = 0
        self._episode_reset_prepared = False
        self._last_obs: Optional[Dict[str, Any]] = None
        self._fake_rng = np.random.default_rng(int(fake_seed))
        self._fake_image_hw = (int(fake_image_hw[0]), int(fake_image_hw[1]))

        self._controller_cfg = dict(controller or {})
        self._controller_enabled = bool(self._controller_cfg.get("enabled", False))

        self._reset_hook = resolve_hook(reset_hook)
        self._success_hook_spec = (
            None if success_hook is None else str(success_hook).strip() or None
        )
        if self._success_hook_spec is not None:
            self.logger.warning(
                "task.success_hook is retained for compatibility only: %s",
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
                terminal_grace_sec=float(
                    self._controller_cfg.get("terminal_grace_sec", 0.15)
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
                name="agibot-fake-controller-loop",
                daemon=True,
            )
            self._control_thread.start()
            self._prearm_controller_for_dry_run()

    def _prearm_controller_for_dry_run(self) -> None:
        """Real env only calls `start_episode` from `reset()`; the actor may spend a long
        time loading JoyRA / weights before the first `reset()`, during which
        `episode_active` is still false and key `g` has no effect.

        Pre-arm so the controller accepts queued actions as soon as the policy rolls.
        Each real `reset()` will call `start_episode` again and re-enter RUNNING.
        """
        assert self._controller is not None
        try:
            obs = self._get_obs()
            self._controller.start_episode()
            self._controller.set_latest_obs(obs)
            self._controller.request_ready()
            self.logger.info(
                "fake env pre-armed controller to RUNNING before first training reset "
                "(dry-run; key g optional until episodes advance)"
            )
        except Exception:  # noqa: BLE001
            self.logger.warning(
                "fake env failed to pre-arm controller; wait for first env.reset()",
                exc_info=True,
            )

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
        h, w = self._fake_image_hw
        phase = float(self._take_action_cnt) * 0.05 + float(self.episode_count) * 0.01
        base_u8 = int((self._take_action_cnt + self.episode_count * 17) % 200) + 40

        def _cam(seed: int) -> np.ndarray:
            noise = self._fake_rng.integers(0, 8, size=(h, w, 3), dtype=np.uint8)
            layer = np.full((h, w, 3), base_u8 + seed * 5, dtype=np.uint8)
            return np.clip(layer.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(
                np.uint8
            )

        pose = np.zeros(14, dtype=np.float32)
        pose[:3] = (np.sin(phase + np.arange(3) * 0.4) * 0.05).astype(np.float32)
        pose[3:6] = (np.cos(phase + np.arange(3) * 0.3) * 0.05).astype(np.float32)
        pose[6] = 0.5
        pose[7:10] = (np.sin(phase * 1.1 + np.arange(3) * 0.2) * 0.05).astype(np.float32)
        pose[10:13] = (np.cos(phase * 1.1 + np.arange(3) * 0.25) * 0.05).astype(np.float32)
        pose[13] = 0.5

        obs: Dict[str, Any] = {
            "image/head": _cam(0),
            "image/left_wrist": _cam(1),
            "image/right_wrist": _cam(2),
            "state/joint": np.zeros(16, dtype=np.float32),
            "state/pose": pose,
            "state/head": np.array([0.01 * np.sin(phase), 0.01 * np.cos(phase)], dtype=np.float32),
            "state/waist": np.array([0.005 * np.sin(phase * 0.9), 0.005 * np.cos(phase * 0.9)], dtype=np.float32),
        }
        self._last_obs = obs
        return obs

    def _controller_error_obs(self) -> Dict[str, Any]:
        obs = self._controller_cached_obs()
        if obs:
            return obs
        try:
            return self._get_obs()
        except Exception as obs_exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to refresh fake observation after controller error: %s",
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
                done=False,
                truncated=True,
                info={
                    "success": False,
                    "infra_abort": True,
                    "controller_error": error_message,
                    "controller_action_executed": False,
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
        del obs, action
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

    def get_latest_obs(self) -> Dict[str, Any]:
        obs = self._get_obs()
        if self._controller_enabled and self._controller is not None:
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

    def prepare_episode_reset(self) -> None:
        if self._episode_reset_prepared:
            return
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
        self._episode_reset_prepared = True

    def start_episode_after_reset(self) -> Dict[str, Any]:
        if not self._episode_reset_prepared:
            self.prepare_episode_reset()
        self._take_action_cnt = 0
        self.episode_count += 1
        obs = self._get_obs()
        if self._controller_enabled and self._controller is not None:
            self._controller.start_episode()
            self._controller.set_latest_obs(obs)
            self._controller.transition_after_reset()
            # Without this, `request_ready` only works after `_episode_active` is set by
            # `start_episode()` above — keypresses before the first `reset()` are ignored,
            # and operators often press `g` while the actor is still loading JoyRA/weights.
            # Fake env skips the real-robot human "arm" gate: enter RUNNING immediately.
            self._controller.request_ready()
        self._episode_reset_prepared = False
        return obs

    def reset(
        self,
    ) -> Dict[str, Any]:
        self.prepare_episode_reset()
        return self.start_episode_after_reset()

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


__all__ = ["AgiBotFakeTaskEnv"]
