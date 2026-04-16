"""Controller-driven runtime helpers for AgiBot real-robot episodes."""
from __future__ import annotations

import logging
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Deque
from typing import Dict
from typing import Mapping
from typing import Optional

import numpy as np


STATE_WAIT_READY = "WAIT_READY"
STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"
STATE_RESETTING = "RESETTING"
STATE_EPISODE_DONE = "EPISODE_DONE"

TERMINAL_SUCCESS = "success"
TERMINAL_FAIL = "fail"
TERMINAL_RESET = "reset"
TERMINAL_TIMEOUT = "timeout"
TERMINAL_HOOK = "hook"


def _clone_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_obs_tree(v) for k, v in value.items()}
    return np.array(value, copy=True)


@dataclass
class QueuedAction:
    sequence_id: int
    action: np.ndarray


@dataclass
class ExecutedTransition:
    sequence_id: int
    obs: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ManualEpisodeController:
    """Thread-safe operator/controller state for real-robot episodes."""

    def __init__(
        self,
        *,
        enabled: bool,
        interface: str = "terminal",
        poll_interval_sec: float = 0.05,
        terminal_grace_sec: float = 0.15,
        keys: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.interface = str(interface).strip().lower()
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self.terminal_grace_sec = max(0.0, float(terminal_grace_sec))
        self.logger = logger or logging.getLogger(__name__)
        key_cfg = dict(keys or {})
        self.keys = {
            "ready": str(key_cfg.get("ready", "g")),
            "pause": str(key_cfg.get("pause", "p")),
            "reset": str(key_cfg.get("reset", "r")),
            "success": str(key_cfg.get("success", "s")),
            "fail": str(key_cfg.get("fail", "f")),
            "help": str(key_cfg.get("help", "h")),
        }

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._operator_thread: Optional[threading.Thread] = None
        self._operator_started = False
        self._queue: Deque[QueuedAction] = deque()
        self._transitions: Deque[ExecutedTransition] = deque()
        self._next_sequence_id = 0
        self._inflight_action: Optional[QueuedAction] = None
        self._state = STATE_WAIT_READY
        self._episode_active = False
        self._terminal_signal: Optional[str] = None
        self._terminal_info: Dict[str, Any] = {}
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._latest_obs_timestamp_ns: Optional[int] = None
        self._terminal_grace_deadline_monotonic = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            return str(self._state)

    def start_episode(self) -> None:
        with self._lock:
            self._queue.clear()
            self._transitions.clear()
            self._inflight_action = None
            self._terminal_signal = None
            self._terminal_info = {}
            self._episode_active = True
            self._state = STATE_WAIT_READY

    def set_latest_obs(self, obs: Dict[str, Any]) -> None:
        with self._lock:
            self._latest_obs = _clone_obs_tree(obs)
            self._latest_obs_timestamp_ns = int(time.time_ns())

    def get_latest_obs(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest_obs is None:
                return None
            return _clone_obs_tree(self._latest_obs)

    def get_meta(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "interface": str(self.interface),
                "terminal_grace_sec": float(self.terminal_grace_sec),
                "terminal_grace_active": bool(self._terminal_grace_active_locked()),
                "state": str(self._state),
                "episode_active": bool(self._episode_active),
                "terminal_signal": self._terminal_signal,
                "terminal_info": dict(self._terminal_info),
                "queue_depth": int(len(self._queue)),
                "transition_depth": int(len(self._transitions)),
                "inflight_sequence_id": (
                    int(self._inflight_action.sequence_id)
                    if self._inflight_action is not None
                    else None
                ),
                "latest_obs_ready": self._latest_obs is not None,
                "latest_obs_timestamp_ns": self._latest_obs_timestamp_ns,
            }

    def _terminal_grace_active_locked(self) -> bool:
        return time.monotonic() < self._terminal_grace_deadline_monotonic

    def _terminal_grace_remaining_locked(self) -> float:
        return max(0.0, self._terminal_grace_deadline_monotonic - time.monotonic())

    def _arm_terminal_grace_locked(self) -> None:
        if self.terminal_grace_sec <= 0.0:
            self._terminal_grace_deadline_monotonic = 0.0
            return
        self._terminal_grace_deadline_monotonic = max(
            self._terminal_grace_deadline_monotonic,
            time.monotonic() + self.terminal_grace_sec,
        )

    def enqueue_action_chunk(self, actions: np.ndarray) -> list[int]:
        action_chunk = np.asarray(actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected 2-D action chunk, got {action_chunk.shape}")
        sequence_ids: list[int] = []
        with self._lock:
            if (not self._episode_active) or self._terminal_signal is not None:
                return sequence_ids
            for action in action_chunk:
                seq = int(self._next_sequence_id)
                self._next_sequence_id += 1
                self._queue.append(
                    QueuedAction(
                        sequence_id=seq,
                        action=np.asarray(action, dtype=np.float32).copy(),
                    )
                )
                sequence_ids.append(seq)
        return sequence_ids

    def pop_next_action(self) -> Optional[QueuedAction]:
        with self._lock:
            if (not self._episode_active) or self._terminal_signal is not None:
                return None
            if self._state != STATE_RUNNING or (not self._queue):
                return None
            queued = self._queue.popleft()
            self._inflight_action = queued
            return queued

    def push_transition(self, transition: ExecutedTransition) -> None:
        with self._lock:
            self._transitions.append(transition)
            self._inflight_action = None
            self._latest_obs = _clone_obs_tree(transition.obs)
            self._latest_obs_timestamp_ns = int(time.time_ns())
            if bool(transition.done) or bool(transition.truncated):
                if self._terminal_signal is None:
                    if bool(transition.info.get("success", False)):
                        self._terminal_signal = TERMINAL_SUCCESS
                    elif bool(transition.truncated):
                        self._terminal_signal = TERMINAL_TIMEOUT
                    else:
                        self._terminal_signal = TERMINAL_HOOK
                    self._terminal_info = dict(transition.info)
                self._arm_terminal_grace_locked()
                self._queue.clear()
                self._episode_active = False
                self._state = STATE_EPISODE_DONE

    def poll_transitions(self, *, max_items: int = 64) -> list[ExecutedTransition]:
        polled: list[ExecutedTransition] = []
        with self._lock:
            target = max(1, int(max_items))
            while self._transitions and len(polled) < target:
                polled.append(self._transitions.popleft())
        return polled

    def request_ready(self) -> Dict[str, Any]:
        with self._lock:
            if self._episode_active and self._state in {
                STATE_WAIT_READY,
                STATE_PAUSED,
            }:
                self._state = STATE_RUNNING
        return self.get_meta()

    def request_pause(self) -> Dict[str, Any]:
        with self._lock:
            if self._episode_active and self._state == STATE_RUNNING:
                self._state = STATE_PAUSED
        return self.get_meta()

    def request_reset(self) -> Dict[str, Any]:
        with self._lock:
            self._queue.clear()
            if self._episode_active:
                self._terminal_signal = TERMINAL_RESET
                self._terminal_info = {"human_reset": True}
                self._arm_terminal_grace_locked()
                self._episode_active = False
                self._state = STATE_RESETTING
            else:
                self._state = STATE_WAIT_READY
        return self.get_meta()

    def mark_success(self) -> Dict[str, Any]:
        with self._lock:
            if self._episode_active:
                self._queue.clear()
                self._terminal_signal = TERMINAL_SUCCESS
                self._terminal_info = {"human_success": True}
                self._arm_terminal_grace_locked()
                self._episode_active = False
                self._state = STATE_EPISODE_DONE
        return self.get_meta()

    def mark_fail(self) -> Dict[str, Any]:
        with self._lock:
            if self._episode_active:
                self._queue.clear()
                self._terminal_signal = TERMINAL_FAIL
                self._terminal_info = {"human_fail": True}
                self._arm_terminal_grace_locked()
                self._episode_active = False
                self._state = STATE_EPISODE_DONE
        return self.get_meta()

    def mark_timeout(
        self, *, info: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        with self._lock:
            self._queue.clear()
            self._terminal_signal = TERMINAL_TIMEOUT
            self._terminal_info = dict(info or {})
            self._arm_terminal_grace_locked()
            self._episode_active = False
            self._state = STATE_EPISODE_DONE
        return self.get_meta()

    def transition_after_reset(self) -> Dict[str, Any]:
        with self._lock:
            if self._state == STATE_RESETTING:
                self._state = STATE_WAIT_READY
        return self.get_meta()

    def start_operator_interface(self) -> None:
        if (not self.enabled) or self.interface != "terminal" or self._operator_started:
            return
        if not sys.stdin.isatty():
            self.logger.warning(
                "Controller terminal interface requested, but stdin is not a TTY; "
                "operator key control is disabled."
            )
            return
        self._operator_thread = threading.Thread(
            target=self._terminal_loop,
            name="agibot-controller-keys",
            daemon=True,
        )
        self._operator_thread.start()
        self._operator_started = True
        self._log_help()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._operator_thread is not None and self._operator_thread.is_alive():
            self._operator_thread.join(timeout=1.0)

    def _log_help(self) -> None:
        self.logger.info(
            "Controller keys: ready/resume=%s pause=%s reset=%s success=%s fail=%s help=%s",
            self.keys["ready"],
            self.keys["pause"],
            self.keys["reset"],
            self.keys["success"],
            self.keys["fail"],
            self.keys["help"],
        )

    def _dispatch_key(self, ch: str) -> None:
        key = str(ch).strip().lower()
        if not key:
            return
        terminal_keys = {
            self.keys["reset"],
            self.keys["success"],
            self.keys["fail"],
        }
        with self._lock:
            terminal_grace_remaining = (
                self._terminal_grace_remaining_locked()
                if key in terminal_keys
                else 0.0
            )
        if terminal_grace_remaining > 0.0:
            self.logger.debug(
                "Controller terminal key ignored during grace window: key=%s remaining=%.3fs",
                key,
                float(terminal_grace_remaining),
            )
            return
        if key == self.keys["ready"]:
            meta = self.request_ready()
            self.logger.info("Controller event: ready/resume -> %s", meta["state"])
        elif key == self.keys["pause"]:
            meta = self.request_pause()
            self.logger.info("Controller event: pause -> %s", meta["state"])
        elif key == self.keys["reset"]:
            meta = self.request_reset()
            self.logger.info("Controller event: reset -> %s", meta["state"])
        elif key == self.keys["success"]:
            meta = self.mark_success()
            self.logger.info("Controller event: success -> %s", meta["state"])
        elif key == self.keys["fail"]:
            meta = self.mark_fail()
            self.logger.info("Controller event: fail -> %s", meta["state"])
        elif key == self.keys["help"]:
            self._log_help()

    def _terminal_loop(self) -> None:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                readable, _, _ = select.select(
                    [sys.stdin], [], [], self.poll_interval_sec
                )
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                self._dispatch_key(ch)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Controller terminal loop exited: %s", exc)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:  # noqa: BLE001
                pass
