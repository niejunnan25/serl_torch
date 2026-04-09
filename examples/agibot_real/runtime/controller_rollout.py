"""Reusable controller-driven rollout helpers for AgiBot scripts."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping as MappingABC
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np

from ..env_wrappers.controller import STATE_PAUSED
from ..env_wrappers.controller import STATE_RUNNING
from ..env_wrappers.controller import STATE_WAIT_READY
from ..env_wrappers.controller import TERMINAL_FAIL
from ..env_wrappers.controller import TERMINAL_RESET
from ..env_wrappers.controller import TERMINAL_SUCCESS
from ..env_wrappers.controller import TERMINAL_TIMEOUT


@dataclass
class ControllerPlannedStep:
    sequence_id: int
    obs_before: Optional[Dict[str, Any]]
    final_action: np.ndarray
    chunk_step: int
    executed_horizon: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerExecutedStep:
    planned: ControllerPlannedStep
    obs: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass
class ControllerEpisodeSummary:
    episode_steps: int = 0
    episode_return: float = 0.0
    success: bool = False
    terminal_signal: Optional[str] = None
    terminal_info: Dict[str, Any] = field(default_factory=dict)


def controller_poll_sec(cfg: Mapping[str, Any]) -> float:
    controller_cfg = cfg.get("controller", {})
    if not isinstance(controller_cfg, Mapping):
        controller_cfg = {}
    return max(0.01, float(controller_cfg.get("poll_interval_sec", 0.05)))


def controller_terminal_grace_sec(cfg: Mapping[str, Any]) -> float:
    controller_cfg = cfg.get("controller", {})
    if not isinstance(controller_cfg, Mapping):
        controller_cfg = {}
    return max(0.0, float(controller_cfg.get("terminal_grace_sec", 0.15)))


def controller_terminal_signals() -> set[str]:
    return {
        TERMINAL_SUCCESS,
        TERMINAL_FAIL,
        TERMINAL_RESET,
        TERMINAL_TIMEOUT,
    }


def require_controller_rollout_capability(
    *,
    env: Any,
    chunk_step_enabled: bool,
    script_name: str,
) -> None:
    if not bool(getattr(env, "controller_enabled", False)):
        raise ValueError(
            f"{script_name} controller rollout requires controller.enabled=true on the env"
        )
    if not bool(chunk_step_enabled):
        raise ValueError(
            f"{script_name} controller rollout requires chunk_step.enabled=true"
        )


def _override_executed_step_from_meta(
    executed: ControllerExecutedStep,
    meta: Mapping[str, Any],
) -> ControllerExecutedStep:
    info = dict(executed.info)
    terminal_signal = meta.get("terminal_signal", None)
    terminal_info = meta.get("terminal_info", {})
    if isinstance(terminal_info, MappingABC):
        info.update(dict(terminal_info))

    reward = float(executed.reward)
    done = bool(executed.done)
    truncated = bool(executed.truncated)
    if terminal_signal == TERMINAL_SUCCESS:
        reward = 1.0
        done = True
        truncated = False
        info["success"] = True
        info["human_success"] = True
    elif terminal_signal == TERMINAL_FAIL:
        reward = 0.0
        done = True
        truncated = False
        info["success"] = False
        info["human_fail"] = True
    elif terminal_signal == TERMINAL_RESET:
        reward = 0.0
        done = False
        truncated = True
        info["success"] = False
        info["human_reset"] = True
    elif terminal_signal == TERMINAL_TIMEOUT:
        reward = 0.0
        done = False
        truncated = True
        info["success"] = False
        info["time_limit_reached"] = True
    return ControllerExecutedStep(
        planned=executed.planned,
        obs=executed.obs,
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        info=info,
    )


def run_controller_episode(
    *,
    env: Any,
    initial_obs: Dict[str, Any],
    max_episode_steps: int,
    chunk_horizon: int,
    cfg: Mapping[str, Any],
    logger: logging.Logger,
    plan_chunk_fn: Callable[[Dict[str, Any], int], Sequence[ControllerPlannedStep]],
    on_step_fn: Callable[[ControllerExecutedStep, int], None],
) -> ControllerEpisodeSummary:
    poll_sec = controller_poll_sec(cfg)
    terminal_grace_sec = controller_terminal_grace_sec(cfg)
    terminal_signals = controller_terminal_signals()

    summary = ControllerEpisodeSummary()
    planned_steps: Deque[ControllerPlannedStep] = deque()
    buffered_step: Optional[ControllerExecutedStep] = None
    queue_empty_since: Optional[float] = None
    terminal_wait_since: Optional[float] = None
    current_obs_raw = initial_obs

    def _commit_step(executed: ControllerExecutedStep) -> None:
        summary.episode_steps = int(summary.episode_steps + 1)
        summary.episode_return += float(executed.reward)
        summary.success = bool(executed.info.get("success", summary.success))
        if executed.done or executed.truncated:
            summary.terminal_info = dict(executed.info)
        on_step_fn(executed, int(summary.episode_steps))

    while summary.episode_steps < int(max_episode_steps):
        meta = env.get_controller_meta()
        controller_state = str(meta.get("state", None))
        terminal_signal = meta.get("terminal_signal", None)

        if (
            terminal_signal not in terminal_signals
            and buffered_step is not None
            and (not planned_steps)
        ):
            if queue_empty_since is None:
                queue_empty_since = time.time()
            if (time.time() - float(queue_empty_since)) >= float(terminal_grace_sec):
                _commit_step(buffered_step)
                if bool(buffered_step.done or buffered_step.truncated):
                    summary.terminal_signal = terminal_signal
                    break
                buffered_step = None
        else:
            queue_empty_since = None

        polled = env.poll_controller_transitions(max_items=max(1, int(chunk_horizon)))
        if polled:
            queue_empty_since = None
            terminal_wait_since = None
            for payload in polled:
                if not planned_steps:
                    logger.warning(
                        "Controller transition received without a planned step: seq=%s",
                        payload.get("sequence_id", None),
                    )
                    continue
                planned = planned_steps.popleft()
                observed_seq = int(payload["sequence_id"])
                if int(planned.sequence_id) != observed_seq:
                    raise RuntimeError(
                        "Controller transition sequence mismatch: "
                        f"planned={planned.sequence_id} observed={observed_seq}"
                    )
                if buffered_step is not None:
                    _commit_step(buffered_step)
                    if bool(buffered_step.done or buffered_step.truncated):
                        buffered_step = None
                        break
                    buffered_step = None

                current_obs_raw = payload["obs"]
                buffered_step = ControllerExecutedStep(
                    planned=planned,
                    obs=payload["obs"],
                    reward=float(payload["reward"]),
                    done=bool(payload["done"]),
                    truncated=bool(payload["truncated"]),
                    info=dict(payload["info"]),
                )
                if planned_steps:
                    next_planned = planned_steps[0]
                    next_planned.obs_before = current_obs_raw
                if bool(buffered_step.done or buffered_step.truncated):
                    _commit_step(buffered_step)
                    summary.terminal_signal = terminal_signal
                    buffered_step = None
                    planned_steps.clear()
                    break
            if buffered_step is None and summary.terminal_info:
                break
            continue

        if terminal_signal in terminal_signals and controller_state != STATE_RUNNING:
            if terminal_wait_since is None:
                terminal_wait_since = time.time()
            if (time.time() - float(terminal_wait_since)) >= float(terminal_grace_sec):
                summary.terminal_signal = str(terminal_signal)
                if isinstance(meta.get("terminal_info", {}), MappingABC):
                    summary.terminal_info = dict(meta.get("terminal_info", {}))
                if buffered_step is not None:
                    patched = _override_executed_step_from_meta(buffered_step, meta)
                    _commit_step(patched)
                    buffered_step = None
                else:
                    if planned_steps:
                        logger.warning(
                            "Controller terminal=%s dropped %s unexecuted planned steps",
                            terminal_signal,
                            len(planned_steps),
                        )
                    summary.success = bool(terminal_signal == TERMINAL_SUCCESS)
                planned_steps.clear()
                break
        else:
            terminal_wait_since = None

        if controller_state == STATE_RUNNING and (not planned_steps):
            remaining_steps = int(max_episode_steps) - int(summary.episode_steps)
            if remaining_steps <= 0:
                break
            current_obs_raw = env.get_latest_obs()
            planned = plan_chunk_fn(current_obs_raw, int(remaining_steps))
            planned_steps = deque(planned)
            queue_empty_since = None
            if planned_steps:
                continue

        if controller_state in {STATE_WAIT_READY, STATE_PAUSED}:
            current_obs_raw = env.get_latest_obs()
        time.sleep(poll_sec)

    if buffered_step is not None and summary.episode_steps < int(max_episode_steps):
        _commit_step(buffered_step)
        if summary.terminal_signal is None and bool(buffered_step.done or buffered_step.truncated):
            summary.terminal_signal = str(
                buffered_step.info.get("terminal_signal", summary.terminal_signal)
            )
    return summary
