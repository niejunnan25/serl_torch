"""Actor-side agentlace bridge helpers for training runtimes."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.training.loop_utils import _count_env_step_update_triggers
from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.training.async_runtime.agentlace import _AgentlaceAsyncLearner
from serl_launcher.training.async_runtime.agentlace import _MixedBatchPrefetcher
from serl_launcher.residual.runtime_agent import ResidualAgentRuntime
from serl_launcher.utils.agentlace_io import resolve_agentlace_bootstrap_path
from serl_launcher.utils.agentlace_io import save_agentlace_bootstrap


@dataclass(frozen=True)
class AgentlaceBridgeConfig:
    external_actor_mode: bool
    host: str
    trainer_port: int
    broadcast_port: int
    data_store_queue_size: int
    spawn_local_worker: bool
    connect_timeout_sec: float
    batch_size: int
    offline_ratio: float
    symmetric_replay: bool
    training_starts: int
    update_every: int
    updates_per_step: int
    update_frequency: int
    idle_sleep_sec: float
    learner_device: Optional[str]
    stats_period_steps: int
    bounded_lag_enabled: bool
    bounded_lag_mode: str
    bounded_lag_max_update_calls: int
    bounded_lag_poll_sec: Optional[float]
    bounded_lag_timeout_sec: Optional[float]
    bounded_lag_sync_on_wait: bool
    bounded_lag_log_period_steps: int
    bounded_lag_env_steps_per_update_call: Optional[int]
    bounded_lag_manual_rate_enabled: bool


@dataclass
class AgentlaceBridgeState:
    timer_last_sent_step: int = -1
    target_update_calls: int = 0
    tracked_env_steps: int = 0
    wait_count: int = 0
    timeout_count: int = 0
    wait_total_sec: float = 0.0
    last_wait_sec: float = 0.0
    last_required_update_steps: int = 0
    last_lag_before_wait: int = 0
    last_lag_after_wait: int = 0


def _replay_capacity(buffer: Any) -> Optional[int]:
    if hasattr(buffer, "capacity"):
        return int(getattr(buffer, "capacity"))
    if hasattr(buffer, "_capacity"):
        return int(getattr(buffer, "_capacity"))
    return None


def create_agentlace_async_learner(
    *,
    config: AgentlaceBridgeConfig,
    algorithm: ResidualAgentRuntime,
    actor_agent: Any,
    replay_buffer: Any,
    offline_buffer: Optional[Any],
    cfg_dict: Dict[str, Any],
    sample_obs: Dict[str, Any],
    action_dim: int,
    critic_action_dim: int,
    image_keys: tuple[str, ...],
    action_transform: Any,
) -> _AgentlaceAsyncLearner:
    async_learner = _AgentlaceAsyncLearner(
        algorithm=algorithm,
        actor_agent=actor_agent,
        replay_buffer=replay_buffer,
        offline_buffer=offline_buffer,
        batch_size=int(config.batch_size),
        offline_ratio=float(config.offline_ratio),
        symmetric_replay=bool(config.symmetric_replay),
        training_starts=int(config.training_starts),
        update_frequency=int(config.update_frequency),
        idle_sleep_sec=float(config.idle_sleep_sec),
        cfg_dict=cfg_dict,
        sample_obs=sample_obs,
        action_dim=int(action_dim),
        critic_action_dim=int(critic_action_dim),
        image_keys=tuple(image_keys),
        action_transform=action_transform,
        learner_device=config.learner_device,
        host=str(config.host),
        port_number=int(config.trainer_port),
        broadcast_port=int(config.broadcast_port),
        data_store_queue_size=int(config.data_store_queue_size),
        replay_capacity=_replay_capacity(replay_buffer),
        spawn_local_worker=bool(config.spawn_local_worker),
        connect_timeout_sec=float(config.connect_timeout_sec),
    )
    async_learner.start()
    return async_learner


def save_actor_bootstrap(
    *,
    run_dir: Path,
    bootstrap_file: str,
    sample_obs: Dict[str, Any],
    state_core_dim: int,
    env_action_dim: int,
    step_action_dim: int,
    agent_action_dim: int,
    critic_action_dim: int,
    image_keys: tuple[str, ...],
    action_transform: Any,
    chunk_step_enabled: bool,
    chunk_horizon: int,
    learner_agent: Any,
    logger: logging.Logger,
) -> Path:
    bootstrap_path = resolve_agentlace_bootstrap_path(
        run_dir=run_dir,
        bootstrap_file=bootstrap_file,
    )
    save_agentlace_bootstrap(
        bootstrap_path,
        {
            "sample_obs": sample_obs,
            "state_core_dim": int(state_core_dim),
            "env_action_dim": int(env_action_dim),
            "step_action_dim": int(step_action_dim),
            "agent_action_dim": int(agent_action_dim),
            "critic_action_dim": int(critic_action_dim),
            "image_keys": tuple(image_keys),
            "action_transform": action_transform,
            "chunk_step_enabled": bool(chunk_step_enabled),
            "chunk_horizon": int(chunk_horizon),
            "initial_agent_payload": snapshot_agent_checkpoint_payload(
                learner_agent,
                step=int(learner_agent.state.step),
            ),
            "saved_at_unix": float(time.time()),
        },
    )
    logger.info("Agentlace bootstrap saved to %s", bootstrap_path)
    return bootstrap_path


def build_agentlace_timer_payload(
    *,
    config: AgentlaceBridgeConfig,
    state: AgentlaceBridgeState,
    profiler: Optional[_RuntimeProfiler],
    replay_buffer: Any,
    offline_buffer: Optional[Any],
    async_learner: Optional[Any],
    sync_replay_prefetcher: Optional[_MixedBatchPrefetcher],
    train_env_step: int,
    decision_step: int,
    train_episode_id: int,
) -> Optional[Dict[str, Any]]:
    if profiler is None or (not profiler.enabled) or (not profiler.has_data()):
        return None
    snapshot = profiler.snapshot()
    payload: Dict[str, Any] = {
        "train_env_step": int(train_env_step),
        "decision_step": int(decision_step),
        "train_episode_id": int(train_episode_id),
        "online_buffer_size": int(len(replay_buffer))
        if replay_buffer is not None
        else 0,
    }
    if offline_buffer is not None:
        payload["offline_buffer_size"] = int(len(offline_buffer))
    if async_learner is not None:
        payload["learner_update_steps"] = int(async_learner.get_update_steps())
        payload["replay_prefetch_queue_size"] = int(
            async_learner.get_prefetch_queue_size()
        )
        if config.bounded_lag_enabled:
            payload["bounded_lag_mode"] = str(config.bounded_lag_mode)
            payload["bounded_lag_target_update_calls"] = int(state.target_update_calls)
            payload["bounded_lag_tracked_env_steps"] = int(state.tracked_env_steps)
            if config.bounded_lag_env_steps_per_update_call is not None:
                payload["bounded_lag_env_steps_per_update_call"] = float(
                    config.bounded_lag_env_steps_per_update_call
                )
            payload["bounded_lag_required_update_steps"] = int(
                state.last_required_update_steps
            )
            payload["bounded_lag_lag_before_wait"] = int(state.last_lag_before_wait)
            payload["bounded_lag_lag_after_wait"] = int(state.last_lag_after_wait)
            payload["bounded_lag_wait_last_sec"] = float(state.last_wait_sec)
    elif sync_replay_prefetcher is not None:
        payload["replay_prefetch_queue_size"] = int(
            sync_replay_prefetcher.get_queue_size()
        )

    def _safe_name(name: str) -> str:
        return str(name).replace(".", "_").replace("/", "_").replace(" ", "_")

    for name, stats in snapshot.get("durations", {}).items():
        mean_ms = stats.get("mean_ms", None)
        if mean_ms is not None:
            payload[f"{_safe_name(name)}_mean_ms"] = float(mean_ms)
    for name, stats in snapshot.get("values", {}).items():
        mean_value = stats.get("mean", None)
        if mean_value is not None:
            payload[f"{_safe_name(name)}_mean"] = float(mean_value)
    return payload


def maybe_send_agentlace_timer_stats(
    *,
    config: AgentlaceBridgeConfig,
    state: AgentlaceBridgeState,
    profiler: Optional[_RuntimeProfiler],
    replay_buffer: Any,
    offline_buffer: Optional[Any],
    async_learner: Optional[Any],
    sync_replay_prefetcher: Optional[_MixedBatchPrefetcher],
    train_env_step: int,
    decision_step: int,
    train_episode_id: int,
    force: bool = False,
) -> None:
    if (not config.external_actor_mode) or async_learner is None:
        return
    current_step = int(train_env_step)
    period = max(1, int(config.stats_period_steps))
    if (not force) and state.timer_last_sent_step >= 0:
        if (current_step - state.timer_last_sent_step) < period:
            return
    payload = build_agentlace_timer_payload(
        config=config,
        state=state,
        profiler=profiler,
        replay_buffer=replay_buffer,
        offline_buffer=offline_buffer,
        async_learner=async_learner,
        sync_replay_prefetcher=sync_replay_prefetcher,
        train_env_step=int(train_env_step),
        decision_step=int(decision_step),
        train_episode_id=int(train_episode_id),
    )
    if payload is None:
        return
    async_learner.request_stats({"timer": payload})
    state.timer_last_sent_step = current_step


def advance_async_target_update_calls(
    *,
    config: AgentlaceBridgeConfig,
    state: AgentlaceBridgeState,
    async_learner: Optional[Any],
    phase_train_flag: bool,
    train_step_before: int,
    train_step_after: int,
    replay_size_before: int,
    replay_size_after: int,
) -> int:
    if (
        (not config.bounded_lag_enabled)
        or async_learner is None
        or (not bool(phase_train_flag))
    ):
        return 0
    if config.bounded_lag_manual_rate_enabled:
        added_trainable_env_steps = _count_env_step_update_triggers(
            train_step_before=int(train_step_before),
            train_step_after=int(train_step_after),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
            training_starts=int(config.training_starts),
            update_every=1,
        )
        if added_trainable_env_steps <= 0:
            return 0
        state.tracked_env_steps += int(added_trainable_env_steps)
        previous_target_update_calls = int(state.target_update_calls)
        state.target_update_calls = int(
            state.tracked_env_steps
            // float(config.bounded_lag_env_steps_per_update_call)
        )
        if state.target_update_calls < previous_target_update_calls:
            state.target_update_calls = int(previous_target_update_calls)
        return int(state.target_update_calls - previous_target_update_calls)
    trigger_count = _count_env_step_update_triggers(
        train_step_before=int(train_step_before),
        train_step_after=int(train_step_after),
        replay_size_before=int(replay_size_before),
        replay_size_after=int(replay_size_after),
        training_starts=int(config.training_starts),
        update_every=int(config.update_every),
    )
    added_update_calls = int(trigger_count * int(config.updates_per_step))
    if added_update_calls > 0:
        state.target_update_calls += int(added_update_calls)
    return int(added_update_calls)


def maybe_wait_for_async_learner_budget(
    *,
    config: AgentlaceBridgeConfig,
    state: AgentlaceBridgeState,
    async_learner: Optional[Any],
    logger: logging.Logger,
    train_env_step: int,
    decision_step: int,
) -> None:
    if (not config.bounded_lag_enabled) or async_learner is None:
        return
    required_update_steps = max(
        0,
        int(state.target_update_calls) - int(config.bounded_lag_max_update_calls),
    )
    state.last_required_update_steps = int(required_update_steps)
    current_update_steps = int(async_learner.get_update_steps())
    lag_before_wait = max(0, required_update_steps - current_update_steps)
    state.last_lag_before_wait = int(lag_before_wait)
    state.last_lag_after_wait = int(lag_before_wait)
    state.last_wait_sec = 0.0
    if lag_before_wait <= 0:
        return

    wait_start = time.perf_counter()
    updated_steps = int(
        async_learner.wait_for_update_steps(
            required_update_steps,
            poll_interval_sec=config.bounded_lag_poll_sec,
            timeout_sec=config.bounded_lag_timeout_sec,
        )
    )
    if config.bounded_lag_sync_on_wait:
        async_learner.sync_now(timeout_sec=config.bounded_lag_timeout_sec)
        updated_steps = int(async_learner.get_update_steps())
    wait_sec = float(time.perf_counter() - wait_start)
    lag_after_wait = max(0, required_update_steps - updated_steps)

    state.wait_count += 1
    state.wait_total_sec += wait_sec
    state.last_wait_sec = float(wait_sec)
    state.last_lag_after_wait = int(lag_after_wait)

    if lag_after_wait > 0:
        state.timeout_count += 1
        logger.warning(
            "Bounded async lag timeout: train_env_step=%s decision_step=%s "
            "required_update_steps=%s learner_update_steps=%s remaining_lag=%s "
            "wait_sec=%.3f",
            int(train_env_step),
            int(decision_step),
            int(required_update_steps),
            int(updated_steps),
            int(lag_after_wait),
            float(wait_sec),
        )
        return

    log_period = max(0, int(config.bounded_lag_log_period_steps))
    if log_period > 0 and int(train_env_step) % log_period == 0:
        logger.info(
            "Bounded async lag wait complete: train_env_step=%s decision_step=%s "
            "target_update_calls=%s required_update_steps=%s learner_update_steps=%s "
            "wait_sec=%.3f",
            int(train_env_step),
            int(decision_step),
            int(state.target_update_calls),
            int(required_update_steps),
            int(updated_steps),
            float(wait_sec),
        )


def sync_async_bounded_lag_baseline_from_learner(
    *,
    config: AgentlaceBridgeConfig,
    state: AgentlaceBridgeState,
    async_learner: Optional[Any],
    logger: logging.Logger,
) -> None:
    if (not config.bounded_lag_enabled) or async_learner is None:
        return
    learner_update_steps = int(async_learner.get_update_steps())
    previous_target_update_calls = int(state.target_update_calls)
    if learner_update_steps > int(state.target_update_calls):
        state.target_update_calls = int(learner_update_steps)
    if config.bounded_lag_manual_rate_enabled:
        min_tracked_env_steps = int(state.target_update_calls) * int(
            config.bounded_lag_env_steps_per_update_call
        )
        if min_tracked_env_steps > int(state.tracked_env_steps):
            state.tracked_env_steps = int(min_tracked_env_steps)
    if int(state.target_update_calls) != int(previous_target_update_calls):
        logger.info(
            "Aligned bounded-lag baseline to learner update steps: "
            "target_update_calls %s -> %s learner_update_steps=%s mode=%s",
            int(previous_target_update_calls),
            int(state.target_update_calls),
            int(learner_update_steps),
            str(config.bounded_lag_mode),
        )
