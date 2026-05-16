from __future__ import annotations

"""Canonical LIBERO residual checkpoint evaluation entrypoint."""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import is_dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from serl_launcher.async_eval import resolve_async_eval_checkpoint_from_index
from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.policy.typed_factory import build_policy_client
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_id
from serl_launcher.policy.typed_factory import resolve_policy_backend_type
from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import build_chunk_residual_sample_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.checkpoint_utils import load_checkpoint_payload
from serl_launcher.utils.checkpoint_utils import resolve_checkpoint_path
from serl_launcher.utils.jsonl import JsonlWriter
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

from serl_torch.examples.libero.config import cfg_to_log_payload
from serl_torch.examples.libero.config import LiberoEvalConfig
from serl_torch.examples.libero.config import parse_eval_cfg
from serl_torch.examples.libero.env.factory import create_env
from serl_torch.examples.libero.env.observation import build_libero_state
from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.observation import LIBERO_STATE_DIM
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_HEIGHT
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_WIDTH
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input


@dataclass
class _EvalLoopStats:
    episode_returns: list[float]
    episode_steps_list: list[int]
    successes: int = 0
    total_env_steps: int = 0
    completed_episodes: int = 0
    policy_requests: int = 0
    policy_batch_requests: int = 0
    policy_samples: int = 0
    active_lane_counts: list[int] | None = None


@dataclass
class _EvalLane:
    lane_id: int
    env: Any
    active: bool = False
    episode_id: int = 0
    init_episode_idx: int = 0
    obs: dict[str, Any] | None = None
    episode_return: float = 0.0
    episode_steps: int = 0
    episode_success: bool = False
    env_episode_done: bool = False
    manual_cap_reached: bool = False
    last_info: dict[str, Any] | None = None


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0, got {resolved}")
    return resolved


def _positive_int(value: Any, field_name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _checkpoint_step_from_path(checkpoint_path: Path) -> int | None:
    stem = checkpoint_path.stem
    try:
        return int(stem.split("_")[-1])
    except Exception:  # noqa: BLE001
        return None


def _resolve_checkpoint_input(
    checkpoint_path_raw: Any,
    checkpoint_step_raw: Any,
    *,
    original_cwd: Path | None = None,
) -> tuple[Path | None, Path | None]:
    if checkpoint_path_raw is None:
        return None, None
    checkpoint_value = str(checkpoint_path_raw).strip()
    if not checkpoint_value or checkpoint_value.lower() == "null":
        return None, None

    checkpoint_input_path = Path(checkpoint_value).expanduser()
    if not checkpoint_input_path.is_absolute():
        base_dir = (
            Path.cwd().resolve() if original_cwd is None else original_cwd.resolve()
        )
        checkpoint_input_path = base_dir / checkpoint_input_path
    checkpoint_input_path = checkpoint_input_path.resolve()

    checkpoint_step = (
        None
        if checkpoint_step_raw is None
        else _positive_int(checkpoint_step_raw, "eval.checkpoint_step")
    )
    resolved_checkpoint_path = None
    if checkpoint_input_path.is_dir():
        resolved_checkpoint_path = resolve_async_eval_checkpoint_from_index(
            checkpoint_input_path,
            checkpoint_step=checkpoint_step,
        )
    if resolved_checkpoint_path is None:
        resolved_checkpoint_path = resolve_checkpoint_path(
            checkpoint_input_path,
            step=checkpoint_step,
        ).resolve()
    return checkpoint_input_path, resolved_checkpoint_path


def _build_decision_obs(
    *,
    obs: dict[str, Any],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: Any,
    residual_alpha: float,
    timer: Timer,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with timer.context("decision_obs_extract"):
        robot_state = build_libero_state(obs)
        image_observations = extract_libero_images(obs)
    with timer.context("decision_obs_policy_input"):
        base_policy_input = build_libero_policy_input(
            prompt=task_prompt,
            state=robot_state,
            images=image_observations,
        )
    with timer.context("policy_infer"):
        base_actions, _ = policy_client.infer(base_policy_input)
    with timer.context("decision_obs_residual"):
        base_actions = prepare_base_actions_chunk(
            base_actions=base_actions,
            chunk_horizon=chunk_horizon,
        )
        residual_obs = build_chunk_residual_obs(
            robot_state=robot_state,
            images=image_observations,
            image_keys=image_keys,
            base_actions=base_actions,
            residual_alpha=residual_alpha,
        )
    return np.asarray(base_actions, dtype=np.float32), residual_obs


def _build_decision_obs_many(
    *,
    observations: list[dict[str, Any]],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: Any,
    residual_alpha: float,
    timer: Timer,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]], int, int, bool]:
    if not observations:
        return [], [], 0, 0, False

    with timer.context("decision_obs_extract"):
        robot_states = [build_libero_state(obs) for obs in observations]
        image_batches = [extract_libero_images(obs) for obs in observations]
    with timer.context("decision_obs_policy_input"):
        policy_inputs = [
            build_libero_policy_input(
                prompt=task_prompt,
                state=robot_state,
                images=image_observations,
            )
            for robot_state, image_observations in zip(robot_states, image_batches)
        ]

    infer_many = getattr(policy_client, "infer_many", None)
    client_calls = 0
    used_infer_many = callable(infer_many)
    with timer.context("policy_batch_infer"):
        if used_infer_many:
            raw_action_chunks, _batch_info = infer_many(policy_inputs)
            client_calls = 1
        else:
            raw_action_chunks = []
            for policy_input in policy_inputs:
                action_chunk, _info = policy_client.infer(policy_input)
                raw_action_chunks.append(action_chunk)
                client_calls += 1

    if len(raw_action_chunks) != len(observations):
        raise ValueError(
            "batched policy response length mismatch: "
            f"got {len(raw_action_chunks)}, expected {len(observations)}"
        )

    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    with timer.context("decision_obs_residual"):
        for robot_state, image_observations, raw_actions in zip(
            robot_states,
            image_batches,
            raw_action_chunks,
        ):
            base_actions = prepare_base_actions_chunk(
                base_actions=raw_actions,
                chunk_horizon=chunk_horizon,
            )
            residual_obs = build_chunk_residual_obs(
                robot_state=robot_state,
                images=image_observations,
                image_keys=image_keys,
                base_actions=base_actions,
                residual_alpha=residual_alpha,
            )
            base_action_chunks.append(np.asarray(base_actions, dtype=np.float32))
            residual_observations.append(residual_obs)
    return (
        base_action_chunks,
        residual_observations,
        int(client_calls),
        len(observations),
        bool(used_infer_many),
    )


def _normalize_chunk_result(chunk_result: dict[str, Any]) -> tuple[
    list[float],
    list[bool],
    list[dict[str, Any]],
    int,
    dict[str, Any],
    bool,
]:
    rewards = [float(value) for value in chunk_result["rewards"]]
    dones = [bool(value) for value in chunk_result["dones"]]
    infos = [dict(value) for value in chunk_result["infos"]]
    executed_steps = int(chunk_result.get("num_steps", len(rewards)))
    if executed_steps <= 0:
        raise RuntimeError("eval step_chunk returned no executed steps")
    if len(rewards) < executed_steps or len(dones) < executed_steps:
        raise RuntimeError("eval step_chunk returned fewer rewards/dones than num_steps")
    if len(infos) < executed_steps:
        raise RuntimeError("eval step_chunk returned fewer infos than num_steps")

    rewards = rewards[:executed_steps]
    dones = dones[:executed_steps]
    infos = infos[:executed_steps]
    last_info = dict(chunk_result.get("info", infos[-1]))
    env_episode_done = bool(
        chunk_result.get("done", dones[-1])
        or chunk_result.get("truncated", False)
    )
    return rewards, dones, infos, executed_steps, last_info, env_episode_done


def _episode_record(
    *,
    episode_id: int,
    init_episode_idx: int,
    seed: int,
    episode_success: bool,
    episode_return: float,
    episode_steps: int,
    completed_episodes: int,
    successes: int,
    env_episode_done: bool,
    manual_cap_reached: bool,
    last_info: dict[str, Any],
    current_init_state_idx: Any,
    checkpoint_step: int | None,
) -> dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "init_episode_idx": int(init_episode_idx),
        "seed": int(seed),
        "success": bool(episode_success),
        "episode_return": float(episode_return),
        "episode_steps": int(episode_steps),
        "running_success_rate": float(successes / max(1, completed_episodes)),
        "env_episode_done": bool(env_episode_done),
        "manual_cap_reached": bool(manual_cap_reached),
        "init_state_idx": last_info.get("init_state_idx", current_init_state_idx),
        "checkpoint_step": checkpoint_step,
        "final_info": to_jsonable(last_info),
    }


def _cfg_with_remote_port(cfg: LiberoEvalConfig, port: int) -> LiberoEvalConfig:
    if str(cfg.env.backend) != "remote":
        return cfg
    if is_dataclass(cfg):
        return replace(
            cfg,
            env=replace(
                cfg.env,
                remote=replace(cfg.env.remote, port=int(port)),
            ),
        )
    remote = SimpleNamespace(**vars(cfg.env.remote))
    remote.port = int(port)
    env = SimpleNamespace(**vars(cfg.env))
    env.remote = remote
    cloned = SimpleNamespace(**vars(cfg))
    cloned.env = env
    return cloned  # type: ignore[return-value]


def _run_serial_eval_loop(
    *,
    env: Any,
    cfg: LiberoEvalConfig,
    episodes: int,
    start_episode_idx: int,
    max_env_steps_per_episode: int | None,
    deterministic: bool,
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: Any,
    residual_action_spec: ResidualActionSpec,
    agent: Any,
    checkpoint_step: int | None,
    timer: Timer,
    episode_logger: JsonlWriter,
    logger: logging.Logger,
) -> _EvalLoopStats:
    stats = _EvalLoopStats(episode_returns=[], episode_steps_list=[])

    for episode_id in range(episodes):
        init_episode_idx = start_episode_idx + episode_id
        reset_seed = cfg.env.seed
        obs = env.reset(seed=reset_seed, init_episode_idx=init_episode_idx)
        prefetched = None
        episode_return = 0.0
        episode_steps = 0
        episode_success = False
        env_episode_done = False
        manual_cap_reached = False
        last_info: dict[str, Any] = {}

        while True:
            if (
                max_env_steps_per_episode is not None
                and episode_steps >= max_env_steps_per_episode
            ):
                manual_cap_reached = True
                break

            remaining_episode_budget = (
                None
                if max_env_steps_per_episode is None
                else max(0, int(max_env_steps_per_episode - episode_steps))
            )
            if remaining_episode_budget is not None and remaining_episode_budget <= 0:
                manual_cap_reached = True
                break
            execute_horizon = (
                int(chunk_horizon)
                if remaining_episode_budget is None
                else min(int(chunk_horizon), int(remaining_episode_budget))
            )

            with timer.context("total"):
                with timer.context("sample_actions"):
                    if prefetched is None:
                        with timer.context("build_decision_obs"):
                            base_actions, residual_obs = _build_decision_obs(
                                obs=obs,
                                task_prompt=task_prompt,
                                policy_client=policy_client,
                                chunk_horizon=chunk_horizon,
                                image_keys=image_keys,
                                residual_alpha=residual_action_spec.alpha,
                                timer=timer,
                            )
                        stats.policy_requests += 1
                        stats.policy_samples += 1
                    else:
                        base_actions = prefetched["base_actions"]
                        residual_obs = prefetched["residual_obs"]
                        prefetched = None

                    if agent is None:
                        residual_actions = np.zeros(
                            (
                                int(chunk_horizon),
                                int(residual_action_spec.policy_action_dim),
                            ),
                            dtype=np.float32,
                        )
                    else:
                        residual_actions = agent.sample_action(
                            residual_obs,
                            deterministic=deterministic,
                        )

                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                action_chunk = np.asarray(
                    final_actions[:execute_horizon],
                    dtype=np.float32,
                )

                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)

                (
                    rewards,
                    _dones,
                    infos,
                    executed_steps,
                    last_info,
                    env_episode_done,
                ) = _normalize_chunk_result(chunk_result)

                episode_steps += int(executed_steps)
                stats.total_env_steps += int(executed_steps)
                episode_return += float(sum(rewards))
                episode_success = bool(
                    episode_success
                    or any(
                        bool(info.get("env_done", False))
                        or bool(info.get("success", False))
                        for info in infos
                    )
                )
                obs = dict(chunk_result["obs"])
                if (
                    (not env_episode_done)
                    and max_env_steps_per_episode is not None
                    and episode_steps >= max_env_steps_per_episode
                ):
                    manual_cap_reached = True

                if not (env_episode_done or manual_cap_reached):
                    with timer.context("build_decision_obs"):
                        next_base_actions, next_residual_obs = _build_decision_obs(
                            obs=obs,
                            task_prompt=task_prompt,
                            policy_client=policy_client,
                            chunk_horizon=chunk_horizon,
                            image_keys=image_keys,
                            residual_alpha=residual_action_spec.alpha,
                            timer=timer,
                        )
                    stats.policy_requests += 1
                    stats.policy_samples += 1
                    prefetched = {
                        "base_actions": next_base_actions,
                        "residual_obs": next_residual_obs,
                    }
                else:
                    prefetched = None

            if env_episode_done or manual_cap_reached:
                break

        stats.completed_episodes += 1
        stats.successes += int(episode_success)
        stats.episode_returns.append(float(episode_return))
        stats.episode_steps_list.append(int(episode_steps))

        episode_logger.write(
            to_jsonable(
                _episode_record(
                    episode_id=int(episode_id),
                    init_episode_idx=int(init_episode_idx),
                    seed=int(reset_seed),
                    episode_success=bool(episode_success),
                    episode_return=float(episode_return),
                    episode_steps=int(episode_steps),
                    completed_episodes=int(stats.completed_episodes),
                    successes=int(stats.successes),
                    env_episode_done=bool(env_episode_done),
                    manual_cap_reached=bool(manual_cap_reached),
                    last_info=last_info,
                    current_init_state_idx=getattr(
                        env,
                        "current_init_state_idx",
                        None,
                    ),
                    checkpoint_step=checkpoint_step,
                )
            )
        )

        logger.info(
            "eval episode=%s init_episode_idx=%s success=%s steps=%s return=%.3f",
            int(episode_id),
            int(init_episode_idx),
            bool(episode_success),
            int(episode_steps),
            float(episode_return),
        )

    return stats


def _run_parallel_eval_loop(
    *,
    envs: list[Any],
    cfg: LiberoEvalConfig,
    episodes: int,
    start_episode_idx: int,
    max_env_steps_per_episode: int | None,
    deterministic: bool,
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: Any,
    residual_action_spec: ResidualActionSpec,
    agent: Any,
    checkpoint_step: int | None,
    timer: Timer,
    episode_logger: JsonlWriter,
    logger: logging.Logger,
) -> _EvalLoopStats:
    stats = _EvalLoopStats(
        episode_returns=[],
        episode_steps_list=[],
        active_lane_counts=[],
    )
    policy_batch_size = max(1, int(getattr(cfg.eval, "policy_batch_size", len(envs))))
    lanes = [_EvalLane(lane_id=idx, env=env) for idx, env in enumerate(envs)]
    next_episode_id = 0

    def prepare_lane(lane: _EvalLane) -> bool:
        nonlocal next_episode_id
        if next_episode_id >= episodes:
            lane.active = False
            lane.obs = None
            return False
        episode_id = int(next_episode_id)
        next_episode_id += 1
        init_episode_idx = int(start_episode_idx + episode_id)
        lane.active = False
        lane.episode_id = episode_id
        lane.init_episode_idx = init_episode_idx
        lane.obs = None
        lane.episode_return = 0.0
        lane.episode_steps = 0
        lane.episode_success = False
        lane.env_episode_done = False
        lane.manual_cap_reached = False
        lane.last_info = {}
        return True

    with ThreadPoolExecutor(max_workers=max(1, len(envs))) as executor:
        def start_lanes(lanes_to_start: list[_EvalLane]) -> None:
            reset_lanes = [lane for lane in lanes_to_start if prepare_lane(lane)]
            if not reset_lanes:
                return
            with timer.context("reset_env"):
                future_to_lane = {
                    executor.submit(
                        lane.env.reset,
                        seed=int(cfg.env.seed),
                        init_episode_idx=int(lane.init_episode_idx),
                    ): lane
                    for lane in reset_lanes
                }
                for future in as_completed(future_to_lane):
                    lane = future_to_lane[future]
                    lane.obs = dict(future.result())
                    lane.active = True

        start_lanes(lanes)
        while stats.completed_episodes < episodes:
            active_lanes = [lane for lane in lanes if lane.active]
            if not active_lanes:
                break
            assert stats.active_lane_counts is not None
            stats.active_lane_counts.append(len(active_lanes))

            lane_actions: dict[int, np.ndarray] = {}
            for start_idx in range(0, len(active_lanes), policy_batch_size):
                batch_lanes = active_lanes[start_idx : start_idx + policy_batch_size]
                batch_observations = []
                for lane in batch_lanes:
                    if lane.obs is None:
                        raise RuntimeError("active eval lane has no observation")
                    batch_observations.append(lane.obs)
                with timer.context("build_decision_obs"):
                    (
                        base_action_chunks,
                        residual_observations,
                        client_calls,
                        policy_samples,
                        used_infer_many,
                    ) = _build_decision_obs_many(
                        observations=batch_observations,
                        task_prompt=task_prompt,
                        policy_client=policy_client,
                        chunk_horizon=chunk_horizon,
                        image_keys=image_keys,
                        residual_alpha=residual_action_spec.alpha,
                        timer=timer,
                    )
                stats.policy_requests += int(client_calls)
                stats.policy_samples += int(policy_samples)
                if used_infer_many:
                    stats.policy_batch_requests += 1

                for lane, base_actions, residual_obs in zip(
                    batch_lanes,
                    base_action_chunks,
                    residual_observations,
                ):
                    if agent is None:
                        residual_actions = np.zeros(
                            (
                                int(chunk_horizon),
                                int(residual_action_spec.policy_action_dim),
                            ),
                            dtype=np.float32,
                        )
                    else:
                        residual_actions = agent.sample_action(
                            residual_obs,
                            deterministic=deterministic,
                        )
                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )
                    remaining_episode_budget = (
                        None
                        if max_env_steps_per_episode is None
                        else max(
                            0,
                            int(max_env_steps_per_episode - lane.episode_steps),
                        )
                    )
                    execute_horizon = (
                        int(chunk_horizon)
                        if remaining_episode_budget is None
                        else min(int(chunk_horizon), int(remaining_episode_budget))
                    )
                    lane_actions[int(lane.lane_id)] = np.asarray(
                        final_actions[:execute_horizon],
                        dtype=np.float32,
                    )

            with timer.context("step_env"):
                future_to_lane = {
                    executor.submit(lane.env.step_chunk, lane_actions[lane.lane_id]): lane
                    for lane in active_lanes
                }
                completed = [
                    (future_to_lane[future], future.result())
                    for future in as_completed(future_to_lane)
                ]

            lanes_to_restart: list[_EvalLane] = []
            for lane, chunk_result in sorted(completed, key=lambda item: item[0].lane_id):
                (
                    rewards,
                    _dones,
                    infos,
                    executed_steps,
                    last_info,
                    env_episode_done,
                ) = _normalize_chunk_result(chunk_result)
                lane.last_info = last_info
                lane.episode_steps += int(executed_steps)
                stats.total_env_steps += int(executed_steps)
                lane.episode_return += float(sum(rewards))
                lane.episode_success = bool(
                    lane.episode_success
                    or any(
                        bool(info.get("env_done", False))
                        or bool(info.get("success", False))
                        for info in infos
                    )
                )
                lane.obs = dict(chunk_result["obs"])
                lane.env_episode_done = bool(env_episode_done)
                if (
                    (not lane.env_episode_done)
                    and max_env_steps_per_episode is not None
                    and lane.episode_steps >= max_env_steps_per_episode
                ):
                    lane.manual_cap_reached = True

                if lane.env_episode_done or lane.manual_cap_reached:
                    stats.completed_episodes += 1
                    stats.successes += int(lane.episode_success)
                    stats.episode_returns.append(float(lane.episode_return))
                    stats.episode_steps_list.append(int(lane.episode_steps))

                    episode_logger.write(
                        to_jsonable(
                            _episode_record(
                                episode_id=int(lane.episode_id),
                                init_episode_idx=int(lane.init_episode_idx),
                                seed=int(cfg.env.seed),
                                episode_success=bool(lane.episode_success),
                                episode_return=float(lane.episode_return),
                                episode_steps=int(lane.episode_steps),
                                completed_episodes=int(stats.completed_episodes),
                                successes=int(stats.successes),
                                env_episode_done=bool(lane.env_episode_done),
                                manual_cap_reached=bool(lane.manual_cap_reached),
                                last_info=dict(lane.last_info or {}),
                                current_init_state_idx=getattr(
                                    lane.env,
                                    "current_init_state_idx",
                                    None,
                                ),
                                checkpoint_step=checkpoint_step,
                            )
                        )
                    )

                    logger.info(
                        "eval lane=%s episode=%s init_episode_idx=%s success=%s "
                        "steps=%s return=%.3f",
                        int(lane.lane_id),
                        int(lane.episode_id),
                        int(lane.init_episode_idx),
                        bool(lane.episode_success),
                        int(lane.episode_steps),
                        float(lane.episode_return),
                    )
                    lane.active = False
                    lanes_to_restart.append(lane)
            start_lanes(lanes_to_restart)

    return stats


def run_residual_eval(
    cfg: LiberoEvalConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    original_cwd: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes = _positive_int(cfg.eval.episodes, "eval.episodes")
    start_episode_idx = _nonnegative_int(
        cfg.eval.start_episode_idx,
        "eval.start_episode_idx",
    )
    max_env_steps_per_episode = _optional_positive_int(
        cfg.eval.max_env_steps_per_episode,
        "eval.max_env_steps_per_episode",
    )
    deterministic = bool(cfg.eval.deterministic)
    episode_log_file = str(cfg.logging.episode_log_file or "episode_logs.jsonl")

    checkpoint_input_path, checkpoint_file = _resolve_checkpoint_input(
        cfg.eval.checkpoint_path,
        cfg.eval.checkpoint_step,
        original_cwd=original_cwd,
    )

    logger.info("Eval run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(cfg), indent=2))

    set_global_seeds(cfg.global_seed)

    parallel_envs = _positive_int(
        getattr(cfg.eval, "parallel_envs", 1),
        "eval.parallel_envs",
    )
    policy_batch_size = _positive_int(
        getattr(cfg.eval, "policy_batch_size", parallel_envs),
        "eval.policy_batch_size",
    )
    env_ports: tuple[int, ...] = ()
    if parallel_envs > 1:
        if str(cfg.env.backend) != "remote":
            raise ValueError("eval.parallel_envs > 1 requires env.backend=remote")
        env_ports = tuple(getattr(cfg.env.remote, "ports", None) or ())
        if len(env_ports) != parallel_envs:
            raise ValueError(
                "env.remote.ports length must equal eval.parallel_envs: "
                f"got {len(env_ports)} ports for {parallel_envs} envs"
            )
        env_cfgs = [_cfg_with_remote_port(cfg, int(port)) for port in env_ports]
    else:
        env_ports = tuple(getattr(cfg.env.remote, "ports", None) or ())
        if env_ports:
            if len(env_ports) != 1:
                raise ValueError(
                    "env.remote.ports must contain exactly one port when "
                    "eval.parallel_envs=1"
                )
            env_cfgs = [_cfg_with_remote_port(cfg, int(env_ports[0]))]
        else:
            env_cfgs = [cfg]

    envs: list[Any] = []
    if len(set(int(port) for port in env_ports)) != len(env_ports):
        raise ValueError(f"env.remote.ports must not contain duplicates: {env_ports}")
    policy_client = None
    episode_logger: JsonlWriter | None = None
    timer = Timer()
    stats = _EvalLoopStats(episode_returns=[], episode_steps_list=[])
    eval_env_ports = (
        [int(port) for port in env_ports]
        if env_ports
        else (
            [int(cfg.env.remote.port)]
            if str(getattr(cfg.env, "backend", "")) == "remote"
            else []
        )
    )
    agent = None
    checkpoint_loaded = False
    checkpoint_step = None
    summary: dict[str, Any] | None = None

    try:
        for env_cfg in env_cfgs:
            envs.append(create_env(env_cfg, logger))
        task_prompt = str(envs[0].task_description)
        policy_backend = describe_policy_backend(cfg)
        logger.info("Chunk policy backend: %s", policy_backend)

        image_keys = cfg.obs.image_keys
        action_dim = cfg.env.action_dim
        chunk_horizon = cfg.residual.chunk_horizon
        residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)

        summary = {
            "role": "eval",
            "mode": "residual",
            "episodes_requested": int(episodes),
            "episodes_completed": 0,
            "successes": 0,
            "success_rate": 0.0,
            "mean_return": 0.0,
            "mean_episode_steps": 0.0,
            "env_steps": 0,
            "deterministic": bool(deterministic),
            "checkpoint_loaded": bool(checkpoint_loaded),
            "checkpoint_path": None if checkpoint_file is None else str(checkpoint_file),
            "checkpoint_input_path": (
                None if checkpoint_input_path is None else str(checkpoint_input_path)
            ),
            "checkpoint_step": checkpoint_step,
            "base_policy_type": resolve_policy_backend_type(cfg),
            "base_policy_id": resolve_policy_backend_id(cfg),
            "base_policy_backend": policy_backend,
            "start_episode_idx": int(start_episode_idx),
            "max_env_steps_per_episode": max_env_steps_per_episode,
            "parallel_envs": int(parallel_envs),
            "policy_batch_size": int(policy_batch_size),
            "eval_env_ports": eval_env_ports,
            "task": {
                "suite_name": cfg.task.suite_name,
                "task_id": int(cfg.task.task_id),
                "task_description": task_prompt,
            },
        }

        policy_client = build_policy_client(cfg, logger=logger)

        if checkpoint_file is not None:
            sample_obs = build_chunk_residual_sample_obs(
                state_dim=LIBERO_STATE_DIM,
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
                image_keys=image_keys,
                image_height=RESIDUAL_IMAGE_HEIGHT,
                image_width=RESIDUAL_IMAGE_WIDTH,
            )
            agent = create_drq_agent_from_typed_cfg(
                cfg,
                sample_obs=sample_obs,
                action_dim=residual_action_spec.chunk_policy_action_dim,
                image_keys=image_keys,
                critic_action_dim=residual_action_spec.chunk_critic_action_dim,
                action_transform=residual_action_spec.build_chunk_action_transform(),
            )
            checkpoint_payload = load_checkpoint_payload(checkpoint_file)
            apply_checkpoint_payload_to_agent(
                agent,
                dict(checkpoint_payload),
                load_optimizers=False,
            )
            checkpoint_loaded = True
            if "step" in checkpoint_payload:
                checkpoint_step = int(checkpoint_payload["step"])
            else:
                checkpoint_step = _checkpoint_step_from_path(checkpoint_file)
            summary["checkpoint_loaded"] = bool(checkpoint_loaded)
            summary["checkpoint_step"] = checkpoint_step
            logger.info(
                "Loaded residual checkpoint from: %s%s",
                checkpoint_file,
                "" if checkpoint_step is None else f" (step={int(checkpoint_step)})",
            )
        else:
            logger.warning(
                "eval.checkpoint_path is not set; running base-policy-only evaluation "
                "with zero residual actions"
            )

        episode_logger = JsonlWriter(run_dir / episode_log_file)
        if parallel_envs == 1:
            stats = _run_serial_eval_loop(
                env=envs[0],
                cfg=cfg,
                episodes=episodes,
                start_episode_idx=start_episode_idx,
                max_env_steps_per_episode=max_env_steps_per_episode,
                deterministic=deterministic,
                task_prompt=task_prompt,
                policy_client=policy_client,
                chunk_horizon=chunk_horizon,
                image_keys=image_keys,
                residual_action_spec=residual_action_spec,
                agent=agent,
                checkpoint_step=checkpoint_step,
                timer=timer,
                episode_logger=episode_logger,
                logger=logger,
            )
        else:
            stats = _run_parallel_eval_loop(
                envs=envs,
                cfg=cfg,
                episodes=episodes,
                start_episode_idx=start_episode_idx,
                max_env_steps_per_episode=max_env_steps_per_episode,
                deterministic=deterministic,
                task_prompt=task_prompt,
                policy_client=policy_client,
                chunk_horizon=chunk_horizon,
                image_keys=image_keys,
                residual_action_spec=residual_action_spec,
                agent=agent,
                checkpoint_step=checkpoint_step,
                timer=timer,
                episode_logger=episode_logger,
                logger=logger,
            )

    finally:
        try:
            if summary is not None:
                active_lane_counts = stats.active_lane_counts or []
                mean_active_lanes = (
                    float(np.mean(np.asarray(active_lane_counts, dtype=np.float32)))
                    if active_lane_counts
                    else (1.0 if stats.completed_episodes > 0 else 0.0)
                )
                summary.update(
                    {
                        "episodes_completed": int(stats.completed_episodes),
                        "successes": int(stats.successes),
                        "success_rate": (
                            float(stats.successes / stats.completed_episodes)
                            if stats.completed_episodes > 0
                            else 0.0
                        ),
                        "mean_return": (
                            float(
                                np.mean(
                                    np.asarray(
                                        stats.episode_returns,
                                        dtype=np.float32,
                                    )
                                )
                            )
                            if stats.episode_returns
                            else 0.0
                        ),
                        "mean_episode_steps": (
                            float(
                                np.mean(
                                    np.asarray(
                                        stats.episode_steps_list,
                                        dtype=np.float32,
                                    )
                                )
                            )
                            if stats.episode_steps_list
                            else 0.0
                        ),
                        "env_steps": int(stats.total_env_steps),
                        "policy_requests": int(stats.policy_requests),
                        "policy_batch_requests": int(stats.policy_batch_requests),
                        "policy_samples": int(stats.policy_samples),
                        "policy_requests_per_env_step": (
                            float(stats.policy_requests / stats.total_env_steps)
                            if stats.total_env_steps > 0
                            else 0.0
                        ),
                        "policy_samples_per_env_step": (
                            float(stats.policy_samples / stats.total_env_steps)
                            if stats.total_env_steps > 0
                            else 0.0
                        ),
                        "mean_active_lanes": float(mean_active_lanes),
                        "timer": to_jsonable(timer.get_average_times()),
                    }
                )
                with open(
                    run_dir / cfg.logging.summary_file,
                    "w",
                    encoding="utf-8",
                ) as fp:
                    json.dump(summary, fp, indent=2)
        finally:
            if episode_logger is not None:
                try:
                    episode_logger.close()
                except Exception:  # noqa: BLE001
                    pass
            for env in envs:
                try:
                    env.close(clear_cache=False)
                except Exception:  # noqa: BLE001
                    pass
            if policy_client is not None:
                policy_client_close = getattr(policy_client, "close", None)
                if callable(policy_client_close):
                    try:
                        policy_client_close()
                    except Exception:  # noqa: BLE001
                        pass

    if summary is None:
        raise RuntimeError("eval summary was not initialized")
    return summary


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="eval_residual",
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    typed_cfg = parse_eval_cfg(cfg)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_eval")
    summary = run_residual_eval(
        typed_cfg,
        run_dir=run_dir,
        logger=logger,
        original_cwd=Path(get_original_cwd()).resolve(),
    )
    logger.info("evaluation done: %s", summary)


if __name__ == "__main__":
    main()
