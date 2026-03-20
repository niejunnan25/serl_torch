"""Offline bootstrap helpers using base policy rollouts."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from ..data import StateActionNormalizer
from ..policy import LiberoObservationCache, OpenPIChunkClient, select_action_chunk_window
from ..utils.obs_utils import _clone_obs_dict, _zero_obs_like
from ..utils.profiling import _RuntimeProfiler, _build_residual_step_obs_profiled, _profile_call

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer


def _bootstrap_offline_with_base_success(
    cfg: DictConfig,
    *,
    env,
    openpi_client: OpenPIChunkClient,
    offline_buffer: "ReplayBuffer",
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_horizon: int,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
    profiler: Optional[_RuntimeProfiler] = None,
) -> Dict[str, int]:
    del sample_obs_template
    stats = {
        "enabled": 0,
        "attempts": 0,
        "episodes_collected": 0,
        "success_episodes": 0,
        "inserted": 0,
        "seed_start": 0,
        "seed_next": 0,
    }

    bootstrap_cfg = cfg.offline.get("bootstrap_base", None)
    if bootstrap_cfg is None or (not bool(bootstrap_cfg.get("enabled", False))):
        return stats

    stats["enabled"] = 1
    target_success_episodes = int(bootstrap_cfg.get("success_episodes", 0))
    if target_success_episodes <= 0:
        logger.warning("offline.bootstrap_base.enabled=true but success_episodes<=0, skip bootstrap")
        return stats

    max_seed_attempts = int(bootstrap_cfg.get("max_seed_attempts", max(1000, target_success_episodes * 100)))
    seed_base_cfg = bootstrap_cfg.get("seed_base", None)
    seed_cursor = int(cfg.task.seed_base) + 1_000_000 if seed_base_cfg is None else int(seed_base_cfg)
    stats["seed_start"] = int(seed_cursor)
    max_ep_steps_override = bootstrap_cfg.get("max_env_steps_per_episode", None)
    only_success = bool(bootstrap_cfg.get("only_success", True))
    obs_cache = LiberoObservationCache()

    while stats["attempts"] < max_seed_attempts and stats["success_episodes"] < target_success_episodes:
        seed = int(seed_cursor)
        seed_cursor += 1
        stats["attempts"] += 1

        obs_cache.clear()
        obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=seed, episode_id=-1)
        max_episode_steps = int(env.step_limit)
        if max_ep_steps_override is not None:
            max_episode_steps = min(max_episode_steps, int(max_ep_steps_override))

        episode_transitions: List[Dict[str, Any]] = []
        episode_steps = 0
        success = False
        episode_done = False
        cached_base_chunk = None

        while episode_steps < max_episode_steps and (not episode_done):
            if cached_base_chunk is None:
                openpi_chunk, _ = openpi_client.infer_chunk(
                    obs_raw,
                    env.current_instruction,
                    obs_cache=obs_cache,
                )
                base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
            else:
                base_chunk = cached_base_chunk
                cached_base_chunk = None

            next_obs_raw = obs_raw
            for chunk_step in range(chunk_horizon):
                if episode_steps >= max_episode_steps:
                    episode_done = True
                    break

                obs_input = _build_residual_step_obs_profiled(
                    profiler,
                    next_obs_raw,
                    base_chunk[chunk_step],
                    image_keys=image_keys,
                    stack_horizon=stack_horizon,
                    normalizer=normalizer,
                    obs_cache=obs_cache,
                )
                next_obs_raw, reward, env_done, _, info = _profile_call(
                    profiler,
                    "env_step",
                    env.step,
                    base_chunk[chunk_step],
                )
                episode_steps += 1
                success = bool(info["success"])
                timeout = bool(episode_steps >= max_episode_steps)
                done = bool(env_done or timeout)

                if done:
                    next_obs_input = _zero_obs_like(obs_input)
                    mask = 0.0
                elif chunk_step < (chunk_horizon - 1):
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        base_chunk[chunk_step + 1],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                    )
                    mask = 1.0
                else:
                    next_openpi_chunk, _ = openpi_client.infer_chunk(
                        next_obs_raw,
                        env.current_instruction,
                        obs_cache=obs_cache,
                    )
                    next_base_chunk = select_action_chunk_window(next_openpi_chunk, horizon=chunk_horizon)
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        next_base_chunk[0],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                    )
                    cached_base_chunk = next_base_chunk
                    mask = 1.0

                episode_transitions.append(
                    {
                        "observations": _clone_obs_dict(obs_input),
                        "actions": np.zeros((action_dim,), dtype=np.float32),
                        "next_observations": _clone_obs_dict(next_obs_input),
                        "rewards": np.float32(reward),
                        "masks": np.float32(mask),
                        "dones": bool(done),
                    }
                )

                if done:
                    episode_done = True
                    break

            obs_raw = next_obs_raw

        should_keep = bool(success or (not only_success))
        if should_keep:
            for transition in episode_transitions:
                offline_buffer.insert(transition)
            stats["inserted"] += int(len(episode_transitions))
            stats["episodes_collected"] += 1
            stats["success_episodes"] += int(success)

    stats["seed_next"] = int(seed_cursor)
    return stats
