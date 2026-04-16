from __future__ import annotations

"""Reusable AgiBot residual checkpoint evaluation runner."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.checkpoint_utils import load_checkpoint_payload
from serl_launcher.utils.checkpoint_utils import resolve_checkpoint_path
from serl_launcher.utils.jsonl import JsonlWriter
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

from serl_torch.examples.agibot_real.config import AgiBotEvalConfig
from serl_torch.examples.agibot_real.config import cfg_to_log_payload
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.env.factory import create_env
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_obs,
)
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_sample_obs,
)
from serl_torch.examples.agibot_real.torch_compile import maybe_enable_torch_compile


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
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
        base_dir = Path.cwd().resolve() if original_cwd is None else original_cwd.resolve()
        checkpoint_input_path = base_dir / checkpoint_input_path
    checkpoint_input_path = checkpoint_input_path.resolve()

    checkpoint_step = (
        None
        if checkpoint_step_raw is None
        else _positive_int(checkpoint_step_raw, "eval.checkpoint_step")
    )
    resolved_checkpoint_path = resolve_checkpoint_path(
        checkpoint_input_path,
        step=checkpoint_step,
    ).resolve()
    return checkpoint_input_path, resolved_checkpoint_path


def run_eval(
    cfg: AgiBotEvalConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    original_cwd: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes = _positive_int(cfg.eval.episodes, "eval.episodes")
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
    if checkpoint_file is None:
        raise ValueError(
            "AgiBot standalone eval requires eval.checkpoint_path to be set"
        )

    logger.info("Eval run dir: %s", run_dir)
    logger.info(
        "Config:\n%s",
        json.dumps(cfg_to_log_payload(cfg), indent=2),
    )

    set_global_seeds(cfg.global_seed)

    env = create_env(cfg, logger)
    base_policy = build_agibot_base_policy(cfg, logger=logger)
    logger.info("Chunk policy backend: %s", base_policy.describe())

    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    image_keys = cfg.obs.image_keys
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)

    sample_obs = build_chunk_residual_sample_obs(
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
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
    agent = maybe_enable_torch_compile(
        agent,
        compile_cfg=cfg.training.torch_compile,
        logger=logger,
    )
    checkpoint_step = (
        int(checkpoint_payload["step"])
        if "step" in checkpoint_payload
        else _checkpoint_step_from_path(checkpoint_file)
    )
    logger.info(
        "Loaded residual checkpoint from: %s%s",
        checkpoint_file,
        "" if checkpoint_step is None else f" (step={int(checkpoint_step)})",
    )

    timer = Timer()
    episode_logger = JsonlWriter(run_dir / episode_log_file)
    episode_returns: list[float] = []
    episode_steps_list: list[int] = []
    successes = 0
    total_env_steps = 0
    completed_episodes = 0

    summary: dict[str, Any] = {
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
        "checkpoint_loaded": True,
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_input_path": (
            None if checkpoint_input_path is None else str(checkpoint_input_path)
        ),
        "checkpoint_step": checkpoint_step,
        "base_policy_backend": base_policy.describe(),
        "episode_log_path": str(run_dir / episode_log_file),
        "max_env_steps_per_episode": max_env_steps_per_episode,
        "task": {
            "task_name": str(cfg.task.name),
            "task_key": str(cfg.task.task_key),
            "task_description": str(cfg.task.prompt),
        },
    }

    try:
        for episode_id in range(episodes):
            obs = env.reset()
            task_prompt = str(env.task_description)
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

                timer.tick("total")
                with timer.context("sample_actions"):
                    if prefetched is None:
                        base_actions, _ = base_policy.infer(
                            obs,
                            prompt=task_prompt,
                        )
                        residual_obs = build_chunk_residual_obs(
                            obs=obs,
                            base_actions=base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_action_spec.alpha,
                        )
                    else:
                        base_actions = prefetched["base_actions"]
                        residual_obs = prefetched["residual_obs"]
                        prefetched = None

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
                    else max(0, int(max_env_steps_per_episode - episode_steps))
                )
                if remaining_episode_budget is not None and remaining_episode_budget <= 0:
                    manual_cap_reached = True
                    timer.tock("total")
                    break
                execute_horizon = (
                    int(chunk_horizon)
                    if remaining_episode_budget is None
                    else min(int(chunk_horizon), int(remaining_episode_budget))
                )

                for action in np.asarray(final_actions[:execute_horizon], dtype=np.float32):
                    with timer.context("step_env"):
                        next_obs, reward, done, truncated, info = env.step(action)

                    done_flag = bool(done or truncated)
                    action_executed = bool(info.get("controller_action_executed", True))
                    last_info = dict(info)

                    if not action_executed:
                        if not done_flag:
                            raise RuntimeError(
                                "controller reported an unexecuted action without a terminal outcome"
                            )
                        episode_return += float(reward)
                        episode_success = bool(
                            episode_success or info.get("success", False)
                        )
                        obs = dict(next_obs)
                        prefetched = None
                        env_episode_done = True
                        break

                    with timer.context("build_decision_obs"):
                        next_base_actions, _ = base_policy.infer(
                            next_obs,
                            prompt=task_prompt,
                        )
                        next_residual_obs = build_chunk_residual_obs(
                            obs=next_obs,
                            base_actions=next_base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_action_spec.alpha,
                        )

                    episode_steps += 1
                    total_env_steps += 1
                    episode_return += float(reward)
                    episode_success = bool(episode_success or info.get("success", False))
                    obs = dict(next_obs)
                    prefetched = {
                        "base_actions": next_base_actions,
                        "residual_obs": next_residual_obs,
                    }

                    if done_flag:
                        env_episode_done = True
                        break

                timer.tock("total")

                if env_episode_done or manual_cap_reached:
                    break

            completed_episodes += 1
            successes += int(episode_success)
            episode_returns.append(float(episode_return))
            episode_steps_list.append(int(episode_steps))

            episode_record = {
                "episode_id": int(episode_id),
                "success": bool(episode_success),
                "episode_return": float(episode_return),
                "episode_steps": int(episode_steps),
                "running_success_rate": float(successes / max(1, completed_episodes)),
                "env_episode_done": bool(env_episode_done),
                "manual_cap_reached": bool(manual_cap_reached),
                "checkpoint_step": checkpoint_step,
                "final_info": to_jsonable(last_info),
            }
            episode_logger.write(to_jsonable(episode_record))

            logger.info(
                "eval episode=%s success=%s steps=%s return=%.3f",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
            )

    finally:
        summary.update(
            {
                "episodes_completed": int(completed_episodes),
                "successes": int(successes),
                "success_rate": (
                    float(successes / completed_episodes)
                    if completed_episodes > 0
                    else 0.0
                ),
                "mean_return": (
                    float(np.mean(np.asarray(episode_returns, dtype=np.float32)))
                    if episode_returns
                    else 0.0
                ),
                "mean_episode_steps": (
                    float(np.mean(np.asarray(episode_steps_list, dtype=np.float32)))
                    if episode_steps_list
                    else 0.0
                ),
                "env_steps": int(total_env_steps),
                "timer": to_jsonable(timer.get_average_times()),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            episode_logger.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        try:
            base_policy.close()
        except Exception:  # noqa: BLE001
            pass

    return summary
