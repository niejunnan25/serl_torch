from __future__ import annotations

"""Canonical LIBERO residual checkpoint evaluation entrypoint."""

import json
import logging
import sys
from pathlib import Path
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

    env = create_env(cfg, logger)
    task_prompt = str(env.task_description)
    policy_client = build_policy_client(cfg, logger=logger)
    policy_backend = describe_policy_backend(cfg)
    logger.info("Chunk policy backend: %s", policy_backend)

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)

    agent = None
    checkpoint_loaded = False
    checkpoint_step = None
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
        "task": {
            "suite_name": cfg.task.suite_name,
            "task_id": int(cfg.task.task_id),
            "task_description": task_prompt,
        },
    }

    try:
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

                timer.tick("total")
                with timer.context("sample_actions"):
                    if prefetched is None:
                        robot_state = build_libero_state(obs)
                        image_observations = extract_libero_images(obs)
                        base_policy_input = build_libero_policy_input(
                            prompt=task_prompt,
                            state=robot_state,
                            images=image_observations,
                        )
                        base_actions, _ = policy_client.infer(base_policy_input)
                        base_actions = prepare_base_actions_chunk(
                            base_actions=base_actions,
                            chunk_horizon=chunk_horizon,
                        )
                        residual_obs = build_chunk_residual_obs(
                            robot_state=robot_state,
                            images=image_observations,
                            image_keys=image_keys,
                            base_actions=base_actions,
                            residual_alpha=residual_action_spec.alpha,
                        )
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

                for action in np.asarray(
                    final_actions[:execute_horizon],
                    dtype=np.float32,
                ):
                    with timer.context("step_env"):
                        next_obs, reward, done, truncated, info = env.step(action)

                    with timer.context("build_decision_obs"):
                        next_robot_state = build_libero_state(next_obs)
                        next_image_observations = extract_libero_images(next_obs)
                        next_base_policy_input = build_libero_policy_input(
                            prompt=task_prompt,
                            state=next_robot_state,
                            images=next_image_observations,
                        )
                        next_base_actions, _ = policy_client.infer(next_base_policy_input)
                        next_base_actions = prepare_base_actions_chunk(
                            base_actions=next_base_actions,
                            chunk_horizon=chunk_horizon,
                        )
                        next_residual_obs = build_chunk_residual_obs(
                            robot_state=next_robot_state,
                            images=next_image_observations,
                            image_keys=image_keys,
                            base_actions=next_base_actions,
                            residual_alpha=residual_action_spec.alpha,
                        )

                    env_done = bool(info.get("env_done", False))
                    last_info = dict(info)
                    episode_steps += 1
                    total_env_steps += 1
                    episode_return += float(reward)
                    episode_success = bool(
                        episode_success
                        or env_done
                        or bool(info.get("success", False))
                    )
                    obs = dict(next_obs)
                    prefetched = {
                        "base_actions": next_base_actions,
                        "residual_obs": next_residual_obs,
                    }

                    if bool(done or truncated):
                        env_episode_done = True
                        break
                    if (
                        max_env_steps_per_episode is not None
                        and episode_steps >= max_env_steps_per_episode
                    ):
                        manual_cap_reached = True
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
                "init_episode_idx": int(init_episode_idx),
                "seed": int(reset_seed),
                "success": bool(episode_success),
                "episode_return": float(episode_return),
                "episode_steps": int(episode_steps),
                "running_success_rate": float(successes / max(1, completed_episodes)),
                "env_episode_done": bool(env_episode_done),
                "manual_cap_reached": bool(manual_cap_reached),
                "init_state_idx": last_info.get(
                    "init_state_idx",
                    getattr(env, "current_init_state_idx", None),
                ),
                "checkpoint_step": checkpoint_step,
                "final_info": to_jsonable(last_info),
            }
            episode_logger.write(to_jsonable(episode_record))

            logger.info(
                "eval episode=%s init_episode_idx=%s success=%s steps=%s return=%.3f",
                int(episode_id),
                int(init_episode_idx),
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
            json.dump(summary, fp, indent=2)
        try:
            episode_logger.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:  # noqa: BLE001
                pass

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
