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
from serl_torch.examples.agibot_real.transition_assembly import (
    count_executed_steps_from_infos,
)
from serl_torch.examples.agibot_real.video_recorder import AsyncImageVideoRecorder
from serl_torch.examples.agibot_real.video_recorder import AsyncVideoRecorderConfig


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
    video_output_dir = run_dir / str(cfg.video.output_dir)
    video_recorder: AsyncImageVideoRecorder | None = None
    if bool(cfg.video.enabled):
        video_recorder = AsyncImageVideoRecorder(
            config=AsyncVideoRecorderConfig(
                camera_key=str(cfg.video.camera_key),
                fps=float(cfg.video.fps),
                output_dir=video_output_dir,
                max_pending_frames=int(cfg.video.max_pending_frames),
                drop_frames_when_busy=bool(cfg.video.drop_frames_when_busy),
            ),
            logger=logger,
        )
        logger.info(
            "Eval video recording enabled: camera_key=%s fps=%.3f output_dir=%s "
            "max_pending_frames=%s drop_frames_when_busy=%s",
            str(cfg.video.camera_key),
            float(cfg.video.fps),
            video_output_dir,
            int(cfg.video.max_pending_frames),
            bool(cfg.video.drop_frames_when_busy),
        )

    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    image_keys = cfg.obs.image_keys
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
    filter_enabled = bool(cfg.action_filter.enabled)
    filter_alpha = float(cfg.action_filter.alpha)
    filter_max_delta = cfg.action_filter.max_delta
    filter_warmup_steps = int(cfg.action_filter.warmup_steps)
    filter_reset_each_episode = bool(cfg.action_filter.reset_each_episode)
    filtered_residual_delta_prev: np.ndarray | None = None
    filtered_action_total_steps = 0

    if filter_enabled:
        logger.info(
            "Residual-delta filter enabled during eval: alpha=%.4f max_delta=%s "
            "warmup_steps=%s reset_each_episode=%s",
            float(filter_alpha),
            None if filter_max_delta is None else float(filter_max_delta),
            int(filter_warmup_steps),
            bool(filter_reset_each_episode),
        )

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
        "video": {
            "enabled": bool(cfg.video.enabled),
            "camera_key": str(cfg.video.camera_key),
            "fps": float(cfg.video.fps),
            "output_dir": str(video_output_dir),
            "max_pending_frames": int(cfg.video.max_pending_frames),
            "drop_frames_when_busy": bool(cfg.video.drop_frames_when_busy),
        },
        "action_filter": {
            "enabled": bool(filter_enabled),
            "alpha": float(filter_alpha),
            "max_delta": None if filter_max_delta is None else float(filter_max_delta),
            "warmup_steps": int(filter_warmup_steps),
            "reset_each_episode": bool(filter_reset_each_episode),
        },
        "task": {
            "task_name": str(cfg.task.name),
            "task_key": str(cfg.task.task_key),
            "task_description": str(cfg.task.prompt),
        },
    }

    def _filter_residual_delta_chunk(residual_delta_chunk: np.ndarray) -> np.ndarray:
        nonlocal filtered_residual_delta_prev
        nonlocal filtered_action_total_steps
        if not filter_enabled:
            return residual_delta_chunk

        filtered = np.array(residual_delta_chunk, dtype=np.float32, copy=True)
        if filtered.ndim != 2:
            return filtered

        for i in range(filtered.shape[0]):
            current_delta = filtered[i]

            if filtered_residual_delta_prev is None:
                filtered_residual_delta_prev = np.array(
                    current_delta,
                    dtype=np.float32,
                    copy=True,
                )
            elif filtered_action_total_steps >= filter_warmup_steps:
                smoothed = (
                    float(filter_alpha) * current_delta
                    + (1.0 - float(filter_alpha)) * filtered_residual_delta_prev
                )
                if filter_max_delta is not None:
                    max_delta = float(filter_max_delta)
                    delta = smoothed - filtered_residual_delta_prev
                    smoothed = filtered_residual_delta_prev + np.clip(
                        delta,
                        -max_delta,
                        max_delta,
                    )
                filtered_residual_delta_prev = np.asarray(smoothed, dtype=np.float32)
            else:
                filtered_residual_delta_prev = np.array(
                    current_delta,
                    dtype=np.float32,
                    copy=True,
                )

            filtered[i] = filtered_residual_delta_prev
            filtered_action_total_steps += 1

        return filtered

    try:
        for episode_id in range(episodes):
            if filter_reset_each_episode:
                filtered_residual_delta_prev = None
            obs = env.reset()
            task_prompt = str(env.task_description)
            if video_recorder is not None:
                video_recorder.start_episode(int(episode_id))
                video_recorder.add_obs_frame(obs)
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

                final_action_array = np.asarray(final_actions, dtype=np.float32)
                action_chunk = final_action_array[:execute_horizon]
                base_action_chunk = np.asarray(base_actions, dtype=np.float32)
                if base_action_chunk.shape != final_action_array.shape:
                    if int(base_action_chunk.size) != int(final_action_array.size):
                        raise ValueError(
                            "base and final action chunk shape mismatch: "
                            f"base={base_action_chunk.shape} "
                            f"final={final_action_array.shape}"
                        )
                    base_action_chunk = base_action_chunk.reshape(
                        final_action_array.shape
                    )
                base_action_chunk = base_action_chunk[:execute_horizon]

                residual_delta_chunk = action_chunk - base_action_chunk
                residual_delta_chunk = _filter_residual_delta_chunk(
                    residual_delta_chunk
                )
                action_chunk = base_action_chunk + residual_delta_chunk

                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)
                if video_recorder is not None:
                    for post_step_obs in chunk_result.get("observations", ()):
                        if isinstance(post_step_obs, dict):
                            video_recorder.add_obs_frame(post_step_obs)

                chunk_infos = [dict(v) for v in chunk_result["infos"]]
                executed_steps = count_executed_steps_from_infos(chunk_infos)
                last_info = dict(chunk_result["info"])
                obs = dict(chunk_result["obs"])
                episode_return += float(chunk_result["reward_sum"])
                episode_success = bool(
                    episode_success
                    or last_info.get("success", False)
                    or any(bool(info.get("success", False)) for info in chunk_infos)
                )

                if executed_steps <= 0:
                    done_flag = bool(chunk_result["done"] or chunk_result["truncated"])
                    if not done_flag:
                        raise RuntimeError(
                            "step_chunk returned no executed actions without a "
                            "terminal outcome"
                        )
                    env_episode_done = True
                    timer.tock("total")
                    break

                episode_steps += int(executed_steps)
                total_env_steps += int(executed_steps)

                if bool(chunk_result["done"] or chunk_result["truncated"]):
                    env_episode_done = True

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
            if bool(cfg.video.enabled):
                episode_record["video_path"] = str(
                    video_output_dir
                    / (
                        f"episode_{int(episode_id):05d}_"
                        f"{str(cfg.video.camera_key).replace('/', '_')}.mp4"
                    )
                )
            episode_logger.write(to_jsonable(episode_record))
            if video_recorder is not None:
                video_recorder.end_episode(
                    episode_id=int(episode_id),
                    success=bool(episode_success),
                    episode_steps=int(episode_steps),
                )

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
        try:
            if video_recorder is not None:
                video_recorder.close()
        except Exception:  # noqa: BLE001
            pass

    return summary
