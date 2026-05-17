from __future__ import annotations

"""Helpers for reporting training progress and episodic eval results."""

from collections.abc import Iterable
from collections.abc import Mapping
import logging
from typing import Any

from serl_launcher.utils.serialization import to_jsonable


def sync_eval_results_to_wandb(
    *,
    records: Iterable[Mapping[str, Any]],
    wandb_logger: Any,
    logger: logging.Logger,
    eval_queue_backlog: int | None = None,
) -> None:
    """Log completed episodic eval results onto the dedicated eval axis."""

    for payload in records:
        status = str(payload.get("status", "")).strip().lower()
        train_update_step = payload.get("train_update_step", None)
        train_env_step = payload.get("train_env_step", None)
        train_episode_id = payload.get("train_episode_id", None)
        if not isinstance(train_episode_id, (int, float)):
            continue

        metrics: dict[str, float] = {
            "eval/train_episode_id": float(train_episode_id),
            "eval/status_failed": 0.0 if status == "ok" else 1.0,
        }
        eval_index = _maybe_float(payload, "eval_index")
        train_update_step_float = _maybe_float(payload, "train_update_step")
        train_env_step_float = _maybe_float(payload, "train_env_step")
        duration_sec = _maybe_float(payload, "duration_sec")
        if eval_index is not None:
            metrics["eval/eval_index"] = float(eval_index)
        if train_update_step_float is not None:
            metrics["eval/train_update_step"] = float(train_update_step_float)
        if train_env_step_float is not None:
            metrics["eval/train_env_step"] = float(train_env_step_float)
        if duration_sec is not None:
            metrics["eval/duration_sec"] = float(duration_sec)
        if eval_queue_backlog is not None:
            metrics["eval/queue_backlog"] = float(max(0, int(eval_queue_backlog)))

        summary = payload.get("summary", None)
        if status == "ok" and isinstance(summary, Mapping):
            success_rate = _maybe_float(summary, "success_rate")
            mean_return = _maybe_float(summary, "mean_return")
            if success_rate is not None:
                metrics["eval/success_rate"] = float(success_rate)
            if mean_return is not None:
                metrics["eval/mean_return"] = float(mean_return)
            for source_key, metric_key in (
                ("episodes_completed", "eval/episodes_completed"),
                ("env_steps", "eval/env_steps"),
                ("parallel_envs", "eval/parallel_envs"),
                ("policy_batch_size", "eval/policy_batch_size"),
                ("policy_requests", "eval/policy_requests"),
                ("policy_batch_requests", "eval/policy_batch_requests"),
                ("policy_samples", "eval/policy_samples"),
                ("policy_requests_per_env_step", "eval/policy_requests_per_env_step"),
                ("policy_samples_per_env_step", "eval/policy_samples_per_env_step"),
                ("mean_active_lanes", "eval/mean_active_lanes"),
            ):
                value = _maybe_float(summary, source_key)
                if value is not None:
                    metrics[metric_key] = float(value)
            timer_metrics = summary.get("timer", None)
            if isinstance(timer_metrics, Mapping):
                for source_key, metric_key in (
                    ("policy_batch_infer", "eval/policy_batch_infer_sec"),
                    ("policy_infer", "eval/policy_infer_sec"),
                    ("reset_env", "eval/reset_env_sec"),
                    ("step_env", "eval/step_env_sec"),
                ):
                    value = _maybe_float(timer_metrics, source_key)
                    if value is not None:
                        metrics[metric_key] = float(value)

        wandb_logger.log(to_jsonable(metrics), step=int(train_episode_id))
        if isinstance(train_env_step, (int, float)):
            env_step_metrics = _build_eval_env_step_metrics(metrics)
            if env_step_metrics:
                wandb_logger.log(
                    to_jsonable(env_step_metrics),
                    step=int(train_env_step),
                )

        if status == "ok":
            logger.info(
                "eval done: eval_index=%s episode=%s update_steps=%s env_steps=%s success_rate=%s",
                payload.get("eval_index", None),
                payload.get("train_episode_id", None),
                train_update_step,
                train_env_step,
                summary.get("success_rate", None)
                if isinstance(summary, Mapping)
                else None,
            )
        else:
            logger.warning(
                "eval failed: eval_index=%s episode=%s update_steps=%s error=%s",
                payload.get("eval_index", None),
                payload.get("train_episode_id", None),
                train_update_step,
                payload.get("error", None),
            )


def format_learner_heartbeat(
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
    updates_per_sec: float,
    update_info: Mapping[str, Any],
    offline_suffix: str = "",
) -> str:
    """Format the learner heartbeat line for logs."""

    return (
        "learner heartbeat: "
        f"update_steps={int(update_steps)} "
        f"env_steps={int(env_steps)} "
        f"replay_size={int(replay_size)} "
        f"updates_per_sec={updates_per_sec:.2f} "
        f"critic_loss={_format_optional_metric(_maybe_float(update_info, 'critic_loss'))} "
        f"critic_td_loss={_format_optional_metric(_maybe_float(update_info, 'critic_td_loss'))} "
        f"actor_loss={_format_optional_metric(_maybe_float(update_info, 'actor_loss'))} "
        f"temperature={_format_optional_metric(_maybe_float(update_info, 'temperature'))} "
        f"entropy={_format_optional_metric(_maybe_float(update_info, 'entropy'))} "
        f"predicted_qs={_format_optional_metric(_maybe_float(update_info, 'predicted_qs'))} "
        f"target_qs={_format_optional_metric(_maybe_float(update_info, 'target_qs'))} "
        f"actor_predicted_q={_format_optional_metric(_maybe_float(update_info, 'actor_predicted_q'))}"
        f"{offline_suffix}"
    )


def _maybe_float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_eval_env_step_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    mapping = (
        ("eval/status_failed", "eval_by_env_steps/status_failed"),
        ("eval/success_rate", "eval_by_env_steps/success_rate"),
        ("eval/mean_return", "eval_by_env_steps/mean_return"),
        ("eval/eval_index", "eval_by_env_steps/eval_index"),
        ("eval/train_episode_id", "eval_by_env_steps/train_episode_id"),
        ("eval/train_update_step", "eval_by_env_steps/train_update_step"),
        ("eval/train_env_step", "eval_by_env_steps/train_env_step"),
        ("eval/duration_sec", "eval_by_env_steps/duration_sec"),
        ("eval/queue_backlog", "eval_by_env_steps/queue_backlog"),
        ("eval/episodes_completed", "eval_by_env_steps/episodes_completed"),
        ("eval/env_steps", "eval_by_env_steps/env_steps"),
        ("eval/parallel_envs", "eval_by_env_steps/parallel_envs"),
        ("eval/policy_batch_size", "eval_by_env_steps/policy_batch_size"),
        ("eval/policy_requests", "eval_by_env_steps/policy_requests"),
        ("eval/policy_batch_requests", "eval_by_env_steps/policy_batch_requests"),
        ("eval/policy_samples", "eval_by_env_steps/policy_samples"),
        (
            "eval/policy_requests_per_env_step",
            "eval_by_env_steps/policy_requests_per_env_step",
        ),
        (
            "eval/policy_samples_per_env_step",
            "eval_by_env_steps/policy_samples_per_env_step",
        ),
        ("eval/mean_active_lanes", "eval_by_env_steps/mean_active_lanes"),
        ("eval/policy_batch_infer_sec", "eval_by_env_steps/policy_batch_infer_sec"),
        ("eval/policy_infer_sec", "eval_by_env_steps/policy_infer_sec"),
        ("eval/reset_env_sec", "eval_by_env_steps/reset_env_sec"),
        ("eval/step_env_sec", "eval_by_env_steps/step_env_sec"),
    )
    env_step_metrics: dict[str, float] = {}
    for source_key, metric_key in mapping:
        value = metrics.get(source_key, None)
        if isinstance(value, (int, float)):
            env_step_metrics[metric_key] = float(value)
    return env_step_metrics


def _format_optional_metric(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"
