from __future__ import annotations

"""Default observability wiring for RL training runs."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from serl_launcher.common.observability import define_metric_group
from serl_launcher.common.observability import extract_numeric_metrics

DEFAULT_EVAL_AXIS_METRIC = "eval/train_episode_id"
DEFAULT_EVAL_METRICS = (
    "eval/status_failed",
    "eval/success_rate",
    "eval/mean_return",
    "eval/eval_index",
    "eval/train_update_step",
    "eval/train_env_step",
    "eval/duration_sec",
    "eval/queue_backlog",
    "eval/episodes_completed",
    "eval/env_steps",
    "eval/parallel_envs",
    "eval/policy_batch_size",
    "eval/policy_requests",
    "eval/policy_batch_requests",
    "eval/policy_samples",
    "eval/policy_requests_per_env_step",
    "eval/policy_samples_per_env_step",
    "eval/mean_active_lanes",
    "eval/policy_batch_infer_sec",
    "eval/policy_infer_sec",
    "eval/reset_env_sec",
    "eval/step_env_sec",
)

DEFAULT_ROLLOUT_AXIS_METRIC = "rollout/episode_id"
DEFAULT_ROLLOUT_METRICS = (
    "rollout/episode_return",
    "rollout/cumulative_success_rate",
    "rollout/recent_success_rate_20",
    "rollout_progress/env_steps",
    "speed/actor_env_steps_per_sec",
    "residual/mean_abs",
    "residual/max_abs",
    "residual/std",
    "residual/saturation_rate",
    "action/mean_abs_delta_from_base",
    "action/max_abs_delta_from_base",
    "reward_model/bonus_sum",
    "reward_model/triggered",
    "reward_model/trigger_step",
    "reward_model/max_score",
    "reward_model/scored_steps",
    "reward_model/shaped_episode_return",
)
DEFAULT_ROLLOUT_METRIC_MAPPING = (
    ("episode_id", DEFAULT_ROLLOUT_AXIS_METRIC),
    ("episode_return", "rollout/episode_return"),
    ("cumulative_success_rate", "rollout/cumulative_success_rate"),
    ("recent_success_rate_20", "rollout/recent_success_rate_20"),
)
ROLLOUT_ENV_STEP_METRIC_MAPPING = (
    ("rollout/episode_id", "rollout_by_env_steps/episode_id"),
    ("rollout/episode_return", "rollout_by_env_steps/episode_return"),
    (
        "rollout/cumulative_success_rate",
        "rollout_by_env_steps/cumulative_success_rate",
    ),
    (
        "rollout/recent_success_rate_20",
        "rollout_by_env_steps/recent_success_rate_20",
    ),
    ("rollout_progress/env_steps", "rollout_by_env_steps/env_steps"),
    ("speed/actor_env_steps_per_sec", "speed_by_env_steps/actor_env_steps_per_sec"),
    ("residual/mean_abs", "residual_by_env_steps/mean_abs"),
    ("residual/max_abs", "residual_by_env_steps/max_abs"),
    ("residual/std", "residual_by_env_steps/std"),
    ("residual/saturation_rate", "residual_by_env_steps/saturation_rate"),
    (
        "action/mean_abs_delta_from_base",
        "action_by_env_steps/mean_abs_delta_from_base",
    ),
    (
        "action/max_abs_delta_from_base",
        "action_by_env_steps/max_abs_delta_from_base",
    ),
    ("reward_model/bonus_sum", "reward_model_by_env_steps/bonus_sum"),
    ("reward_model/triggered", "reward_model_by_env_steps/triggered"),
    ("reward_model/trigger_step", "reward_model_by_env_steps/trigger_step"),
    ("reward_model/max_score", "reward_model_by_env_steps/max_score"),
    ("reward_model/scored_steps", "reward_model_by_env_steps/scored_steps"),
    (
        "reward_model/shaped_episode_return",
        "reward_model_by_env_steps/shaped_episode_return",
    ),
)

DEFAULT_LEARNER_METRICS = (
    "progress/env_steps",
    "progress/update_steps",
    "progress/replay_size",
    "speed/learner_updates_per_sec",
    "eval_queue/backlog",
    "replay/online_size",
    "replay/offline_size",
    "replay/sample_online_count",
    "replay/sample_offline_count",
    "replay/offline_ratio_actual",
    "learner_time/sample_replay_buffer_sec",
    "learner_time/train_sec",
    "learner_time/train_critics_sec",
    "learner_time/publish_snapshot_sec",
    "learner_time/publish_network_sec",
    "learner/loss_critic",
    "learner/loss_critic_td",
    "learner/loss_actor",
    "learner/q_predicted_mean",
    "learner/q_target_mean",
    "learner/q_actor_predicted_mean",
    "learner/q_predicted_gap",
    "learner/temperature",
    "learner/entropy",
)
DEFAULT_LEARNER_METRIC_MAPPING = (
    ("critic_loss", "learner/loss_critic"),
    ("critic_td_loss", "learner/loss_critic_td"),
    ("actor_loss", "learner/loss_actor"),
    ("predicted_qs", "learner/q_predicted_mean"),
    ("target_qs", "learner/q_target_mean"),
    ("actor_predicted_q", "learner/q_actor_predicted_mean"),
    ("predicted_q_gap", "learner/q_predicted_gap"),
    ("temperature", "learner/temperature"),
    ("entropy", "learner/entropy"),
)


def configure_eval_wandb_metrics(
    *,
    wandb_logger: Any,
    axis_metric: str = DEFAULT_EVAL_AXIS_METRIC,
    metric_names: Sequence[str] = DEFAULT_EVAL_METRICS,
) -> None:
    """Register the default episodic eval metrics on a dedicated episode axis."""

    _configure_wandb_metric_group(
        wandb_logger=wandb_logger,
        axis_metric=axis_metric,
        metric_names=metric_names,
    )


def configure_rollout_wandb_metrics(
    *,
    wandb_logger: Any,
    axis_metric: str = DEFAULT_ROLLOUT_AXIS_METRIC,
    metric_names: Sequence[str] = DEFAULT_ROLLOUT_METRICS,
) -> None:
    """Register the default rollout metrics on a dedicated episode axis."""

    _configure_wandb_metric_group(
        wandb_logger=wandb_logger,
        axis_metric=axis_metric,
        metric_names=metric_names,
        hide_axis_metric=True,
    )


def configure_learner_wandb_metrics(
    *,
    wandb_logger: Any,
    metric_names: Sequence[str] = DEFAULT_LEARNER_METRICS,
) -> None:
    """Register the default learner update metrics on W&B's default step axis."""

    run = getattr(wandb_logger, "run", None)
    if run is None:
        return
    define_metric = getattr(run, "define_metric", None)
    if define_metric is None:
        return
    for metric_name in metric_names:
        define_metric(metric_name)


def extract_rollout_wandb_metrics(
    payload: Mapping[str, Any],
    *,
    rollout_key: str = "rollout",
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] = DEFAULT_ROLLOUT_METRIC_MAPPING,
) -> dict[str, float]:
    """Extract standard rollout metrics from a structured rollout payload."""

    rollout = payload.get(rollout_key, None)
    if not isinstance(rollout, Mapping):
        return {}
    metrics = extract_numeric_metrics(rollout, mapping)

    env_steps = payload.get("env_steps", None)
    if isinstance(env_steps, (int, float)):
        metrics["rollout_progress/env_steps"] = float(env_steps)

    residual = payload.get("residual", None)
    if isinstance(residual, Mapping):
        metrics.update(
            extract_numeric_metrics(
                residual,
                (
                    ("mean_abs", "residual/mean_abs"),
                    ("max_abs", "residual/max_abs"),
                    ("std", "residual/std"),
                    ("saturation_rate", "residual/saturation_rate"),
                    ("action_delta_mean_abs", "action/mean_abs_delta_from_base"),
                    ("action_delta_max_abs", "action/max_abs_delta_from_base"),
                ),
            )
        )
    reward_model = payload.get("reward_model", None)
    if isinstance(reward_model, Mapping):
        metrics.update(
            extract_numeric_metrics(
                reward_model,
                (
                    ("bonus_sum", "reward_model/bonus_sum"),
                    ("triggered", "reward_model/triggered"),
                    ("trigger_step", "reward_model/trigger_step"),
                    ("max_score", "reward_model/max_score"),
                    ("scored_steps", "reward_model/scored_steps"),
                    (
                        "shaped_episode_return",
                        "reward_model/shaped_episode_return",
                    ),
                ),
            )
        )
    return metrics


def extract_learner_wandb_metrics(
    update_info: Mapping[str, Any],
    *,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] = DEFAULT_LEARNER_METRIC_MAPPING,
) -> dict[str, float]:
    """Extract standard learner update metrics from an agent update payload."""

    return extract_numeric_metrics(update_info, mapping)


def build_rollout_env_step_wandb_metrics(
    rollout_metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Mirror rollout metrics into an env-step-indexed namespace."""

    return extract_numeric_metrics(rollout_metrics, ROLLOUT_ENV_STEP_METRIC_MAPPING)


def build_learner_runtime_wandb_metrics(
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
    updates_per_sec: float,
    eval_queue_backlog: int | None = None,
    offline_replay_size: int | None = None,
    batch_mix: Mapping[str, Any] | None = None,
    timer_metrics: Mapping[str, Any] | None = None,
    sample_profile_metrics: Mapping[str, Any] | None = None,
    update_profile_metrics: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build scalar learner runtime metrics for experiment dashboards."""

    metrics: dict[str, float] = {
        "progress/env_steps": float(env_steps),
        "progress/update_steps": float(update_steps),
        "progress/replay_size": float(replay_size),
        "speed/learner_updates_per_sec": float(updates_per_sec),
        "replay/online_size": float(replay_size),
    }
    if eval_queue_backlog is not None:
        metrics["eval_queue/backlog"] = float(max(0, int(eval_queue_backlog)))
    if offline_replay_size is not None:
        metrics["replay/offline_size"] = float(max(0, int(offline_replay_size)))

    if batch_mix is not None:
        online_batch_size = _numeric(batch_mix.get("online_batch_size", None))
        offline_batch_size = _numeric(batch_mix.get("offline_batch_size", None))
        if online_batch_size is not None:
            metrics["replay/sample_online_count"] = float(online_batch_size)
        if offline_batch_size is not None:
            metrics["replay/sample_offline_count"] = float(offline_batch_size)
        if online_batch_size is not None and offline_batch_size is not None:
            total = float(online_batch_size + offline_batch_size)
            if total > 0:
                metrics["replay/offline_ratio_actual"] = float(
                    offline_batch_size / total
                )

    if timer_metrics is not None:
        for source_key, metric_key in (
            ("sample_replay_buffer", "learner_time/sample_replay_buffer_sec"),
            ("train", "learner_time/train_sec"),
            ("train_critics", "learner_time/train_critics_sec"),
            ("publish_snapshot", "learner_time/publish_snapshot_sec"),
            ("publish_network", "learner_time/publish_network_sec"),
        ):
            value = _numeric(timer_metrics.get(source_key, None))
            if value is not None:
                metrics[metric_key] = float(value)

    if sample_profile_metrics is not None:
        for key, value in sample_profile_metrics.items():
            numeric_value = _numeric(value)
            if numeric_value is not None:
                metrics[f"learner_time/sample_profile/{key}"] = float(numeric_value)

    if update_profile_metrics is not None:
        for key, value in update_profile_metrics.items():
            numeric_value = _numeric(value)
            if numeric_value is not None:
                metrics[f"learner_time/update_profile/{key}"] = float(numeric_value)

    return metrics


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _configure_wandb_metric_group(
    *,
    wandb_logger: Any,
    axis_metric: str,
    metric_names: Sequence[str],
    hide_axis_metric: bool = False,
) -> None:
    run = getattr(wandb_logger, "run", None)
    if run is None:
        return
    define_metric_group(
        run,
        axis_metric=axis_metric,
        metric_names=metric_names,
        hide_axis_metric=hide_axis_metric,
    )
