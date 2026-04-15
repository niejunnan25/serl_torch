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
)

DEFAULT_ROLLOUT_AXIS_METRIC = "rollout/episode_id"
DEFAULT_ROLLOUT_METRICS = (
    "rollout/episode_return",
    "rollout/cumulative_success_rate",
    "rollout/recent_success_rate_20",
)
DEFAULT_ROLLOUT_METRIC_MAPPING = (
    ("episode_id", DEFAULT_ROLLOUT_AXIS_METRIC),
    ("episode_return", "rollout/episode_return"),
    ("cumulative_success_rate", "rollout/cumulative_success_rate"),
    ("recent_success_rate_20", "rollout/recent_success_rate_20"),
)

DEFAULT_LEARNER_METRICS = (
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
    return extract_numeric_metrics(rollout, mapping)


def extract_learner_wandb_metrics(
    update_info: Mapping[str, Any],
    *,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] = DEFAULT_LEARNER_METRIC_MAPPING,
) -> dict[str, float]:
    """Extract standard learner update metrics from an agent update payload."""

    return extract_numeric_metrics(update_info, mapping)


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
