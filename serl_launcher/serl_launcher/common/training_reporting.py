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
) -> None:
    """Log completed episodic eval results onto the dedicated eval axis."""

    for payload in records:
        status = str(payload.get("status", "")).strip().lower()
        train_update_step = payload.get("train_update_step", None)
        train_episode_id = payload.get("train_episode_id", None)
        if not isinstance(train_episode_id, (int, float)):
            continue

        metrics: dict[str, float] = {
            "eval/train_episode_id": float(train_episode_id),
            "eval/status_failed": 0.0 if status == "ok" else 1.0,
        }

        summary = payload.get("summary", None)
        if status == "ok" and isinstance(summary, Mapping):
            success_rate = _maybe_float(summary, "success_rate")
            mean_return = _maybe_float(summary, "mean_return")
            if success_rate is not None:
                metrics["eval/success_rate"] = float(success_rate)
            if mean_return is not None:
                metrics["eval/mean_return"] = float(mean_return)

        wandb_logger.log(to_jsonable(metrics))

        if status == "ok":
            logger.info(
                "eval done: eval_index=%s episode=%s update_steps=%s env_steps=%s success_rate=%s",
                payload.get("eval_index", None),
                payload.get("train_episode_id", None),
                train_update_step,
                payload.get("train_env_step", None),
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


def _format_optional_metric(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"
