from __future__ import annotations

"""Structured payload helpers for actor/learner training messages."""

from collections.abc import Mapping
from typing import Any
from typing import TypedDict

from serl_launcher.utils.serialization import to_jsonable


class _RequiredRolloutPayload(TypedDict):
    episode_id: int
    episode_steps: int
    episode_return: float
    success: bool
    cumulative_success_rate: float
    recent_success_rate_20: float


class RolloutPayload(_RequiredRolloutPayload, total=False):
    init_episode_idx: int


class RolloutStatsPayload(TypedDict):
    env_steps: int
    rollout: RolloutPayload
    env_info: dict[str, Any]
    residual: dict[str, Any]


class ActorProgressPayload(TypedDict):
    env_steps: int
    episode_id: int
    actor_done: bool


def build_rollout_payload(
    *,
    episode_id: int,
    episode_steps: int,
    episode_return: float,
    init_episode_idx: int | None = None,
    success: bool,
    cumulative_success_rate: float,
    recent_success_rate_20: float,
) -> RolloutPayload:
    """Build a normalized rollout summary payload."""

    payload: RolloutPayload = {
        "episode_id": int(episode_id),
        "episode_steps": int(episode_steps),
        "episode_return": float(episode_return),
        "success": bool(success),
        "cumulative_success_rate": float(cumulative_success_rate),
        "recent_success_rate_20": float(recent_success_rate_20),
    }
    if init_episode_idx is not None:
        payload["init_episode_idx"] = int(init_episode_idx)
    return payload


def build_rollout_stats_payload(
    *,
    env_steps: int,
    rollout: RolloutPayload,
    env_info: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
) -> RolloutStatsPayload:
    """Build a transport-safe actor->learner rollout stats payload."""

    env_info_payload: dict[str, Any] = {}
    if env_info is not None:
        serialized_env_info = to_jsonable(dict(env_info))
        if isinstance(serialized_env_info, Mapping):
            env_info_payload = dict(serialized_env_info)

    residual_payload: dict[str, Any] = {}
    if residual is not None:
        serialized_residual = to_jsonable(dict(residual))
        if isinstance(serialized_residual, Mapping):
            residual_payload = dict(serialized_residual)

    return {
        "env_steps": int(env_steps),
        "rollout": dict(rollout),
        "env_info": env_info_payload,
        "residual": residual_payload,
    }


def build_actor_progress_payload(
    *,
    env_steps: int,
    episode_id: int,
    actor_done: bool,
) -> ActorProgressPayload:
    """Build a normalized actor progress payload for coordinated shutdown."""

    return {
        "env_steps": int(env_steps),
        "episode_id": int(episode_id),
        "actor_done": bool(actor_done),
    }


def parse_rollout_stats_payload(
    payload: Mapping[str, Any],
) -> RolloutStatsPayload | None:
    """Parse and normalize a rollout stats payload from transport."""

    env_steps = _maybe_int(payload.get("env_steps", None))
    rollout_raw = payload.get("rollout", None)
    if env_steps is None or not isinstance(rollout_raw, Mapping):
        return None

    rollout = _parse_rollout_payload(rollout_raw)
    if rollout is None:
        return None

    env_info_payload: dict[str, Any] = {}
    env_info = payload.get("env_info", None)
    if isinstance(env_info, Mapping):
        env_info_payload = dict(env_info)

    residual_payload: dict[str, Any] = {}
    residual = payload.get("residual", None)
    if isinstance(residual, Mapping):
        residual_payload = dict(residual)

    return {
        "env_steps": int(env_steps),
        "rollout": rollout,
        "env_info": env_info_payload,
        "residual": residual_payload,
    }


def parse_actor_progress_payload(
    payload: Mapping[str, Any],
) -> ActorProgressPayload | None:
    """Parse and normalize an actor progress payload from transport."""

    env_steps = _maybe_int(payload.get("env_steps", None))
    episode_id = _maybe_int(payload.get("episode_id", None))
    actor_done = _maybe_bool(payload.get("actor_done", None))
    if env_steps is None or episode_id is None or actor_done is None:
        return None
    return {
        "env_steps": int(env_steps),
        "episode_id": int(episode_id),
        "actor_done": bool(actor_done),
    }


def _parse_rollout_payload(payload: Mapping[str, Any]) -> RolloutPayload | None:
    episode_id = _maybe_int(payload.get("episode_id", None))
    episode_steps = _maybe_int(payload.get("episode_steps", None))
    episode_return = _maybe_float(payload.get("episode_return", None))
    init_episode_idx = _maybe_int(payload.get("init_episode_idx", None))
    success = _maybe_bool(payload.get("success", None))
    cumulative_success_rate = _maybe_float(
        payload.get("cumulative_success_rate", None)
    )
    recent_success_rate_20 = _maybe_float(payload.get("recent_success_rate_20", None))

    if (
        episode_id is None
        or episode_steps is None
        or episode_return is None
        or success is None
        or cumulative_success_rate is None
        or recent_success_rate_20 is None
    ):
        return None

    rollout: RolloutPayload = {
        "episode_id": int(episode_id),
        "episode_steps": int(episode_steps),
        "episode_return": float(episode_return),
        "success": bool(success),
        "cumulative_success_rate": float(cumulative_success_rate),
        "recent_success_rate_20": float(recent_success_rate_20),
    }
    if init_episode_idx is not None:
        rollout["init_episode_idx"] = int(init_episode_idx)
    return rollout


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _maybe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    return None
