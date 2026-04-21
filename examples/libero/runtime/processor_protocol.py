from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .transition_assembly import BatchAwareLiberoTransitionAssembler
from .transition_assembly import ChunkExecutionRecord
from .transition_assembly import PrefetchedDecisionObs


@dataclass(frozen=True)
class NormalizedChunkResult:
    steps: list[dict[str, Any]]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict[str, Any]]
    post_step_observations: list[dict[str, Any]]
    executed_steps: int
    reward_sum: float
    final_obs: dict[str, Any]
    chunk_done: bool
    chunk_truncated: bool
    chunk_info: dict[str, Any]


@dataclass(frozen=True)
class ActorRolloutChunkSummary:
    executed_steps: int
    reward_sum: float
    final_obs: dict[str, Any]
    chunk_done: bool
    chunk_truncated: bool
    chunk_info: dict[str, Any]
    episode_success: bool


def build_processor_submission_payload(
    *,
    chunk_seq: int,
    episode_id: int,
    episode_step_start: int,
    task_prompt: str,
    chunk_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunk_seq": int(chunk_seq),
        "episode_id": int(episode_id),
        "episode_step_start": int(episode_step_start),
        "task_prompt": str(task_prompt),
        "chunk_result": dict(chunk_result),
    }


def extract_actor_rollout_chunk_summary(
    chunk_result: dict[str, Any],
) -> ActorRolloutChunkSummary:
    normalized_chunk = normalize_chunk_result(dict(chunk_result))
    chunk_info = dict(normalized_chunk.chunk_info)
    episode_success = bool(chunk_info.get("env_done", False))
    if not episode_success:
        episode_success = any(
            bool(info.get("env_done", False)) for info in normalized_chunk.infos
        )

    return ActorRolloutChunkSummary(
        executed_steps=int(normalized_chunk.executed_steps),
        reward_sum=float(normalized_chunk.reward_sum),
        final_obs=dict(normalized_chunk.final_obs),
        chunk_done=bool(normalized_chunk.chunk_done),
        chunk_truncated=bool(normalized_chunk.chunk_truncated),
        chunk_info=chunk_info,
        episode_success=bool(episode_success),
    )


def normalize_chunk_result(chunk_result: dict[str, Any]) -> NormalizedChunkResult:
    steps = [dict(step) for step in list(chunk_result.get("steps", ()))]
    if not steps:
        raise ValueError("received empty chunk_result.steps")

    rewards = [float(dict(step)["reward"]) for step in steps]
    dones = [bool(dict(step)["done"]) for step in steps]
    infos = [dict(dict(step)["info"]) for step in steps]
    post_step_observations = [dict(dict(step)["next_obs"]) for step in steps]
    last_step = dict(steps[-1])

    executed_steps = int(len(steps))
    declared_steps = chunk_result.get("num_steps", None)
    if declared_steps is not None and int(declared_steps) != int(executed_steps):
        raise ValueError(
            "chunk_result.num_steps does not match chunk_result.steps: "
            f"num_steps={int(declared_steps)} steps={int(executed_steps)}"
        )

    reward_sum = float(sum(rewards))
    declared_reward_sum = chunk_result.get("reward_sum", None)
    if declared_reward_sum is not None and not bool(
        np.isclose(float(declared_reward_sum), float(reward_sum))
    ):
        raise ValueError(
            "chunk_result.reward_sum does not match summed step rewards: "
            f"reward_sum={float(declared_reward_sum)} summed={float(reward_sum)}"
        )

    chunk_done = bool(last_step["done"])
    declared_done = chunk_result.get("done", None)
    if declared_done is not None and bool(declared_done) != bool(chunk_done):
        raise ValueError(
            "chunk_result.done does not match final step done flag: "
            f"done={bool(declared_done)} step_done={bool(chunk_done)}"
        )

    chunk_truncated = bool(
        chunk_result.get("truncated", last_step.get("truncated", False))
    )
    if "truncated" in last_step and bool(last_step["truncated"]) != bool(
        chunk_truncated
    ):
        raise ValueError(
            "chunk_result.truncated does not match final step truncated flag: "
            f"truncated={bool(chunk_truncated)} "
            f"step_truncated={bool(last_step['truncated'])}"
        )

    final_obs = dict(chunk_result.get("obs", last_step["next_obs"]))
    if ("obs" in chunk_result) and (
        not _structures_equal(chunk_result["obs"], last_step["next_obs"])
    ):
        raise ValueError("chunk_result.obs does not match final step next_obs")

    return NormalizedChunkResult(
        steps=steps,
        rewards=rewards,
        dones=dones,
        infos=infos,
        post_step_observations=post_step_observations,
        executed_steps=int(executed_steps),
        reward_sum=float(reward_sum),
        final_obs=final_obs,
        chunk_done=bool(chunk_done),
        chunk_truncated=bool(chunk_truncated),
        chunk_info=dict(chunk_result.get("info", last_step["info"])),
    )


def reconstruct_chunk_execution_record_from_normalized(
    *,
    payload: dict[str, Any],
    normalized_chunk: NormalizedChunkResult,
    assembler: BatchAwareLiberoTransitionAssembler,
) -> ChunkExecutionRecord:
    steps = list(normalized_chunk.steps)
    if not steps:
        raise ValueError("processor received empty normalized chunk")

    start_obs = dict(dict(steps[0])["obs"])
    task_prompt = str(payload["task_prompt"])
    decision_obs: PrefetchedDecisionObs = assembler.infer_decision_obs(
        obs=start_obs,
        task_prompt=task_prompt,
    )
    action_chunk = np.stack(
        [
            np.asarray(dict(step)["action"], dtype=np.float32).reshape(-1)
            for step in steps
        ],
        axis=0,
    )
    return ChunkExecutionRecord(
        episode_id=int(payload["episode_id"]),
        episode_step_start=int(payload["episode_step_start"]),
        residual_obs_before_chunk=decision_obs.residual_obs,
        action_chunk=action_chunk,
        post_step_observations=list(normalized_chunk.post_step_observations),
        rewards=list(normalized_chunk.rewards),
        dones=list(normalized_chunk.dones),
        infos=list(normalized_chunk.infos),
        final_obs=dict(normalized_chunk.final_obs),
        chunk_done=bool(normalized_chunk.chunk_done),
        chunk_truncated=bool(normalized_chunk.chunk_truncated),
        reward_sum=float(normalized_chunk.reward_sum),
        chunk_info=dict(normalized_chunk.chunk_info),
        executed_steps=int(normalized_chunk.executed_steps),
    )


def reconstruct_chunk_execution_record(
    *,
    payload: dict[str, Any],
    assembler: BatchAwareLiberoTransitionAssembler,
) -> ChunkExecutionRecord:
    normalized_chunk = normalize_chunk_result(dict(payload["chunk_result"]))
    return reconstruct_chunk_execution_record_from_normalized(
        payload=payload,
        normalized_chunk=normalized_chunk,
        assembler=assembler,
    )


def _structures_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except Exception:  # noqa: BLE001
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        if left_keys != right_keys:
            return False
        return all(_structures_equal(left[key], right[key]) for key in left_keys)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_structures_equal(lv, rv) for lv, rv in zip(left, right))
    return bool(left == right)


__all__ = [
    "ActorRolloutChunkSummary",
    "NormalizedChunkResult",
    "build_processor_submission_payload",
    "extract_actor_rollout_chunk_summary",
    "normalize_chunk_result",
    "reconstruct_chunk_execution_record",
    "reconstruct_chunk_execution_record_from_normalized",
]
