from __future__ import annotations

"""Post-hoc transition assembly helpers for AgiBot residual chunk rollout."""

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.rollout.async_transition_assembly import (
    AsyncTransitionAssemblyCoordinator,
)
from serl_torch.examples.agibot_real.env.base_policy import (
    build_agibot_base_policy,
)
from serl_torch.examples.agibot_real.env.observation import build_agibot_layout_state
from serl_torch.examples.agibot_real.env.observation import extract_agibot_residual_images


@dataclass(frozen=True)
class PrefetchedDecisionObs:
    base_actions: np.ndarray
    residual_obs: dict[str, np.ndarray]


def count_executed_steps_from_infos(infos: list[dict[str, Any]]) -> int:
    executed_steps = 0
    for info in infos:
        if not bool(info.get("controller_action_executed", True)):
            break
        executed_steps += 1
    return int(executed_steps)


@dataclass(frozen=True)
class RawChunkRecord:
    episode_id: int
    episode_step_start: int
    residual_obs_before_chunk: dict[str, np.ndarray]
    action_chunk: np.ndarray
    post_step_observations: list[dict[str, Any]]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict[str, Any]]
    final_obs: dict[str, Any]
    chunk_done: bool
    chunk_truncated: bool
    reward_sum: float
    chunk_info: dict[str, Any]
    executed_steps: int

    @classmethod
    def from_step_chunk_result(
        cls,
        *,
        episode_id: int,
        episode_step_start: int,
        residual_obs_before_chunk: dict[str, np.ndarray],
        action_chunk: np.ndarray,
        chunk_result: dict[str, Any],
    ) -> "RawChunkRecord":
        post_step_observations = list(chunk_result["observations"])
        rewards = [float(value) for value in chunk_result["rewards"]]
        dones = [bool(value) for value in chunk_result["dones"]]
        infos = [dict(value) for value in chunk_result["infos"]]
        executed_steps = count_executed_steps_from_infos(infos)
        if executed_steps <= 0:
            raise RuntimeError("step_chunk returned no executed controller actions")

        action_chunk_array = np.asarray(action_chunk, dtype=np.float32)
        if int(action_chunk_array.shape[0]) < executed_steps:
            raise ValueError(
                "action_chunk is shorter than executed_steps: "
                f"{action_chunk_array.shape[0]} < {executed_steps}"
            )

        chunk_field_lengths = {
            "observations": len(post_step_observations),
            "rewards": len(rewards),
            "dones": len(dones),
            "infos": len(infos),
        }
        short_fields = {
            key: value
            for key, value in chunk_field_lengths.items()
            if value < executed_steps
        }
        if short_fields:
            raise ValueError(
                "step_chunk result is shorter than executed_steps: "
                f"executed_steps={executed_steps} short_fields={short_fields}"
            )

        executed_rewards = rewards[:executed_steps]
        return cls(
            episode_id=int(episode_id),
            episode_step_start=int(episode_step_start),
            residual_obs_before_chunk=residual_obs_before_chunk,
            action_chunk=action_chunk_array[:executed_steps],
            post_step_observations=post_step_observations[:executed_steps],
            rewards=executed_rewards,
            dones=dones[:executed_steps],
            infos=infos[:executed_steps],
            final_obs=dict(chunk_result["obs"]),
            chunk_done=bool(chunk_result["done"]),
            chunk_truncated=bool(chunk_result["truncated"]),
            reward_sum=float(sum(executed_rewards)),
            chunk_info=dict(chunk_result["info"]),
            executed_steps=executed_steps,
        )


@dataclass(frozen=True)
class AssemblyResult:
    transitions: list[dict[str, Any]]
    prefetched: PrefetchedDecisionObs | None
    next_obs: dict[str, Any]
    episode_done: bool
    env_steps_delta: int
    episode_steps_delta: int
    episode_return_delta: float
    episode_success: bool
    last_info: dict[str, Any]


def infer_chunk_residual_obs(
    *,
    obs: dict[str, Any],
    task_prompt: str,
    base_policy: Any,
    image_keys: tuple[str, ...],
    residual_alpha: float,
    arm_layout: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base_actions, _infer_info = base_policy.infer(obs, prompt=task_prompt)
    base_actions = np.asarray(base_actions, dtype=np.float32)
    residual_obs = build_chunk_residual_obs(
        robot_state=build_agibot_layout_state(obs, arm_layout=arm_layout),
        images=extract_agibot_residual_images(
            obs,
            image_keys=image_keys,
        ),
        image_keys=image_keys,
        base_actions=base_actions,
        residual_alpha=residual_alpha,
    )
    return base_actions, residual_obs


def _supports_batched_backfill(base_policy: Any) -> bool:
    policy_infer_many = getattr(base_policy, "infer_many", None)
    client = getattr(base_policy, "client", None)
    client_infer_many = getattr(client, "infer_many", None)
    return bool(callable(policy_infer_many) and callable(client_infer_many))


class AgiBotTransitionAssembler:
    def __init__(
        self,
        *,
        cfg: Any,
        base_policy: Any,
        logger: logging.Logger,
    ) -> None:
        self.base_policy = base_policy
        self.image_keys = tuple(cfg.obs.image_keys)
        self.residual_alpha = float(cfg.residual.alpha)
        self.arm_layout = str(cfg.env.arm_layout)
        self._logger = logger
        self._prefetched: PrefetchedDecisionObs | None = None
        self._last_submitted_chunk_seq: int | None = None
        self._max_pending_chunks = int(cfg.backfill_policy.max_pending_chunks)
        self._async_assembly: AsyncTransitionAssemblyCoordinator[
            RawChunkRecord,
            dict[str, Any],
            dict[str, np.ndarray],
            AssemblyResult,
        ] | None = None
        if bool(cfg.backfill_policy.enabled):
            backfill_base_policy = build_agibot_base_policy(
                cfg,
                logger=logger,
                host=str(cfg.backfill_policy.host),
                port=int(cfg.backfill_policy.port),
            )
            close_fn = getattr(backfill_base_policy, "close", None)
            self._async_assembly = AsyncTransitionAssemblyCoordinator(
                backfill_fn=lambda observations, task_prompt: self._backfill_residual_observations(
                    observations=observations,
                    task_prompt=task_prompt,
                    base_policy=backfill_base_policy,
                ),
                build_result_fn=self._build_assembly_result,
                thread_name_prefix="agibot-transition-assembly",
                logger=logger,
                close_fn=close_fn if callable(close_fn) else None,
                close_error_message="ignored backfill base policy close error",
            )
            self._logger.info(
                "transition assembler: async_transition_assembly enabled mode=%s endpoint=%s:%s max_pending_chunks=%s",
                str(cfg.backfill_policy.mode),
                str(cfg.backfill_policy.host),
                int(cfg.backfill_policy.port),
                int(self._max_pending_chunks),
            )

    @property
    def async_transition_assembly_enabled(self) -> bool:
        return self._async_assembly is not None

    def infer_decision_obs(
        self,
        *,
        obs: dict[str, Any],
        task_prompt: str,
    ) -> PrefetchedDecisionObs:
        base_actions, residual_obs = infer_chunk_residual_obs(
            obs=obs,
            task_prompt=task_prompt,
            base_policy=self.base_policy,
            image_keys=self.image_keys,
            residual_alpha=self.residual_alpha,
            arm_layout=self.arm_layout,
        )
        return PrefetchedDecisionObs(
            base_actions=base_actions,
            residual_obs=residual_obs,
        )

    def pop_prefetched_decision_obs(self) -> PrefetchedDecisionObs | None:
        if self._async_assembly is not None:
            return None
        decision_obs = self._prefetched
        self._prefetched = None
        return decision_obs

    def backfill_post_step_residual_obs(
        self,
        *,
        observations: list[dict[str, Any]],
        task_prompt: str,
    ) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        return backfill_post_step_residual_obs(
            observations=observations,
            task_prompt=task_prompt,
            base_policy=self.base_policy,
            image_keys=self.image_keys,
            residual_alpha=self.residual_alpha,
            arm_layout=self.arm_layout,
        )

    def process_chunk(
        self,
        *,
        raw: RawChunkRecord,
        task_prompt: str,
    ) -> AssemblyResult:
        backfilled_base_actions, backfilled_residual_obs = (
            self.backfill_post_step_residual_obs(
                observations=raw.post_step_observations,
                task_prompt=task_prompt,
            )
        )
        transitions = assemble_chunk_step_transitions(
            episode_id=int(raw.episode_id),
            episode_step_start=int(raw.episode_step_start),
            residual_obs_before_chunk=raw.residual_obs_before_chunk,
            executed_actions=raw.action_chunk,
            rewards=raw.rewards,
            dones=raw.dones,
            infos=raw.infos,
            chunk_truncated=bool(raw.chunk_truncated),
            next_residual_observations=backfilled_residual_obs,
        )
        episode_done = bool(raw.chunk_done or raw.chunk_truncated)
        prefetched = None
        if not episode_done:
            prefetched = PrefetchedDecisionObs(
                base_actions=backfilled_base_actions[-1],
                residual_obs=backfilled_residual_obs[-1],
            )
        return AssemblyResult(
            transitions=transitions,
            prefetched=prefetched,
            next_obs=dict(raw.final_obs),
            episode_done=episode_done,
            env_steps_delta=int(raw.executed_steps),
            episode_steps_delta=int(raw.executed_steps),
            episode_return_delta=float(raw.reward_sum),
            episode_success=any(bool(info.get("success", False)) for info in raw.infos),
            last_info=dict(raw.chunk_info),
        )

    def _build_assembly_result(
        self,
        raw: RawChunkRecord,
        next_residual_observations: list[dict[str, np.ndarray]],
    ) -> AssemblyResult:
        transitions = assemble_chunk_step_transitions(
            episode_id=int(raw.episode_id),
            episode_step_start=int(raw.episode_step_start),
            residual_obs_before_chunk=raw.residual_obs_before_chunk,
            executed_actions=raw.action_chunk,
            rewards=raw.rewards,
            dones=raw.dones,
            infos=raw.infos,
            chunk_truncated=bool(raw.chunk_truncated),
            next_residual_observations=next_residual_observations,
        )
        return AssemblyResult(
            transitions=transitions,
            prefetched=None,
            next_obs=dict(raw.final_obs),
            episode_done=bool(raw.chunk_done or raw.chunk_truncated),
            env_steps_delta=int(raw.executed_steps),
            episode_steps_delta=int(raw.executed_steps),
            episode_return_delta=float(raw.reward_sum),
            episode_success=any(bool(info.get("success", False)) for info in raw.infos),
            last_info=dict(raw.chunk_info),
        )

    def _backfill_residual_observations(
        self,
        *,
        observations: list[dict[str, Any]],
        task_prompt: str,
        base_policy: Any,
    ) -> list[dict[str, np.ndarray]]:
        _base_action_chunks, next_residual_observations = (
            backfill_post_step_residual_obs(
                observations=observations,
                task_prompt=task_prompt,
                base_policy=base_policy,
                image_keys=self.image_keys,
                residual_alpha=self.residual_alpha,
                arm_layout=self.arm_layout,
            )
        )
        return next_residual_observations

    def drain_ready(self) -> list[AssemblyResult]:
        if self._async_assembly is None:
            return []
        return self._async_assembly.pop_committable()

    def handle_chunk(
        self,
        *,
        raw: RawChunkRecord,
        task_prompt: str,
    ) -> list[AssemblyResult]:
        if self._async_assembly is None:
            assembled_chunk = self.process_chunk(
                raw=raw,
                task_prompt=task_prompt,
            )
            self._prefetched = assembled_chunk.prefetched
            return [assembled_chunk]

        self._last_submitted_chunk_seq = self._async_assembly.submit_chunk(
            raw=raw,
            observations=raw.post_step_observations,
            task_prompt=task_prompt,
        )
        assembled_chunks = self._async_assembly.pop_committable()
        while self._async_assembly.pending_count > int(self._max_pending_chunks):
            assembled_chunks.extend(
                self._async_assembly.pop_committable(
                    block_until_seq=self._async_assembly.next_commit_chunk_seq
                )
            )
        return assembled_chunks

    def finish_episode(
        self,
        *,
        block: bool = True,
    ) -> list[AssemblyResult]:
        self._prefetched = None
        if self._async_assembly is None:
            return []
        if self._last_submitted_chunk_seq is None:
            return []
        if bool(block):
            assembled_chunks = self._async_assembly.pop_committable(
                block_until_seq=int(self._last_submitted_chunk_seq)
            )
            self._last_submitted_chunk_seq = None
            return assembled_chunks
        assembled_chunks = self._async_assembly.pop_committable()
        if self._async_assembly.pending_count <= 0:
            self._last_submitted_chunk_seq = None
        return assembled_chunks

    def close(self) -> None:
        if self._async_assembly is not None:
            self._async_assembly.close()


def backfill_post_step_residual_obs(
    *,
    observations: list[dict[str, Any]],
    task_prompt: str,
    base_policy: Any,
    image_keys: tuple[str, ...],
    residual_alpha: float,
    arm_layout: str,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    if not observations:
        return [], []
    if _supports_batched_backfill(base_policy):
        base_action_chunks, _batch_info = base_policy.infer_many(
            observations=observations,
            prompt=task_prompt,
        )
        residual_observations = [
            build_chunk_residual_obs(
                robot_state=build_agibot_layout_state(
                    post_step_obs,
                    arm_layout=arm_layout,
                ),
                images=extract_agibot_residual_images(
                    post_step_obs,
                    image_keys=image_keys,
                ),
                image_keys=image_keys,
                base_actions=base_actions,
                residual_alpha=residual_alpha,
            )
            for post_step_obs, base_actions in zip(
                observations,
                base_action_chunks,
                strict=True,
            )
        ]
        return base_action_chunks, residual_observations
    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    for post_step_obs in observations:
        next_base_actions, next_residual_obs = infer_chunk_residual_obs(
            obs=post_step_obs,
            task_prompt=task_prompt,
            base_policy=base_policy,
            image_keys=image_keys,
            residual_alpha=residual_alpha,
            arm_layout=arm_layout,
        )
        base_action_chunks.append(next_base_actions)
        residual_observations.append(next_residual_obs)
    return base_action_chunks, residual_observations


def assemble_chunk_step_transitions(
    *,
    episode_id: int,
    episode_step_start: int,
    residual_obs_before_chunk: dict[str, np.ndarray],
    executed_actions: np.ndarray,
    rewards: list[float],
    dones: list[bool],
    infos: list[dict[str, Any]],
    chunk_truncated: bool,
    next_residual_observations: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    executed_actions = np.asarray(executed_actions, dtype=np.float32)
    executed_steps = int(executed_actions.shape[0])
    expected_lengths = {
        "rewards": len(rewards),
        "dones": len(dones),
        "infos": len(infos),
        "next_residual_observations": len(next_residual_observations),
    }
    mismatched_lengths = {
        key: value for key, value in expected_lengths.items() if value != executed_steps
    }
    if mismatched_lengths:
        raise ValueError(
            "chunk transition fields must have the same length as executed_actions: "
            f"executed_actions={executed_steps} mismatched={mismatched_lengths}"
        )

    current_residual_obs = residual_obs_before_chunk
    transitions: list[dict[str, Any]] = []
    for step_idx, next_residual_obs in enumerate(next_residual_observations):
        done_flag = bool(dones[step_idx]) or bool(
            step_idx == (executed_steps - 1) and chunk_truncated
        )
        transitions.append(
            {
                "episode_id": int(episode_id),
                "episode_step": int(episode_step_start + step_idx),
                "observations": current_residual_obs,
                "actions": np.asarray(
                    executed_actions[step_idx],
                    dtype=np.float32,
                ).reshape(-1),
                "next_observations": next_residual_obs,
                "rewards": float(rewards[step_idx]),
                "masks": float(0.0 if done_flag else 1.0),
                "dones": done_flag,
            }
        )
        current_residual_obs = next_residual_obs
    return transitions


__all__ = [
    "AgiBotTransitionAssembler",
    "AssemblyResult",
    "PrefetchedDecisionObs",
    "RawChunkRecord",
    "assemble_chunk_step_transitions",
    "backfill_post_step_residual_obs",
    "count_executed_steps_from_infos",
    "infer_chunk_residual_obs",
]
