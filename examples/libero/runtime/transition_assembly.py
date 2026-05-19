from __future__ import annotations

"""Post-hoc transition assembly helpers for LIBERO residual chunk rollout."""

from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import Any
from typing import Sequence

import numpy as np

from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.rollout.async_transition_assembly import (
    AsyncTransitionAssemblyCoordinator,
)
from serl_torch.examples.libero.env.observation import build_libero_state
from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input
from serl_torch.examples.libero.runtime.key_rl import StageRange
from serl_torch.examples.libero.runtime.key_rl import key_rl_active_step_ranges
from serl_torch.examples.libero.runtime.key_rl import key_rl_step_in_ranges


@dataclass(frozen=True)
class PrefetchedDecisionObs:
    base_actions: np.ndarray
    residual_obs: dict[str, np.ndarray]


@dataclass(frozen=True)
class ChunkExecutionRecord:
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
    start_obs: dict[str, Any] | None = None

    @classmethod
    def from_env_chunk_result(
        cls,
        *,
        episode_id: int,
        episode_step_start: int,
        residual_obs_before_chunk: dict[str, np.ndarray],
        action_chunk: np.ndarray,
        chunk_result: dict[str, Any],
    ) -> "ChunkExecutionRecord":
        post_step_observations = list(chunk_result["observations"])
        rewards = [float(value) for value in chunk_result["rewards"]]
        dones = [bool(value) for value in chunk_result["dones"]]
        infos = [dict(value) for value in chunk_result["infos"]]
        executed_steps = int(chunk_result.get("num_steps", len(rewards)))
        if executed_steps <= 0:
            raise RuntimeError("step_chunk returned no executed steps")

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
        chunk_done = bool(chunk_result["done"])
        if bool(dones[executed_steps - 1]) != bool(chunk_done):
            raise ValueError(
                "step_chunk result done does not match final per-step done: "
                f"done={chunk_done} final_step_done={bool(dones[executed_steps - 1])}"
            )

        return cls(
            episode_id=int(episode_id),
            episode_step_start=int(episode_step_start),
            residual_obs_before_chunk=residual_obs_before_chunk,
            action_chunk=action_chunk_array[:executed_steps],
            post_step_observations=post_step_observations[:executed_steps],
            rewards=rewards[:executed_steps],
            dones=dones[:executed_steps],
            infos=infos[:executed_steps],
            final_obs=dict(chunk_result["obs"]),
            chunk_done=chunk_done,
            chunk_truncated=bool(chunk_result["truncated"]),
            reward_sum=float(chunk_result["reward_sum"]),
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


def _build_policy_endpoint_cfg(
    cfg: Any,
    *,
    host: str,
    port: int,
) -> Any:
    return SimpleNamespace(
        policy=SimpleNamespace(
            type=str(cfg.policy.type),
            host=str(host),
            port=int(port),
        ),
        env=SimpleNamespace(
            action_dim=int(cfg.env.action_dim),
        ),
    )


def infer_chunk_residual_obs(
    *,
    obs: dict[str, Any],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    robot_state = build_libero_state(obs)
    image_observations = extract_libero_images(obs)
    policy_input = build_libero_policy_input(
        prompt=task_prompt,
        state=robot_state,
        images=image_observations,
    )
    base_actions, _ = policy_client.infer(policy_input)
    base_actions = prepare_base_actions_chunk(
        base_actions=base_actions,
        chunk_horizon=chunk_horizon,
    )
    residual_obs = build_chunk_residual_obs(
        robot_state=robot_state,
        images=image_observations,
        image_keys=image_keys,
        base_actions=base_actions,
        residual_alpha=residual_alpha,
    )
    return np.asarray(base_actions, dtype=np.float32), residual_obs


def infer_chunk_residual_obs_many(
    *,
    observations: list[dict[str, Any]],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    return infer_chunk_residual_obs_many_with_prompts(
        observations=observations,
        task_prompts=[str(task_prompt)] * len(observations),
        policy_client=policy_client,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        residual_alpha=residual_alpha,
    )


def infer_chunk_residual_obs_many_with_prompts(
    *,
    observations: list[dict[str, Any]],
    task_prompts: Sequence[str],
    policy_client: Any,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    if not observations:
        return [], []
    prompt_list = [str(task_prompt) for task_prompt in task_prompts]
    if len(prompt_list) != len(observations):
        raise ValueError(
            "task_prompts must align with observations for batched residual inference: "
            f"got prompts={len(prompt_list)} observations={len(observations)}"
        )

    infer_many = getattr(policy_client, "infer_many", None)
    if not callable(infer_many):
        base_action_chunks: list[np.ndarray] = []
        residual_observations: list[dict[str, np.ndarray]] = []
        for obs, task_prompt in zip(observations, prompt_list):
            base_actions, residual_obs = infer_chunk_residual_obs(
                obs=obs,
                task_prompt=task_prompt,
                policy_client=policy_client,
                chunk_horizon=chunk_horizon,
                image_keys=image_keys,
                residual_alpha=residual_alpha,
            )
            base_action_chunks.append(base_actions)
            residual_observations.append(residual_obs)
        return base_action_chunks, residual_observations

    robot_states = [build_libero_state(obs) for obs in observations]
    image_batches = [extract_libero_images(obs) for obs in observations]
    policy_inputs = [
        build_libero_policy_input(
            prompt=prompt,
            state=robot_state,
            images=image_observations,
        )
        for prompt, robot_state, image_observations in zip(
            prompt_list,
            robot_states,
            image_batches,
        )
    ]
    action_chunks, _batch_info = infer_many(policy_inputs)
    if len(action_chunks) != len(observations):
        raise ValueError(
            "infer_many returned a mismatched batch length: "
            f"got {len(action_chunks)}, expected {len(observations)}"
        )

    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    for robot_state, image_observations, raw_actions in zip(
        robot_states,
        image_batches,
        action_chunks,
    ):
        next_base_actions = prepare_base_actions_chunk(
            base_actions=raw_actions,
            chunk_horizon=chunk_horizon,
        )
        next_residual_obs = build_chunk_residual_obs(
            robot_state=robot_state,
            images=image_observations,
            image_keys=image_keys,
            base_actions=next_base_actions,
            residual_alpha=residual_alpha,
        )
        base_action_chunks.append(np.asarray(next_base_actions, dtype=np.float32))
        residual_observations.append(next_residual_obs)
    return base_action_chunks, residual_observations


def _build_chunk_assembly_result(
    *,
    raw: ChunkExecutionRecord,
    residual_obs_before_chunk: dict[str, np.ndarray],
    next_base_actions: Sequence[np.ndarray],
    next_residual_observations: Sequence[dict[str, np.ndarray]],
    active_step_ranges: tuple[StageRange, ...] | None = None,
) -> AssemblyResult:
    expected_steps = int(raw.executed_steps)
    if len(next_base_actions) != expected_steps:
        raise ValueError(
            "next_base_actions must match raw.executed_steps: "
            f"got {len(next_base_actions)} expected {expected_steps}"
        )
    if len(next_residual_observations) != expected_steps:
        raise ValueError(
            "next_residual_observations must match raw.executed_steps: "
            f"got {len(next_residual_observations)} expected {expected_steps}"
        )

    transitions = assemble_chunk_step_transitions(
        episode_id=int(raw.episode_id),
        episode_step_start=int(raw.episode_step_start),
        residual_obs_before_chunk=residual_obs_before_chunk,
        executed_actions=raw.action_chunk,
        rewards=raw.rewards,
        dones=raw.dones,
        infos=raw.infos,
        next_residual_observations=list(next_residual_observations),
        chunk_truncated=bool(raw.chunk_truncated),
        active_step_ranges=active_step_ranges,
    )

    episode_done = bool(raw.chunk_done or raw.chunk_truncated)
    prefetched = None
    if not episode_done and expected_steps > 0:
        prefetched = PrefetchedDecisionObs(
            base_actions=np.asarray(next_base_actions[-1], dtype=np.float32),
            residual_obs=dict(next_residual_observations[-1]),
        )

    return AssemblyResult(
        transitions=transitions,
        prefetched=prefetched,
        next_obs=dict(raw.final_obs),
        episode_done=episode_done,
        env_steps_delta=expected_steps,
        episode_steps_delta=expected_steps,
        episode_return_delta=float(raw.reward_sum),
        episode_success=any(bool(info.get("env_done", False)) for info in raw.infos),
        last_info=dict(raw.chunk_info),
    )


class LiberoTransitionAssembler:
    def __init__(
        self,
        *,
        policy_client: Any,
        chunk_horizon: int,
        image_keys: tuple[str, ...],
        residual_alpha: float,
        active_step_ranges: tuple[StageRange, ...] | None = None,
    ) -> None:
        self.policy_client = policy_client
        self.chunk_horizon = int(chunk_horizon)
        self.image_keys = tuple(image_keys)
        self.residual_alpha = float(residual_alpha)
        self.active_step_ranges = active_step_ranges

    def infer_decision_obs(
        self,
        *,
        obs: dict[str, Any],
        task_prompt: str,
    ) -> PrefetchedDecisionObs:
        base_actions, residual_obs = infer_chunk_residual_obs(
            obs=obs,
            task_prompt=task_prompt,
            policy_client=self.policy_client,
            chunk_horizon=self.chunk_horizon,
            image_keys=self.image_keys,
            residual_alpha=self.residual_alpha,
        )
        return PrefetchedDecisionObs(
            base_actions=base_actions,
            residual_obs=residual_obs,
        )

    def backfill_post_step_residual_obs(
        self,
        *,
        observations: list[dict[str, Any]],
        task_prompt: str,
    ) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        return backfill_post_step_residual_obs(
            observations=observations,
            task_prompt=task_prompt,
            policy_client=self.policy_client,
            chunk_horizon=self.chunk_horizon,
            image_keys=self.image_keys,
            residual_alpha=self.residual_alpha,
        )

    def process_chunk(
        self,
        *,
        raw: ChunkExecutionRecord,
        task_prompt: str,
    ) -> AssemblyResult:
        (
            backfilled_base_actions,
            backfilled_residual_obs,
        ) = self.backfill_post_step_residual_obs(
            observations=raw.post_step_observations,
            task_prompt=task_prompt,
        )
        transitions = assemble_chunk_step_transitions(
            episode_id=int(raw.episode_id),
            episode_step_start=int(raw.episode_step_start),
            residual_obs_before_chunk=raw.residual_obs_before_chunk,
            executed_actions=raw.action_chunk,
            rewards=raw.rewards,
            dones=raw.dones,
            infos=raw.infos,
            next_residual_observations=backfilled_residual_obs,
            chunk_truncated=bool(raw.chunk_truncated),
            active_step_ranges=getattr(self, "active_step_ranges", None),
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
            episode_success=any(
                bool(info.get("env_done", False)) for info in raw.infos
            ),
            last_info=dict(raw.chunk_info),
        )


class BatchAwareLiberoTransitionAssembler(LiberoTransitionAssembler):
    def backfill_post_step_residual_obs(
        self,
        *,
        observations: list[dict[str, Any]],
        task_prompt: str,
    ) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        return backfill_post_step_residual_obs_batch_aware(
            observations=observations,
            task_prompt=task_prompt,
            policy_client=self.policy_client,
            chunk_horizon=self.chunk_horizon,
            image_keys=self.image_keys,
            residual_alpha=self.residual_alpha,
        )

    def process_chunk(
        self,
        *,
        raw: ChunkExecutionRecord,
        task_prompt: str,
    ) -> AssemblyResult:
        if raw.start_obs is None:
            return super().process_chunk(
                raw=raw,
                task_prompt=task_prompt,
            )
        return self.process_chunk_batch(
            raw_chunks=(raw,),
            task_prompts=(task_prompt,),
        )[0]

    def process_chunk_batch(
        self,
        *,
        raw_chunks: Sequence[ChunkExecutionRecord],
        task_prompts: Sequence[str],
    ) -> list[AssemblyResult]:
        raw_list = list(raw_chunks)
        prompt_list = [str(task_prompt) for task_prompt in task_prompts]
        if len(raw_list) != len(prompt_list):
            raise ValueError(
                "raw_chunks and task_prompts must have the same length: "
                f"got raw_chunks={len(raw_list)} task_prompts={len(prompt_list)}"
            )
        if not raw_list:
            return []

        assembled_results: list[AssemblyResult | None] = [None] * len(raw_list)
        batchable_specs: list[tuple[int, int]] = []
        batched_observations: list[dict[str, Any]] = []
        batched_prompts: list[str] = []

        for idx, (raw, task_prompt) in enumerate(zip(raw_list, prompt_list)):
            if raw.start_obs is None:
                assembled_results[idx] = super().process_chunk(
                    raw=raw,
                    task_prompt=task_prompt,
                )
                continue

            observation_count = int(raw.executed_steps) + 1
            batchable_specs.append((idx, observation_count))
            batched_observations.append(dict(raw.start_obs))
            batched_prompts.append(task_prompt)
            batched_observations.extend(list(raw.post_step_observations))
            batched_prompts.extend([task_prompt] * int(raw.executed_steps))

        if batchable_specs:
            (
                batched_base_actions,
                batched_residual_obs,
            ) = infer_chunk_residual_obs_many_with_prompts(
                observations=batched_observations,
                task_prompts=batched_prompts,
                policy_client=self.policy_client,
                chunk_horizon=self.chunk_horizon,
                image_keys=self.image_keys,
                residual_alpha=self.residual_alpha,
            )
            offset = 0
            for idx, observation_count in batchable_specs:
                current_base_actions = batched_base_actions[
                    offset : offset + observation_count
                ]
                current_residual_obs = batched_residual_obs[
                    offset : offset + observation_count
                ]
                if len(current_residual_obs) != int(observation_count):
                    raise ValueError(
                        "combined chunk residual backfill returned an unexpected length slice: "
                        f"got {len(current_residual_obs)} expected {int(observation_count)}"
                    )
                assembled_results[idx] = _build_chunk_assembly_result(
                    raw=raw_list[idx],
                    residual_obs_before_chunk=current_residual_obs[0],
                    next_base_actions=current_base_actions[1:],
                    next_residual_observations=current_residual_obs[1:],
                    active_step_ranges=self.active_step_ranges,
                )
                offset += int(observation_count)
            if offset != len(batched_residual_obs):
                raise ValueError(
                    "combined chunk residual backfill left unused outputs: "
                    f"consumed {offset} of {len(batched_residual_obs)}"
                )

        final_results: list[AssemblyResult] = []
        for idx, assembled in enumerate(assembled_results):
            if assembled is None:
                raise RuntimeError(
                    f"chunk batch assembly did not produce a result for index {idx}"
                )
            final_results.append(assembled)
        return final_results


class LiberoActorTransitionAssembler:
    def __init__(
        self,
        *,
        cfg: Any,
        policy_client: Any,
        logger: logging.Logger,
    ) -> None:
        self._logger = logger
        self._sync_assembler = BatchAwareLiberoTransitionAssembler(
            policy_client=policy_client,
            chunk_horizon=int(cfg.residual.chunk_horizon),
            image_keys=tuple(cfg.obs.image_keys),
            residual_alpha=float(cfg.residual.alpha),
            active_step_ranges=key_rl_active_step_ranges(
                getattr(cfg, "key_rl", None),
            ),
        )
        self._prefetched: PrefetchedDecisionObs | None = None
        self._last_submitted_chunk_seq: int | None = None
        self._max_pending_chunks = int(cfg.backfill_policy.max_pending_chunks)
        self._async_assembly: AsyncTransitionAssemblyCoordinator[
            ChunkExecutionRecord,
            dict[str, Any],
            dict[str, np.ndarray],
            AssemblyResult,
        ] | None = None

        if not bool(cfg.backfill_policy.enabled):
            return

        endpoint_cfg = _build_policy_endpoint_cfg(
            cfg,
            host=str(cfg.backfill_policy.host),
            port=int(cfg.backfill_policy.port),
        )
        from serl_launcher.policy.typed_factory import build_policy_client

        backfill_policy_client = build_policy_client(endpoint_cfg, logger=logger)
        close_fn = getattr(backfill_policy_client, "close", None)
        self._async_assembly = AsyncTransitionAssemblyCoordinator(
            backfill_fn=lambda observations, task_prompt: self._backfill_residual_observations(
                observations=observations,
                task_prompt=task_prompt,
                policy_client=backfill_policy_client,
            ),
            build_result_fn=self._build_assembly_result,
            thread_name_prefix="libero-transition-assembly",
            logger=logger,
            close_fn=close_fn if callable(close_fn) else None,
            close_error_message="ignored backfill policy client close error",
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
        return self._sync_assembler.infer_decision_obs(
            obs=obs,
            task_prompt=task_prompt,
        )

    def pop_prefetched_decision_obs(self) -> PrefetchedDecisionObs | None:
        if self._async_assembly is not None:
            return None
        decision_obs = self._prefetched
        self._prefetched = None
        return decision_obs

    def process_chunk(
        self,
        *,
        raw: ChunkExecutionRecord,
        task_prompt: str,
    ) -> AssemblyResult:
        return self._sync_assembler.process_chunk(
            raw=raw,
            task_prompt=task_prompt,
        )

    def drain_ready(self) -> list[AssemblyResult]:
        if self._async_assembly is None:
            return []
        return self._async_assembly.pop_committable()

    def handle_chunk(
        self,
        *,
        raw: ChunkExecutionRecord,
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

    def _backfill_residual_observations(
        self,
        *,
        observations: list[dict[str, Any]],
        task_prompt: str,
        policy_client: Any,
    ) -> list[dict[str, np.ndarray]]:
        (
            _base_action_chunks,
            next_residual_observations,
        ) = backfill_post_step_residual_obs_batch_aware(
            observations=observations,
            task_prompt=task_prompt,
            policy_client=policy_client,
            chunk_horizon=self._sync_assembler.chunk_horizon,
            image_keys=self._sync_assembler.image_keys,
            residual_alpha=self._sync_assembler.residual_alpha,
        )
        return next_residual_observations

    def _build_assembly_result(
        self,
        raw: ChunkExecutionRecord,
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
            next_residual_observations=next_residual_observations,
            chunk_truncated=bool(raw.chunk_truncated),
            active_step_ranges=getattr(
                self._sync_assembler,
                "active_step_ranges",
                None,
            ),
        )
        return AssemblyResult(
            transitions=transitions,
            prefetched=None,
            next_obs=dict(raw.final_obs),
            episode_done=bool(raw.chunk_done or raw.chunk_truncated),
            env_steps_delta=int(raw.executed_steps),
            episode_steps_delta=int(raw.executed_steps),
            episode_return_delta=float(raw.reward_sum),
            episode_success=any(
                bool(info.get("env_done", False)) for info in raw.infos
            ),
            last_info=dict(raw.chunk_info),
        )


def backfill_post_step_residual_obs(
    *,
    observations: list[dict[str, Any]],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    for post_step_obs in observations:
        next_base_actions, next_residual_obs = infer_chunk_residual_obs(
            obs=post_step_obs,
            task_prompt=task_prompt,
            policy_client=policy_client,
            chunk_horizon=chunk_horizon,
            image_keys=image_keys,
            residual_alpha=residual_alpha,
        )
        base_action_chunks.append(next_base_actions)
        residual_observations.append(next_residual_obs)
    return base_action_chunks, residual_observations


def backfill_post_step_residual_obs_batch_aware(
    *,
    observations: list[dict[str, Any]],
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    return infer_chunk_residual_obs_many(
        observations=observations,
        task_prompt=task_prompt,
        policy_client=policy_client,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        residual_alpha=residual_alpha,
    )


def assemble_chunk_step_transitions(
    *,
    episode_id: int,
    episode_step_start: int,
    residual_obs_before_chunk: dict[str, np.ndarray],
    executed_actions: np.ndarray,
    rewards: list[float],
    dones: list[bool],
    infos: list[dict[str, Any]],
    next_residual_observations: list[dict[str, np.ndarray]],
    chunk_truncated: bool = False,
    active_step_ranges: tuple[StageRange, ...] | None = None,
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
        step_info = dict(infos[step_idx])
        is_last_step = int(step_idx) == int(executed_steps - 1)
        done_flag = bool(dones[step_idx]) or (
            bool(chunk_truncated) and bool(is_last_step)
        )
        next_episode_step = int(episode_step_start + step_idx + 1)
        next_key_rl_active = key_rl_step_in_ranges(
            next_episode_step,
            active_step_ranges,
        )
        mask_flag = float(
            0.0
            if (
                done_flag
                or bool(step_info.get("env_done", False))
                or not bool(next_key_rl_active)
            )
            else 1.0
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
                "masks": mask_flag,
                "dones": done_flag,
            }
        )
        current_residual_obs = next_residual_obs
    return transitions


__all__ = [
    "AssemblyResult",
    "BatchAwareLiberoTransitionAssembler",
    "LiberoActorTransitionAssembler",
    "LiberoTransitionAssembler",
    "PrefetchedDecisionObs",
    "ChunkExecutionRecord",
    "assemble_chunk_step_transitions",
    "backfill_post_step_residual_obs",
    "backfill_post_step_residual_obs_batch_aware",
    "infer_chunk_residual_obs",
    "infer_chunk_residual_obs_many",
    "infer_chunk_residual_obs_many_with_prompts",
]
