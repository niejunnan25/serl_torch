from __future__ import annotations

"""Synthetic 30Hz rollout benchmark for LIBERO actor dataflow variants.

This benchmark compares four actor-side rollout shapes:

1. baseline_step:
   Mirrors examples/libero/scripts/run_residual_training.py
2. copy_sync:
   Mirrors the synchronous chunk assembly path in
   examples/libero/scripts/run_residual_training_copy.py
3. copy_async:
   Mirrors the async backfill / ordered commit path in
   examples/libero/scripts/run_residual_training_copy.py
4. copy_copy:
   Mirrors the batch-aware async backfill path in
   examples/libero/scripts/run_residual_training_copy_copy.py

The environment is synthetic but shaped like LIBERO observations so the
benchmark still exercises:

- build_libero_policy_input(...)
- build_chunk_residual_obs(...)
- RawChunkRecord / assemble_chunk_step_transitions(...)

Usage:

python test/benchmark_libero_rollout_30hz.py
python test/benchmark_libero_rollout_30hz.py --env-steps 300 --json-out /tmp/libero_30hz.json
"""

import argparse
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from threading import Lock
import time
from types import SimpleNamespace
from typing import Any
from typing import Iterable
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input
from serl_torch.examples.libero.residual_observation import build_chunk_residual_obs
from serl_torch.examples.libero.residual_observation import prepare_base_actions_chunk
from serl_torch.examples.libero.transition_assembly import PrefetchedDecisionObs
from serl_torch.examples.libero.transition_assembly import RawChunkRecord
from serl_torch.examples.libero.transition_assembly import (
    assemble_chunk_step_transitions,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    env_hz: float
    env_steps: int
    episode_length: int
    action_dim: int
    chunk_horizon: int
    residual_alpha: float
    image_keys: tuple[str, ...]
    policy_infer_ms: float
    backfill_policy_infer_ms: float
    residual_sample_ms: float
    replay_insert_ms: float
    step_rpc_ms: float
    step_chunk_rpc_ms: float
    image_size: int
    max_pending_chunks: int
    variants: tuple[str, ...]
    json_out: str | None


@dataclass(frozen=True)
class VariantResult:
    variant: str
    env_steps: int
    chunk_count: int
    wall_time_sec: float
    inserts: int
    averages: dict[str, float]
    totals: dict[str, float]
    counts: dict[str, int]
    normalized: dict[str, float]
    top_bottlenecks: list[dict[str, Any]]


class StatsCollector:
    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    @contextmanager
    def context(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(key, time.perf_counter() - start)

    def add_duration(self, key: str, duration_sec: float) -> None:
        with self._lock:
            self._totals[key] += float(duration_sec)
            self._counts[key] += 1

    def snapshot(self) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
        with self._lock:
            totals = dict(self._totals)
            counts = dict(self._counts)
        averages = {
            key: (totals[key] / counts[key]) for key in totals.keys() if counts[key] > 0
        }
        return totals, counts, averages


class FakePolicyClient:
    def __init__(
        self,
        *,
        action_dim: int,
        chunk_horizon: int,
        infer_ms: float,
    ) -> None:
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.infer_sec = float(infer_ms) / 1000.0

    def infer(self, policy_input: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self.infer_sec > 0.0:
            time.sleep(self.infer_sec)
        state = np.asarray(policy_input.state, dtype=np.float32).reshape(-1)
        action_chunk = np.zeros(
            (int(self.chunk_horizon), int(self.action_dim)),
            dtype=np.float32,
        )
        base_value = float(state[0]) if state.size > 0 else 0.0
        for step_idx in range(int(self.chunk_horizon)):
            action_chunk[step_idx, :] = base_value + float(step_idx) * 0.01
        return action_chunk, {"backend": "fake"}

    def infer_many(
        self,
        policy_inputs: Sequence[Any],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        if self.infer_sec > 0.0:
            time.sleep(self.infer_sec)
        action_chunks: list[np.ndarray] = []
        for policy_input in policy_inputs:
            state = np.asarray(policy_input.state, dtype=np.float32).reshape(-1)
            action_chunk = np.zeros(
                (int(self.chunk_horizon), int(self.action_dim)),
                dtype=np.float32,
            )
            base_value = float(state[0]) if state.size > 0 else 0.0
            for step_idx in range(int(self.chunk_horizon)):
                action_chunk[step_idx, :] = base_value + float(step_idx) * 0.01
            action_chunks.append(action_chunk)
        return action_chunks, {
            "backend": "fake",
            "batch_size": int(len(policy_inputs)),
        }

    def close(self) -> None:
        return


class FakeResidualAgent:
    def __init__(
        self,
        *,
        chunk_horizon: int,
        policy_action_dim: int,
        sample_ms: float,
    ) -> None:
        self.chunk_horizon = int(chunk_horizon)
        self.policy_action_dim = int(policy_action_dim)
        self.sample_sec = float(sample_ms) / 1000.0

    def sample_action(
        self,
        residual_obs: dict[str, np.ndarray],
        *,
        deterministic: bool,
    ) -> np.ndarray:
        del residual_obs, deterministic
        if self.sample_sec > 0.0:
            time.sleep(self.sample_sec)
        return np.zeros(
            (int(self.chunk_horizon), int(self.policy_action_dim)),
            dtype=np.float32,
        )


class FakeDataStore:
    def __init__(self, *, replay_insert_ms: float) -> None:
        self.replay_insert_sec = float(replay_insert_ms) / 1000.0
        self.insert_count = 0

    def insert(self, transition: dict[str, Any]) -> None:
        del transition
        if self.replay_insert_sec > 0.0:
            time.sleep(self.replay_insert_sec)
        self.insert_count += 1


class FakeLibero30HzEnv:
    def __init__(
        self,
        *,
        hz: float,
        episode_length: int,
        image_size: int,
        action_dim: int,
        step_rpc_ms: float,
        step_chunk_rpc_ms: float,
        stats: StatsCollector,
    ) -> None:
        self.task_description = "synthetic libero 30hz benchmark"
        self._step_dt = 1.0 / float(hz)
        self._episode_length = int(episode_length)
        self._image_size = int(image_size)
        self._action_dim = int(action_dim)
        self._step_rpc_sec = float(step_rpc_ms) / 1000.0
        self._step_chunk_rpc_sec = float(step_chunk_rpc_ms) / 1000.0
        self._stats = stats
        self._global_step = 0
        self._episode_step = 0

        self._agentview_template = np.zeros(
            (int(self._image_size), int(self._image_size), 3),
            dtype=np.uint8,
        )
        self._wrist_template = np.full(
            (int(self._image_size), int(self._image_size), 3),
            fill_value=64,
            dtype=np.uint8,
        )

    def reset(self, seed: int | None = None, init_episode_idx: int | None = None) -> dict[str, Any]:
        del seed, init_episode_idx
        self._episode_step = 0
        return self._make_obs(global_step=self._global_step, episode_step=self._episode_step)

    def _make_obs(self, *, global_step: int, episode_step: int) -> dict[str, Any]:
        pos = np.asarray(
            [
                0.01 * float(global_step),
                0.02 * float(episode_step),
                0.3,
            ],
            dtype=np.float32,
        )
        axis_angle = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        gripper = np.asarray([0.04, -0.04], dtype=np.float32)
        return {
            "robot0_eef_pos": pos,
            "robot0_eef_axis_angle": axis_angle,
            "robot0_gripper_qpos": gripper,
            "agentview_image": self._agentview_template,
            "robot0_eye_in_hand_image": self._wrist_template,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if int(action.shape[0]) != int(self._action_dim):
            raise ValueError(
                f"Unexpected action shape {action.shape}, expected ({self._action_dim},)"
            )
        if self._step_rpc_sec > 0.0:
            with self._stats.context("env.step_rpc_overhead"):
                time.sleep(self._step_rpc_sec)
        with self._stats.context("env.wait"):
            time.sleep(self._step_dt)

        self._global_step += 1
        self._episode_step += 1
        obs = self._make_obs(
            global_step=int(self._global_step),
            episode_step=int(self._episode_step),
        )
        done = bool(self._episode_step >= int(self._episode_length))
        reward = 1.0 if done else 0.0
        info = {
            "env_done": bool(done),
            "success": bool(done),
        }
        return obs, float(reward), bool(done), False, info

    def step_chunk(self, actions: np.ndarray) -> dict[str, Any]:
        action_chunk = np.asarray(actions, dtype=np.float32)
        if action_chunk.ndim != 2 or int(action_chunk.shape[1]) != int(self._action_dim):
            raise ValueError(
                f"Unexpected action chunk shape {action_chunk.shape}, "
                f"expected (*, {self._action_dim})"
            )

        if self._step_chunk_rpc_sec > 0.0:
            with self._stats.context("env.step_chunk_rpc_overhead"):
                time.sleep(self._step_chunk_rpc_sec)

        observations: list[dict[str, Any]] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, Any]] = []
        for _step_action in action_chunk:
            with self._stats.context("env.wait"):
                time.sleep(self._step_dt)
            self._global_step += 1
            self._episode_step += 1
            done = bool(self._episode_step >= int(self._episode_length))
            obs = self._make_obs(
                global_step=int(self._global_step),
                episode_step=int(self._episode_step),
            )
            reward = 1.0 if done else 0.0
            info = {
                "env_done": bool(done),
                "success": bool(done),
            }
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
            if done:
                break

        if not observations:
            raise RuntimeError("step_chunk received an empty action chunk")

        return {
            "obs": dict(observations[-1]),
            "observations": observations,
            "reward_sum": float(sum(rewards)),
            "rewards": rewards,
            "dones": dones,
            "done": bool(dones[-1]),
            "truncated": False,
            "infos": infos,
            "info": dict(infos[-1]),
            "num_steps": int(len(rewards)),
        }


def _build_action_spec(cfg: BenchmarkConfig) -> ResidualActionSpec:
    typed_cfg = SimpleNamespace(
        residual=SimpleNamespace(
            alpha=float(cfg.residual_alpha),
            action_mask=tuple(True for _ in range(int(cfg.action_dim))),
            action_limits=tuple(1.0 for _ in range(int(cfg.action_dim))),
            clip_gripper=True,
            chunk_horizon=int(cfg.chunk_horizon),
        )
    )
    return ResidualActionSpec.from_cfg(typed_cfg, action_dim=int(cfg.action_dim))


def _infer_decision_obs(
    *,
    stats: StatsCollector,
    metric_prefix: str,
    obs: dict[str, Any],
    task_prompt: str,
    policy_client: FakePolicyClient,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> PrefetchedDecisionObs:
    with stats.context(f"{metric_prefix}.build_policy_input"):
        policy_input = build_libero_policy_input(obs, task_prompt)
    with stats.context(f"{metric_prefix}.policy_infer"):
        base_actions, _ = policy_client.infer(policy_input)
    with stats.context(f"{metric_prefix}.prepare_base_actions"):
        base_actions = prepare_base_actions_chunk(
            base_actions=base_actions,
            chunk_horizon=int(chunk_horizon),
        )
    with stats.context(f"{metric_prefix}.build_residual_obs"):
        residual_obs = build_chunk_residual_obs(
            obs=obs,
            base_actions=base_actions,
            image_keys=image_keys,
            residual_alpha=float(residual_alpha),
        )
    return PrefetchedDecisionObs(
        base_actions=np.asarray(base_actions, dtype=np.float32),
        residual_obs=residual_obs,
    )


def _backfill_post_step_residual_obs(
    *,
    stats: StatsCollector,
    metric_prefix: str,
    observations: Sequence[dict[str, Any]],
    task_prompt: str,
    policy_client: FakePolicyClient,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    for post_step_obs in observations:
        decision_obs = _infer_decision_obs(
            stats=stats,
            metric_prefix=metric_prefix,
            obs=post_step_obs,
            task_prompt=task_prompt,
            policy_client=policy_client,
            chunk_horizon=int(chunk_horizon),
            image_keys=image_keys,
            residual_alpha=float(residual_alpha),
        )
        base_action_chunks.append(np.asarray(decision_obs.base_actions, dtype=np.float32))
        residual_observations.append(decision_obs.residual_obs)
    return base_action_chunks, residual_observations


def _backfill_post_step_residual_obs_batch_aware(
    *,
    stats: StatsCollector,
    metric_prefix: str,
    observations: Sequence[dict[str, Any]],
    task_prompt: str,
    policy_client: FakePolicyClient,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    if not observations:
        return [], []

    with stats.context(f"{metric_prefix}.build_policy_input_many"):
        policy_inputs = [
            build_libero_policy_input(obs, task_prompt) for obs in observations
        ]
    with stats.context(f"{metric_prefix}.policy_infer_many"):
        action_chunks, _ = policy_client.infer_many(policy_inputs)

    base_action_chunks: list[np.ndarray] = []
    residual_observations: list[dict[str, np.ndarray]] = []
    for post_step_obs, raw_actions in zip(observations, action_chunks):
        with stats.context(f"{metric_prefix}.prepare_base_actions"):
            base_actions = prepare_base_actions_chunk(
                base_actions=raw_actions,
                chunk_horizon=int(chunk_horizon),
            )
        with stats.context(f"{metric_prefix}.build_residual_obs"):
            residual_obs = build_chunk_residual_obs(
                obs=post_step_obs,
                base_actions=base_actions,
                image_keys=image_keys,
                residual_alpha=float(residual_alpha),
            )
        base_action_chunks.append(np.asarray(base_actions, dtype=np.float32))
        residual_observations.append(residual_obs)
    return base_action_chunks, residual_observations


@dataclass
class _PendingAsyncChunk:
    chunk_seq: int
    raw: RawChunkRecord
    backfill_future: Future[tuple[list[np.ndarray], list[dict[str, np.ndarray]]]]
    expects_tail_handoff: bool
    tail_next_residual_obs: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class _AssembledChunk:
    transitions: list[dict[str, Any]]
    next_obs: dict[str, Any]
    episode_done: bool
    env_steps_delta: int
    episode_steps_delta: int
    episode_return_delta: float
    episode_success: bool
    last_info: dict[str, Any]


class BenchmarkAsyncChunkAssemblyCoordinator:
    def __init__(
        self,
        *,
        stats: StatsCollector,
        policy_client: FakePolicyClient,
        chunk_horizon: int,
        image_keys: tuple[str, ...],
        residual_alpha: float,
        batch_aware: bool,
    ) -> None:
        self._stats = stats
        self._policy_client = policy_client
        self._chunk_horizon = int(chunk_horizon)
        self._image_keys = tuple(image_keys)
        self._residual_alpha = float(residual_alpha)
        self._batch_aware = bool(batch_aware)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bench-libero-backfill",
        )
        self._pending: dict[int, _PendingAsyncChunk] = {}
        self._next_submit_chunk_seq = 0
        self._next_commit_chunk_seq = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def next_commit_chunk_seq(self) -> int:
        return int(self._next_commit_chunk_seq)

    def submit_chunk(
        self,
        *,
        raw: RawChunkRecord,
        task_prompt: str,
        expect_tail_handoff: bool,
    ) -> int:
        chunk_seq = int(self._next_submit_chunk_seq)
        self._next_submit_chunk_seq += 1
        observations_to_backfill: list[dict[str, Any]]
        if bool(expect_tail_handoff):
            observations_to_backfill = list(raw.post_step_observations[:-1])
        else:
            observations_to_backfill = list(raw.post_step_observations)
        backfill_fn = (
            _backfill_post_step_residual_obs_batch_aware
            if self._batch_aware
            else _backfill_post_step_residual_obs
        )
        backfill_future = self._executor.submit(
            backfill_fn,
            stats=self._stats,
            metric_prefix="async_backfill",
            observations=observations_to_backfill,
            task_prompt=task_prompt,
            policy_client=self._policy_client,
            chunk_horizon=int(self._chunk_horizon),
            image_keys=self._image_keys,
            residual_alpha=float(self._residual_alpha),
        )
        self._pending[chunk_seq] = _PendingAsyncChunk(
            chunk_seq=chunk_seq,
            raw=raw,
            backfill_future=backfill_future,
            expects_tail_handoff=bool(expect_tail_handoff),
        )
        return chunk_seq

    def provide_tail(
        self,
        *,
        chunk_seq: int,
        decision_obs: PrefetchedDecisionObs,
    ) -> None:
        pending = self._pending.get(int(chunk_seq))
        if pending is None:
            raise KeyError(f"Unknown pending chunk_seq={chunk_seq}")
        if not pending.expects_tail_handoff:
            raise ValueError(f"Chunk {chunk_seq} does not expect a tail handoff")
        if pending.tail_next_residual_obs is not None:
            raise ValueError(f"Chunk {chunk_seq} tail handoff already provided")
        pending.tail_next_residual_obs = {
            key: np.array(value, copy=True)
            for key, value in decision_obs.residual_obs.items()
        }

    def finalize_tail_with_fallback(
        self,
        *,
        chunk_seq: int,
        task_prompt: str,
    ) -> None:
        pending = self._pending.get(int(chunk_seq))
        if pending is None or not pending.expects_tail_handoff:
            return
        if pending.tail_next_residual_obs is not None:
            return
        decision_obs = _infer_decision_obs(
            stats=self._stats,
            metric_prefix="async_backfill_tail",
            obs=pending.raw.final_obs,
            task_prompt=task_prompt,
            policy_client=self._policy_client,
            chunk_horizon=int(self._chunk_horizon),
            image_keys=self._image_keys,
            residual_alpha=float(self._residual_alpha),
        )
        pending.tail_next_residual_obs = {
            key: np.array(value, copy=True)
            for key, value in decision_obs.residual_obs.items()
        }

    def pop_committable(self, *, block_until_seq: int | None = None) -> list[_AssembledChunk]:
        assembled_chunks: list[_AssembledChunk] = []
        while int(self._next_commit_chunk_seq) in self._pending:
            next_seq = int(self._next_commit_chunk_seq)
            pending = self._pending[next_seq]
            if block_until_seq is not None and next_seq <= int(block_until_seq):
                _base_actions, next_residual_observations = pending.backfill_future.result()
            elif pending.backfill_future.done():
                _base_actions, next_residual_observations = pending.backfill_future.result()
            else:
                break

            if pending.expects_tail_handoff:
                if pending.tail_next_residual_obs is None:
                    if block_until_seq is not None and next_seq <= int(block_until_seq):
                        raise RuntimeError(
                            f"Chunk {next_seq} is missing tail handoff during blocking commit"
                        )
                    break
                next_residual_observations = list(next_residual_observations) + [
                    {
                        key: np.array(value, copy=True)
                        for key, value in pending.tail_next_residual_obs.items()
                    }
                ]

            with self._stats.context("commit_replay.assemble_transitions"):
                transitions = assemble_chunk_step_transitions(
                    episode_id=int(pending.raw.episode_id),
                    episode_step_start=int(pending.raw.episode_step_start),
                    residual_obs_before_chunk=pending.raw.residual_obs_before_chunk,
                    executed_actions=pending.raw.action_chunk,
                    rewards=pending.raw.rewards,
                    dones=pending.raw.dones,
                    infos=pending.raw.infos,
                    next_residual_observations=next_residual_observations,
                )

            assembled_chunks.append(
                _AssembledChunk(
                    transitions=transitions,
                    next_obs=dict(pending.raw.final_obs),
                    episode_done=bool(
                        pending.raw.chunk_done or pending.raw.chunk_truncated
                    ),
                    env_steps_delta=int(pending.raw.executed_steps),
                    episode_steps_delta=int(pending.raw.executed_steps),
                    episode_return_delta=float(pending.raw.reward_sum),
                    episode_success=any(
                        bool(info.get("env_done", False)) for info in pending.raw.infos
                    ),
                    last_info=dict(pending.raw.chunk_info),
                )
            )
            self._pending.pop(next_seq, None)
            self._next_commit_chunk_seq += 1
        return assembled_chunks

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def _run_variant(variant: str, cfg: BenchmarkConfig) -> VariantResult:
    stats = StatsCollector()
    task_prompt = "synthetic libero benchmark"
    env = FakeLibero30HzEnv(
        hz=float(cfg.env_hz),
        episode_length=int(cfg.episode_length),
        image_size=int(cfg.image_size),
        action_dim=int(cfg.action_dim),
        step_rpc_ms=float(cfg.step_rpc_ms),
        step_chunk_rpc_ms=float(cfg.step_chunk_rpc_ms),
        stats=stats,
    )
    action_spec = _build_action_spec(cfg)
    policy_client = FakePolicyClient(
        action_dim=int(cfg.action_dim),
        chunk_horizon=int(cfg.chunk_horizon),
        infer_ms=float(cfg.policy_infer_ms),
    )
    backfill_policy_client = FakePolicyClient(
        action_dim=int(cfg.action_dim),
        chunk_horizon=int(cfg.chunk_horizon),
        infer_ms=float(cfg.backfill_policy_infer_ms),
    )
    agent = FakeResidualAgent(
        chunk_horizon=int(cfg.chunk_horizon),
        policy_action_dim=int(action_spec.policy_action_dim),
        sample_ms=float(cfg.residual_sample_ms),
    )
    data_store = FakeDataStore(replay_insert_ms=float(cfg.replay_insert_ms))

    async_coord: BenchmarkAsyncChunkAssemblyCoordinator | None = None
    if variant in {"copy_async", "copy_copy"}:
        async_coord = BenchmarkAsyncChunkAssemblyCoordinator(
            stats=stats,
            policy_client=backfill_policy_client,
            chunk_horizon=int(cfg.chunk_horizon),
            image_keys=tuple(cfg.image_keys),
            residual_alpha=float(cfg.residual_alpha),
            batch_aware=bool(variant == "copy_copy"),
        )

    env_steps = 0
    episode_id = 0
    chunk_count = 0
    wall_start = time.perf_counter()
    try:
        while env_steps < int(cfg.env_steps):
            episode_id += 1
            obs = env.reset(seed=0, init_episode_idx=episode_id - 1)
            prefetched: PrefetchedDecisionObs | None = None
            pending_tail_chunk_seq: int | None = None
            last_submitted_chunk_seq: int | None = None
            episode_steps = 0

            def commit_async_chunks(*, block_until_seq: int | None = None) -> None:
                if async_coord is None:
                    return
                assembled_chunks = async_coord.pop_committable(
                    block_until_seq=block_until_seq
                )
                for assembled_chunk in assembled_chunks:
                    with stats.context("replay_insert"):
                        for transition in assembled_chunk.transitions:
                            data_store.insert(transition)

            episode_done = False
            while env_steps < int(cfg.env_steps) and not episode_done:
                if async_coord is not None:
                    commit_async_chunks()

                chunk_count += 1
                with stats.context("total"):
                    with stats.context("sample_actions"):
                        if variant in {"copy_async", "copy_copy"}:
                            decision_obs = _infer_decision_obs(
                                stats=stats,
                                metric_prefix="sample_actions.current",
                                obs=obs,
                                task_prompt=task_prompt,
                                policy_client=policy_client,
                                chunk_horizon=int(cfg.chunk_horizon),
                                image_keys=tuple(cfg.image_keys),
                                residual_alpha=float(cfg.residual_alpha),
                            )
                            if pending_tail_chunk_seq is not None:
                                async_coord.provide_tail(
                                    chunk_seq=int(pending_tail_chunk_seq),
                                    decision_obs=decision_obs,
                                )
                                pending_tail_chunk_seq = None
                            base_actions = decision_obs.base_actions
                            residual_obs = decision_obs.residual_obs
                        elif prefetched is None:
                            decision_obs = _infer_decision_obs(
                                stats=stats,
                                metric_prefix="sample_actions.current",
                                obs=obs,
                                task_prompt=task_prompt,
                                policy_client=policy_client,
                                chunk_horizon=int(cfg.chunk_horizon),
                                image_keys=tuple(cfg.image_keys),
                                residual_alpha=float(cfg.residual_alpha),
                            )
                            base_actions = decision_obs.base_actions
                            residual_obs = decision_obs.residual_obs
                        else:
                            base_actions = prefetched.base_actions
                            residual_obs = prefetched.residual_obs
                            prefetched = None

                        with stats.context("sample_actions.residual_sample"):
                            residual_actions = agent.sample_action(
                                residual_obs,
                                deterministic=False,
                            )
                        with stats.context("sample_actions.compose"):
                            final_actions = action_spec.compose_chunk(
                                base_action_chunk=base_actions,
                                residual_action=residual_actions,
                            )

                    remaining_steps = int(cfg.env_steps - env_steps)
                    action_chunk = np.asarray(final_actions, dtype=np.float32)[
                        :remaining_steps
                    ]

                    if variant == "baseline_step":
                        current_residual_obs = residual_obs
                        for action in action_chunk:
                            with stats.context("step_env"):
                                next_obs, reward, done, truncated, info = env.step(action)
                            with stats.context("build_decision_obs"):
                                next_decision_obs = _infer_decision_obs(
                                    stats=stats,
                                    metric_prefix="build_decision_obs.next_step",
                                    obs=next_obs,
                                    task_prompt=task_prompt,
                                    policy_client=policy_client,
                                    chunk_horizon=int(cfg.chunk_horizon),
                                    image_keys=tuple(cfg.image_keys),
                                    residual_alpha=float(cfg.residual_alpha),
                                )
                            transition = {
                                "episode_id": int(episode_id),
                                "episode_step": int(episode_steps),
                                "observations": current_residual_obs,
                                "actions": np.asarray(action, dtype=np.float32).reshape(-1),
                                "next_observations": next_decision_obs.residual_obs,
                                "rewards": float(reward),
                                "masks": float(0.0 if bool(info.get("env_done", False)) else 1.0),
                                "dones": bool(done or truncated),
                            }
                            with stats.context("replay_insert"):
                                data_store.insert(transition)

                            env_steps += 1
                            episode_steps += 1
                            obs = dict(next_obs)
                            current_residual_obs = next_decision_obs.residual_obs
                            prefetched = next_decision_obs
                            episode_done = bool(done or truncated or env_steps >= int(cfg.env_steps))
                            if episode_done:
                                break
                    else:
                        with stats.context("step_env"):
                            chunk_result = env.step_chunk(action_chunk)

                        raw_chunk = RawChunkRecord.from_step_chunk_result(
                            episode_id=int(episode_id),
                            episode_step_start=int(episode_steps),
                            residual_obs_before_chunk=residual_obs,
                            action_chunk=action_chunk,
                            chunk_result=chunk_result,
                        )

                        if variant == "copy_sync":
                            with stats.context("build_decision_obs"):
                                backfilled_base_actions, backfilled_residual_obs = (
                                    _backfill_post_step_residual_obs(
                                        stats=stats,
                                        metric_prefix="build_decision_obs.backfill",
                                        observations=raw_chunk.post_step_observations,
                                        task_prompt=task_prompt,
                                        policy_client=policy_client,
                                        chunk_horizon=int(cfg.chunk_horizon),
                                        image_keys=tuple(cfg.image_keys),
                                        residual_alpha=float(cfg.residual_alpha),
                                    )
                                )
                                transitions = assemble_chunk_step_transitions(
                                    episode_id=int(raw_chunk.episode_id),
                                    episode_step_start=int(raw_chunk.episode_step_start),
                                    residual_obs_before_chunk=raw_chunk.residual_obs_before_chunk,
                                    executed_actions=raw_chunk.action_chunk,
                                    rewards=raw_chunk.rewards,
                                    dones=raw_chunk.dones,
                                    infos=raw_chunk.infos,
                                    next_residual_observations=backfilled_residual_obs,
                                )

                            with stats.context("replay_insert"):
                                for transition in transitions:
                                    data_store.insert(transition)

                            env_steps += int(raw_chunk.executed_steps)
                            episode_steps += int(raw_chunk.executed_steps)
                            obs = dict(raw_chunk.final_obs)
                            episode_done = bool(
                                raw_chunk.chunk_done
                                or raw_chunk.chunk_truncated
                                or env_steps >= int(cfg.env_steps)
                            )
                            if not episode_done:
                                prefetched = PrefetchedDecisionObs(
                                    base_actions=backfilled_base_actions[-1],
                                    residual_obs=backfilled_residual_obs[-1],
                                )
                            else:
                                prefetched = None
                        elif variant in {"copy_async", "copy_copy"}:
                            next_env_steps = int(env_steps + raw_chunk.executed_steps)
                            expect_tail_handoff = (
                                not bool(raw_chunk.chunk_done or raw_chunk.chunk_truncated)
                                and next_env_steps < int(cfg.env_steps)
                            )
                            with stats.context("build_decision_obs"):
                                last_submitted_chunk_seq = async_coord.submit_chunk(
                                    raw=raw_chunk,
                                    task_prompt=task_prompt,
                                    expect_tail_handoff=expect_tail_handoff,
                                )
                            pending_tail_chunk_seq = (
                                int(last_submitted_chunk_seq) if expect_tail_handoff else None
                            )
                            env_steps += int(raw_chunk.executed_steps)
                            episode_steps += int(raw_chunk.executed_steps)
                            obs = dict(raw_chunk.final_obs)
                            episode_done = bool(
                                raw_chunk.chunk_done
                                or raw_chunk.chunk_truncated
                                or env_steps >= int(cfg.env_steps)
                            )
                            prefetched = None
                            commit_async_chunks()
                            while async_coord.pending_count > int(cfg.max_pending_chunks):
                                with stats.context("commit_replay"):
                                    commit_async_chunks(
                                        block_until_seq=async_coord.next_commit_chunk_seq
                                    )
                        else:
                            raise ValueError(f"Unsupported variant {variant!r}")

            if async_coord is not None and pending_tail_chunk_seq is not None:
                async_coord.finalize_tail_with_fallback(
                    chunk_seq=int(pending_tail_chunk_seq),
                    task_prompt=task_prompt,
                )
                pending_tail_chunk_seq = None
            if async_coord is not None and last_submitted_chunk_seq is not None:
                with stats.context("commit_replay"):
                    commit_async_chunks(block_until_seq=int(last_submitted_chunk_seq))
    finally:
        if async_coord is not None:
            async_coord.close()
        policy_client.close()
        backfill_policy_client.close()

    wall_time_sec = time.perf_counter() - wall_start
    totals, counts, averages = stats.snapshot()
    if int(data_store.insert_count) != int(env_steps):
        raise AssertionError(
            f"{variant}: replay inserts {data_store.insert_count} != env_steps {env_steps}"
        )

    normalized = {
        "wall_per_step_ms": 1000.0 * wall_time_sec / max(1, int(env_steps)),
        "wall_per_chunk_ms": 1000.0 * wall_time_sec / max(1, int(chunk_count)),
        "wall_steps_per_sec": float(env_steps) / max(1e-6, wall_time_sec),
        "wall_chunks_per_sec": float(chunk_count) / max(1e-6, wall_time_sec),
        "sample_actions_per_chunk_ms": 1000.0
        * totals.get("sample_actions", 0.0)
        / max(1, int(chunk_count)),
        "step_env_per_chunk_ms": 1000.0
        * totals.get("step_env", 0.0)
        / max(1, int(chunk_count)),
        "step_env_per_step_ms": 1000.0
        * totals.get("step_env", 0.0)
        / max(1, int(env_steps)),
        "build_decision_obs_per_chunk_ms": 1000.0
        * totals.get("build_decision_obs", 0.0)
        / max(1, int(chunk_count)),
        "build_decision_obs_per_step_ms": 1000.0
        * totals.get("build_decision_obs", 0.0)
        / max(1, int(env_steps)),
        "commit_replay_per_chunk_ms": 1000.0
        * totals.get("commit_replay", 0.0)
        / max(1, int(chunk_count)),
        "replay_insert_per_step_ms": 1000.0
        * totals.get("replay_insert", 0.0)
        / max(1, int(env_steps)),
        "env_wait_per_step_ms": 1000.0
        * totals.get("env.wait", 0.0)
        / max(1, int(env_steps)),
        "target_env_step_ms": 1000.0 / float(cfg.env_hz),
    }
    total_wall_reference = max(1e-9, wall_time_sec)
    top_bottlenecks = []
    for key, total_sec in sorted(
        totals.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]:
        top_bottlenecks.append(
            {
                "key": key,
                "calls": int(counts.get(key, 0)),
                "avg_ms": 1000.0 * float(averages.get(key, 0.0)),
                "total_sec": float(total_sec),
                "share": float(total_sec) / total_wall_reference,
            }
        )

    return VariantResult(
        variant=variant,
        env_steps=int(env_steps),
        chunk_count=int(chunk_count),
        wall_time_sec=float(wall_time_sec),
        inserts=int(data_store.insert_count),
        averages=averages,
        totals=totals,
        counts=counts,
        normalized=normalized,
        top_bottlenecks=top_bottlenecks,
    )


def _print_summary(results: Sequence[VariantResult]) -> None:
    print()
    print("=== LIBERO 30Hz Rollout Benchmark ===")
    print(
        "variant            wall_s   step/s   chunk/s  wall_ms/step  "
        "sample_ms/chunk  step_env_ms/chunk  build_ms/chunk  commit_ms/chunk"
    )
    for result in results:
        normalized = result.normalized
        print(
            f"{result.variant:<17}"
            f"{result.wall_time_sec:>7.2f} "
            f"{normalized['wall_steps_per_sec']:>8.2f} "
            f"{normalized['wall_chunks_per_sec']:>8.2f} "
            f"{normalized['wall_per_step_ms']:>13.2f} "
            f"{normalized['sample_actions_per_chunk_ms']:>16.2f} "
            f"{normalized['step_env_per_chunk_ms']:>18.2f} "
            f"{normalized['build_decision_obs_per_chunk_ms']:>15.2f} "
            f"{normalized['commit_replay_per_chunk_ms']:>16.2f}"
        )
    print()
    for result in results:
        print(f"--- {result.variant} top bottlenecks ---")
        for item in result.top_bottlenecks:
            print(
                f"{item['key']:<44}"
                f"calls={item['calls']:<4} "
                f"avg_ms={item['avg_ms']:<10.3f} "
                f"total_s={item['total_sec']:<9.3f} "
                f"share={item['share']:.2%}"
            )
        print()


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Synthetic 30Hz LIBERO rollout benchmark",
    )
    parser.add_argument(
        "--env-hz",
        type=float,
        default=30.0,
        help="Target synthetic environment frequency",
    )
    parser.add_argument(
        "--env-steps",
        type=int,
        default=150,
        help="Total rollout steps to benchmark",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=10_000,
        help="Synthetic episode length before done=true",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="LIBERO action dimension",
    )
    parser.add_argument(
        "--chunk-horizon",
        type=int,
        default=5,
        help="Chunk horizon used by both copy and baseline variants",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.1,
        help="Residual alpha used to build residual observations",
    )
    parser.add_argument(
        "--policy-infer-ms",
        type=float,
        default=8.0,
        help="Simulated main policy inference latency in milliseconds",
    )
    parser.add_argument(
        "--backfill-policy-infer-ms",
        type=float,
        default=None,
        help="Simulated backfill-policy inference latency in milliseconds. "
        "Defaults to --policy-infer-ms.",
    )
    parser.add_argument(
        "--residual-sample-ms",
        type=float,
        default=1.0,
        help="Simulated residual policy sample latency in milliseconds",
    )
    parser.add_argument(
        "--replay-insert-ms",
        type=float,
        default=0.05,
        help="Simulated replay insert latency per transition in milliseconds",
    )
    parser.add_argument(
        "--step-rpc-ms",
        type=float,
        default=0.4,
        help="Per-step RPC overhead for baseline step env calls in milliseconds",
    )
    parser.add_argument(
        "--step-chunk-rpc-ms",
        type=float,
        default=0.4,
        help="Per-chunk RPC overhead for step_chunk calls in milliseconds",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Synthetic source image size before LIBERO preprocessing",
    )
    parser.add_argument(
        "--max-pending-chunks",
        type=int,
        default=2,
        help="Maximum async pending chunks before blocking commit",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("baseline_step", "copy_sync", "copy_async", "copy_copy"),
        choices=("baseline_step", "copy_sync", "copy_async", "copy_copy"),
        help="Variants to benchmark",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()
    backfill_policy_infer_ms = (
        float(args.policy_infer_ms)
        if args.backfill_policy_infer_ms is None
        else float(args.backfill_policy_infer_ms)
    )
    return BenchmarkConfig(
        env_hz=float(args.env_hz),
        env_steps=int(args.env_steps),
        episode_length=int(args.episode_length),
        action_dim=int(args.action_dim),
        chunk_horizon=int(args.chunk_horizon),
        residual_alpha=float(args.residual_alpha),
        image_keys=("image", "wrist_image"),
        policy_infer_ms=float(args.policy_infer_ms),
        backfill_policy_infer_ms=float(backfill_policy_infer_ms),
        residual_sample_ms=float(args.residual_sample_ms),
        replay_insert_ms=float(args.replay_insert_ms),
        step_rpc_ms=float(args.step_rpc_ms),
        step_chunk_rpc_ms=float(args.step_chunk_rpc_ms),
        image_size=int(args.image_size),
        max_pending_chunks=int(args.max_pending_chunks),
        variants=tuple(str(value) for value in args.variants),
        json_out=args.json_out,
    )


def main() -> None:
    cfg = _parse_args()
    results = [_run_variant(variant, cfg) for variant in cfg.variants]
    _print_summary(results)
    if cfg.json_out is not None:
        payload = {
            "config": asdict(cfg),
            "results": [asdict(result) for result in results],
        }
        output_path = Path(cfg.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"wrote benchmark json to {output_path}")


if __name__ == "__main__":
    main()
