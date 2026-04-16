from __future__ import annotations

"""Benchmark learner update speed with fake StepWindowReplay data.

This script intentionally lives outside the normal pytest suite because it is a
GPU benchmark, not a deterministic unit test. It builds the real AgiBot DRQ
learner architecture, fills a StepWindowReplay buffer with fake transitions,
then runs the same update pattern used by the learner loop:

    (critic_actor_ratio - 1) * update_critics + update_high_utd

Usage from the repository root:

    conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
      python test/benchmark_learner_update_speed.py
"""

import argparse
import gc
import json
import statistics
import sys
import threading
import time
import types
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for path in (REPO_ROOT, SERL_LAUNCHER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from examples.agibot_real.config import parse_train_cfg
from examples.agibot_real.residual_observation import (
    build_chunk_residual_observation_space,
    build_chunk_residual_sample_obs,
)
from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.agents.continuous.drq import _unpack
from serl_launcher.agents.continuous.sac import _to_torch
from serl_launcher.agents.continuous.sac import _index_batch
from serl_launcher.agents.continuous.sac import _split_batch
from serl_launcher.agents.continuous.sac import _tree_mean
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_actor_network_payload
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import reshape_chunk_batch_for_training
from serl_launcher.residual.chunk_window_replay import sample_mixed_training_batch
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.vision.data_augmentations import random_crop


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_cfg(
    *,
    config_path: Path,
    mixed_precision: bool,
    random_init: bool,
    freeze_backbone: bool,
) -> Any:
    cfg_raw = OmegaConf.load(config_path)
    cfg_raw.runtime.role = "learner"
    cfg_raw.training.mixed_precision.enabled = bool(mixed_precision)
    cfg_raw.training.mixed_precision.dtype = "bfloat16"
    if bool(random_init):
        cfg_raw.encoder.resnet.pretrained = False
    if cfg_raw.encoder.resnet is not None:
        cfg_raw.encoder.resnet.freeze_backbone = bool(freeze_backbone)
    return parse_train_cfg(cfg_raw)


def _make_sample_obs_and_spec(cfg: Any) -> tuple[dict[str, np.ndarray], ResidualActionSpec]:
    residual_action_spec = ResidualActionSpec.from_cfg(
        cfg,
        action_dim=int(cfg.env.action_dim),
    )
    sample_obs = build_chunk_residual_sample_obs(
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        image_keys=tuple(cfg.obs.image_keys),
    )
    return sample_obs, residual_action_spec


def _make_agent(
    *,
    cfg: Any,
    sample_obs: dict[str, np.ndarray],
    residual_action_spec: ResidualActionSpec,
) -> Any:
    return create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=tuple(cfg.obs.image_keys),
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )


def _fake_obs(
    *,
    rng: np.random.Generator,
    sample_obs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key, value in sample_obs.items():
        value_array = np.asarray(value)
        if value_array.dtype == np.uint8:
            obs[key] = rng.integers(
                0,
                256,
                size=value_array.shape,
                dtype=np.uint8,
            )
        elif key == "alpha":
            obs[key] = np.zeros(value_array.shape, dtype=np.float32)
        else:
            obs[key] = rng.standard_normal(size=value_array.shape).astype(np.float32)
    return obs


def _build_fake_replay(
    *,
    cfg: Any,
    sample_obs: dict[str, np.ndarray],
    num_steps: int,
    episode_length: int,
    seed: int,
) -> Any:
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=tuple(cfg.obs.image_keys),
    )
    replay_buffer = create_chunk_replay_buffer(
        observation_space=observation_space,
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        discount=float(cfg.sac.discount),
        image_keys=tuple(cfg.obs.image_keys),
        capacity=max(int(num_steps) + 32, int(cfg.replay.batch_size) + 32),
    )
    rng = np.random.default_rng(int(seed))
    action_dim = int(cfg.env.action_dim)
    episode_length = max(1, int(episode_length))

    for step in range(int(num_steps)):
        episode_step = int(step % episode_length)
        done = bool(episode_step == episode_length - 1)
        replay_buffer.insert(
            {
                "episode_id": int(step // episode_length),
                "episode_step": episode_step,
                "observations": _fake_obs(rng=rng, sample_obs=sample_obs),
                "actions": rng.standard_normal(size=(action_dim,)).astype(np.float32),
                "next_observations": _fake_obs(rng=rng, sample_obs=sample_obs),
                "rewards": float(rng.standard_normal()),
                "masks": float(0.0 if done else 1.0),
                "dones": done,
            }
        )
    return replay_buffer


def _legacy_update(
    self,
    batch,
    *,
    pmap_axis: str | None = None,
    networks_to_update=frozenset({"actor", "critic", "temperature"}),
):
    """Original SACAgent.update behavior before freezing critic in actor update."""

    del pmap_axis
    batch = _to_torch(batch, self.device)
    info = {}

    if "critic" in networks_to_update:
        self.state.zero_grad(["critic"])
        with self._autocast_context():
            critic_loss, critic_info = self.critic_loss_fn(batch)
        critic_loss.backward()
        self.state.optimizer_step("critic")
        self.state.target_update(self.config["soft_target_update_rate"])
        info.update(critic_info)

    if "actor" in networks_to_update:
        self.state.zero_grad(["actor"])
        with self._autocast_context():
            actor_loss, actor_info = self.policy_loss_fn(batch)
        actor_loss.backward()
        self.state.optimizer_step("actor")
        info.update(actor_info)

    if "temperature" in networks_to_update:
        self.state.zero_grad(["temperature"])
        with self._autocast_context():
            temperature_loss, temperature_info = self.temperature_loss_fn(batch)
        temperature_loss.backward()
        self.state.optimizer_step("temperature")
        info.update(temperature_info)

    self.state.step += 1
    info.update(self.state.lr_info())
    return self, info


def _compile_agent_modules(
    agent: Any,
    *,
    target: str,
    mode: str,
    backend: str,
    fullgraph: bool,
    dynamic: bool,
) -> None:
    target = str(target)
    if target == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    compile_kwargs = {
        "backend": str(backend),
        "mode": str(mode),
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
    }
    if target not in {"critic", "actor_critic"}:
        raise ValueError(f"Unsupported torch_compile target: {target}")

    agent.state.modules["critic"] = torch.compile(
        agent.state.modules["critic"],
        **compile_kwargs,
    )
    if "critic" in agent.state.target_modules:
        agent.state.target_modules["critic"] = torch.compile(
            agent.state.target_modules["critic"],
            **compile_kwargs,
        )
    if target == "actor_critic":
        agent.state.modules["actor"] = torch.compile(
            agent.state.modules["actor"],
            **compile_kwargs,
        )


def _batch_copy_at_indices(value: Any, indices: np.ndarray) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value[indices], copy=True)
    if isinstance(value, dict):
        return {key: _batch_copy_at_indices(item, indices) for key, item in value.items()}
    raise TypeError(f"Unsupported replay value type: {type(value)}")


def _select_batch_slice(value: Any, start: int, end: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _select_batch_slice(item, start, end)
            for key, item in value.items()
        }
    array = np.asarray(value)
    return array[int(start) : int(end)]


def _split_batch_tree(value: Any, *, num_batches: int, batch_size: int) -> list[Any]:
    return [
        _select_batch_slice(
            value,
            int(index) * int(batch_size),
            (int(index) + 1) * int(batch_size),
        )
        for index in range(int(num_batches))
    ]


def _trim_actor_temperature_batch(batch: dict[str, Any]) -> dict[str, Any]:
    trimmed = {"observations": batch["observations"]}
    if "action_mask" in batch:
        trimmed["action_mask"] = batch["action_mask"]
    return trimmed


def _update_high_utd_trim_actor_batch(
    self,
    batch,
    *,
    utd_ratio: int,
    pmap_axis: str | None = None,
):
    """Benchmark-only DrQ update_high_utd variant with smaller actor/temp batch."""

    del pmap_axis
    if self.config["image_keys"][0] not in batch["next_observations"]:
        batch = _unpack(batch)

    batch_t = _to_torch(batch, self.device)
    batch_t["observations"] = self.data_augmentation_fn(batch_t["observations"])
    batch_t["next_observations"] = self.data_augmentation_fn(
        batch_t["next_observations"]
    )
    minibatches = _split_batch(batch_t, int(utd_ratio))

    critic_infos = []
    for index in range(int(utd_ratio)):
        minibatch = _index_batch(minibatches, index)
        self, info = self.update(
            minibatch,
            networks_to_update=frozenset({"critic"}),
        )
        critic_infos.append(info)

    _, actor_temp_info = self.update(
        _trim_actor_temperature_batch(batch_t),
        networks_to_update=frozenset({"actor", "temperature"}),
    )
    info = _tree_mean(critic_infos) if critic_infos else {}
    info.update(actor_temp_info)
    return self, info


def _fast_step_window_sample_impl(
    replay_buffer: Any,
    batch_size: int,
    keys: Any = None,
    indx: np.ndarray | None = None,
) -> dict[str, Any]:
    """Vectorized benchmark-only equivalent of MemoryEfficientStepWindow sample."""

    replay_buffer._cleanup_stale_candidates()
    if indx is None:
        if not replay_buffer._candidate_start_step_ids:
            raise RuntimeError(
                "StepWindowReplayBuffer has no eligible window starts. "
                f"(num_steps={replay_buffer.num_steps}, "
                f"num_windows={replay_buffer.num_windows}, "
                f"sample_stride={replay_buffer.sample_stride})"
            )
        candidates = np.asarray(
            replay_buffer._candidate_start_step_ids,
            dtype=np.int64,
        )
        sampled_start_ids = replay_buffer.np_random.choice(
            candidates,
            size=int(batch_size),
            replace=True,
        )
    else:
        sampled_start_ids = np.asarray(indx, dtype=np.int64).reshape(-1)
        if int(sampled_start_ids.shape[0]) != int(batch_size):
            raise ValueError(
                "indx length must equal batch_size, got "
                f"{sampled_start_ids.shape[0]} != {batch_size}"
            )

    sampled_start_ids = sampled_start_ids.astype(np.int64, copy=False)
    batch_size = int(sampled_start_ids.shape[0])
    window_size = int(replay_buffer.window_size)
    offsets = np.arange(window_size, dtype=np.int64)
    step_ids = sampled_start_ids[:, None] + offsets[None, :]
    indices = np.mod(step_ids, int(replay_buffer._capacity)).astype(np.int64)

    min_active = int(replay_buffer._min_active_step_id())
    active = (
        (step_ids >= min_active)
        & (step_ids < int(replay_buffer._insert_count))
        & (replay_buffer._step_ids[indices] == step_ids)
    )
    start_indices = np.mod(
        sampled_start_ids,
        int(replay_buffer._capacity),
    ).astype(np.int64)
    start_episode_ids = replay_buffer._episode_ids[start_indices]
    start_episode_steps = replay_buffer._episode_steps[start_indices]
    same_episode = replay_buffer._episode_ids[indices] == start_episode_ids[:, None]
    sequential_step = (
        replay_buffer._episode_steps[indices]
        == (start_episode_steps[:, None] + offsets[None, :])
    )
    structural_valid = active & same_episode & sequential_step
    prefix_valid = np.cumprod(structural_valid.astype(np.int8), axis=1).astype(bool)
    done_flags = replay_buffer.dataset_dict["dones"][indices] & prefix_valid
    done_seen_before = np.concatenate(
        [
            np.zeros((batch_size, 1), dtype=bool),
            np.cumsum(done_flags, axis=1)[:, :-1] > 0,
        ],
        axis=1,
    )
    valid = prefix_valid & ~done_seen_before
    window_steps = valid.sum(axis=1).astype(np.int32)
    if np.any(window_steps <= 0):
        raise RuntimeError("fast step-window sample produced an empty window")

    step_actions = replay_buffer.dataset_dict["actions"]
    gathered_actions = step_actions[indices]
    valid_action_shape = valid.reshape(
        batch_size,
        window_size,
        *([1] * len(replay_buffer._step_action_shape)),
    )
    action_window = np.where(
        valid_action_shape,
        gathered_actions,
        np.zeros((), dtype=step_actions.dtype),
    ).astype(step_actions.dtype, copy=False)
    action_mask = np.broadcast_to(valid_action_shape, action_window.shape).astype(
        np.float32,
        copy=True,
    )

    discounts = np.power(float(replay_buffer.discount), offsets).astype(np.float64)
    rewards = replay_buffer.dataset_dict["rewards"][indices].astype(np.float64)
    discounted_reward = (rewards * valid * discounts[None, :]).sum(axis=1).astype(
        np.float32,
    )

    last_offsets = window_steps.astype(np.int64) - 1
    batch_indices = np.arange(batch_size, dtype=np.int64)
    last_step_ids = step_ids[batch_indices, last_offsets]
    last_indices = indices[batch_indices, last_offsets]
    last_masks = replay_buffer.dataset_dict["masks"][last_indices].astype(np.float32)
    masks = (
        np.power(
            float(replay_buffer.discount),
            np.maximum(0, window_steps.astype(np.int64) - 1),
        ).astype(np.float32)
        * last_masks
    ).astype(np.float32)
    dones = np.any(replay_buffer.dataset_dict["dones"][indices] & valid, axis=1)

    next_observations = _batch_copy_at_indices(
        replay_buffer.dataset_dict["next_observations"],
        last_indices,
    )
    for pixel_key in replay_buffer.pixel_keys:
        explicit_mask = replay_buffer._has_explicit_next_pixels[last_indices]
        pixel_space = replay_buffer._pixel_spaces[pixel_key]
        pixel_batch = np.empty(
            (batch_size, *pixel_space.shape),
            dtype=pixel_space.dtype,
        )
        if np.any(explicit_mask):
            pixel_batch[explicit_mask] = replay_buffer._explicit_next_pixels[
                pixel_key
            ][last_indices[explicit_mask]]
        if np.any(~explicit_mask):
            next_step_ids = last_step_ids[~explicit_mask] + 1
            next_indices = np.mod(
                next_step_ids,
                int(replay_buffer._capacity),
            ).astype(np.int64)
            pixel_batch[~explicit_mask] = replay_buffer.dataset_dict["observations"][
                pixel_key
            ][next_indices]
        next_observations[pixel_key] = np.array(pixel_batch, copy=True)

    batch = {
        "observations": _batch_copy_at_indices(
            replay_buffer.dataset_dict["observations"],
            start_indices,
        ),
        "actions": np.array(action_window, copy=True),
        "action_mask": action_mask,
        "next_observations": next_observations,
        "rewards": discounted_reward,
        "masks": masks,
        "dones": dones.astype(bool, copy=False),
        "window_steps": window_steps,
    }
    if keys is None:
        return batch
    selected_keys = list(keys)
    return {key: batch[key] for key in selected_keys}


def _fast_step_window_sample(
    self,
    batch_size: int,
    keys: Any = None,
    indx: np.ndarray | None = None,
):
    with self._lock:
        return _fast_step_window_sample_impl(
            self,
            batch_size=batch_size,
            keys=keys,
            indx=indx,
        )


def _compare_batch_trees(left: Any, right: Any, *, path: str = "batch") -> None:
    if isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            raise AssertionError(f"{path} keys differ: {left.keys()} != {right.keys()}")
        for key in left:
            _compare_batch_trees(left[key], right[key], path=f"{path}.{key}")
        return

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise AssertionError(
            f"{path} shape differs: {left_array.shape} != {right_array.shape}"
        )
    if left_array.dtype != right_array.dtype:
        raise AssertionError(
            f"{path} dtype differs: {left_array.dtype} != {right_array.dtype}"
        )
    if np.issubdtype(left_array.dtype, np.floating):
        np.testing.assert_allclose(
            left_array,
            right_array,
            rtol=1e-6,
            atol=1e-6,
            err_msg=path,
        )
    else:
        np.testing.assert_array_equal(left_array, right_array, err_msg=path)


def _validate_fast_step_window_sample(replay_buffer: Any, *, batch_size: int) -> None:
    replay_buffer._cleanup_stale_candidates()
    candidates = np.asarray(replay_buffer._candidate_start_step_ids, dtype=np.int64)
    if int(candidates.shape[0]) <= 0:
        raise RuntimeError("Cannot validate fast sample without candidate windows")
    sample_count = min(int(batch_size), int(candidates.shape[0]))
    positions = np.linspace(
        0,
        int(candidates.shape[0]) - 1,
        num=sample_count,
        dtype=np.int64,
    )
    start_ids = candidates[positions]
    legacy = replay_buffer.sample(sample_count, indx=start_ids)
    fast = _fast_step_window_sample_impl(
        replay_buffer,
        batch_size=sample_count,
        indx=start_ids,
    )
    _compare_batch_trees(legacy, fast)


class _ReplayBatchPrefetcher:
    def __init__(self, sample_fn, *, queue_size: int):
        self._sample_fn = sample_fn
        self._queue: Queue[tuple[bool, Any]] = Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="learner-benchmark-replay-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = (True, self._sample_fn())
            except BaseException as exc:  # noqa: BLE001
                item = (False, exc)
            while not self._stop_event.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except Full:
                    continue

    def get(self):
        while True:
            ok, value = self._queue.get()
            if ok:
                return value
            raise value

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)


def _sample_training_batch(*, cfg: Any, replay_buffer: Any):
    return sample_mixed_training_batch(
        online_replay_buffer=replay_buffer,
        offline_replay_buffer=None,
        batch_size=int(cfg.replay.batch_size),
        offline_ratio=0.0,
    )


def _sample_big_training_batches(
    *,
    cfg: Any,
    replay_buffer: Any,
    num_batches: int,
) -> list[dict[str, Any]]:
    batch_size = int(cfg.replay.batch_size)
    big_batch, _ = sample_mixed_training_batch(
        online_replay_buffer=replay_buffer,
        offline_replay_buffer=None,
        batch_size=int(batch_size * int(num_batches)),
        offline_ratio=0.0,
    )
    return _split_batch_tree(
        big_batch,
        num_batches=int(num_batches),
        batch_size=int(batch_size),
    )


def _run_outer_update(
    *,
    agent: Any,
    cfg: Any,
    replay_buffer: Any,
) -> dict[str, float]:
    critic_actor_ratio = max(1, int(cfg.training.critic_actor_ratio))
    sample_times: list[float] = []
    critic_times: list[float] = []

    outer_start = time.perf_counter()
    for _ in range(max(0, critic_actor_ratio - 1)):
        _sync_cuda()
        sample_start = time.perf_counter()
        batch, _ = _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer)
        _sync_cuda()
        sample_end = time.perf_counter()
        agent, _ = agent.update_critics(batch)
        _sync_cuda()
        critic_end = time.perf_counter()
        sample_times.append(sample_end - sample_start)
        critic_times.append(critic_end - sample_end)

    _sync_cuda()
    sample_start = time.perf_counter()
    batch, _ = _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer)
    _sync_cuda()
    sample_end = time.perf_counter()
    agent, _ = agent.update_high_utd(batch, utd_ratio=int(cfg.sac.utd_ratio))
    _sync_cuda()
    train_end = time.perf_counter()

    sample_times.append(sample_end - sample_start)
    return {
        "outer_s": train_end - outer_start,
        "sample_s": float(sum(sample_times)),
        "critic_only_s": float(sum(critic_times)),
        "high_utd_s": train_end - sample_end,
    }


def _run_outer_update_big_batch(
    *,
    agent: Any,
    cfg: Any,
    replay_buffer: Any,
) -> dict[str, float]:
    critic_actor_ratio = max(1, int(cfg.training.critic_actor_ratio))
    critic_times: list[float] = []

    outer_start = time.perf_counter()
    _sync_cuda()
    sample_start = time.perf_counter()
    batches = _sample_big_training_batches(
        cfg=cfg,
        replay_buffer=replay_buffer,
        num_batches=int(critic_actor_ratio),
    )
    _sync_cuda()
    sample_end = time.perf_counter()

    train_cursor = sample_end
    for batch in batches[: max(0, critic_actor_ratio - 1)]:
        agent, _ = agent.update_critics(batch)
        _sync_cuda()
        critic_end = time.perf_counter()
        critic_times.append(critic_end - train_cursor)
        train_cursor = critic_end

    agent, _ = agent.update_high_utd(
        batches[-1],
        utd_ratio=int(cfg.sac.utd_ratio),
    )
    _sync_cuda()
    train_end = time.perf_counter()

    return {
        "outer_s": train_end - outer_start,
        "sample_s": sample_end - sample_start,
        "critic_only_s": float(sum(critic_times)),
        "high_utd_s": train_end - train_cursor,
    }


def _run_outer_update_prefetch(
    *,
    agent: Any,
    cfg: Any,
    prefetcher: _ReplayBatchPrefetcher,
) -> dict[str, float]:
    critic_actor_ratio = max(1, int(cfg.training.critic_actor_ratio))
    wait_times: list[float] = []
    critic_times: list[float] = []

    outer_start = time.perf_counter()
    for _ in range(max(0, critic_actor_ratio - 1)):
        _sync_cuda()
        wait_start = time.perf_counter()
        batch, _ = prefetcher.get()
        _sync_cuda()
        wait_end = time.perf_counter()
        agent, _ = agent.update_critics(batch)
        _sync_cuda()
        critic_end = time.perf_counter()
        wait_times.append(wait_end - wait_start)
        critic_times.append(critic_end - wait_end)

    _sync_cuda()
    wait_start = time.perf_counter()
    batch, _ = prefetcher.get()
    _sync_cuda()
    wait_end = time.perf_counter()
    agent, _ = agent.update_high_utd(batch, utd_ratio=int(cfg.sac.utd_ratio))
    _sync_cuda()
    train_end = time.perf_counter()

    wait_times.append(wait_end - wait_start)
    return {
        "outer_s": train_end - outer_start,
        "sample_s": float(sum(wait_times)),
        "critic_only_s": float(sum(critic_times)),
        "high_utd_s": train_end - wait_end,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def _tree_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return int(sum(_tree_tensor_bytes(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return int(sum(_tree_tensor_bytes(item) for item in value))
    return 0


def _time_call(fn, *, iterations: int) -> tuple[float, Any]:
    times: list[float] = []
    last_value = None
    for _ in range(int(iterations)):
        _sync_cuda()
        start = time.perf_counter()
        last_value = fn()
        _sync_cuda()
        times.append(time.perf_counter() - start)
    return _mean(times), last_value


def _legacy_batched_random_crop(
    img,
    rng=None,
    *,
    padding: int,
    num_batch_dims: int = 1,
):
    img = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
    original_shape = img.shape
    flat = img.reshape(-1, *img.shape[num_batch_dims:])
    generator = torch.Generator(device=img.device.type)
    if isinstance(rng, int):
        generator.manual_seed(int(rng))
    else:
        generator.seed()

    crops = []
    for i in range(flat.shape[0]):
        sample_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                generator=generator,
                device=img.device,
            )
        )
        crops.append(random_crop(flat[i], sample_seed, padding=padding))
    return torch.stack(crops, dim=0).reshape(original_shape)


def _benchmark_random_crop(*, iterations: int, warmup: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    image = torch.randint(
        0,
        256,
        (128, 1, 224, 224, 3),
        device=torch.device("cuda"),
        dtype=torch.uint8,
    )

    def _measure(fn) -> float:
        times: list[float] = []
        for index in range(int(warmup) + int(iterations)):
            _sync_cuda()
            start = time.perf_counter()
            out = fn(image)
            _sync_cuda()
            elapsed = time.perf_counter() - start
            if index >= int(warmup):
                times.append(elapsed)
            del out
        return _mean(times)

    legacy_s = _measure(
        lambda value: _legacy_batched_random_crop(
            value,
            padding=4,
            num_batch_dims=2,
        )
    )
    vectorized_s = _measure(
        lambda value: batched_random_crop(
            value,
            padding=4,
            num_batch_dims=2,
        )
    )
    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "shape": list(image.shape),
        "legacy_loop_mean_s": float(legacy_s),
        "vectorized_mean_s": float(vectorized_s),
        "speedup_pct": (
            float((legacy_s - vectorized_s) / legacy_s * 100.0)
            if legacy_s > 0
            else 0.0
        ),
    }


def _time_cpu_call(fn, *, iterations: int, warmup: int) -> float:
    times: list[float] = []
    for index in range(int(warmup) + int(iterations)):
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
        if index >= int(warmup):
            times.append(elapsed)
        del value
    return _mean(times)


def _set_fast_replay_sample(replay_buffer: Any) -> tuple[bool, Any]:
    had_instance_sample = "sample" in getattr(replay_buffer, "__dict__", {})
    previous_sample = getattr(replay_buffer, "sample", None)
    replay_buffer.sample = types.MethodType(_fast_step_window_sample, replay_buffer)
    return had_instance_sample, previous_sample


def _restore_replay_sample(
    replay_buffer: Any,
    *,
    had_instance_sample: bool,
    previous_sample: Any,
) -> None:
    if had_instance_sample:
        replay_buffer.sample = previous_sample
    else:
        try:
            delattr(replay_buffer, "sample")
        except AttributeError:
            pass


def _benchmark_replay_sampling(
    *,
    cfg: Any,
    replay_buffer: Any,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    critic_actor_ratio = max(1, int(cfg.training.critic_actor_ratio))
    batch_size = int(cfg.replay.batch_size)

    legacy_single_s = _time_cpu_call(
        lambda: _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer),
        iterations=int(iterations),
        warmup=int(warmup),
    )
    legacy_outer_separate_s = _time_cpu_call(
        lambda: [
            _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer)
            for _ in range(int(critic_actor_ratio))
        ],
        iterations=int(iterations),
        warmup=int(warmup),
    )
    legacy_big_s = _time_cpu_call(
        lambda: _sample_big_training_batches(
            cfg=cfg,
            replay_buffer=replay_buffer,
            num_batches=int(critic_actor_ratio),
        ),
        iterations=int(iterations),
        warmup=int(warmup),
    )

    had_instance_sample, previous_sample = _set_fast_replay_sample(replay_buffer)
    try:
        fast_single_s = _time_cpu_call(
            lambda: _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer),
            iterations=int(iterations),
            warmup=int(warmup),
        )
        fast_big_s = _time_cpu_call(
            lambda: _sample_big_training_batches(
                cfg=cfg,
                replay_buffer=replay_buffer,
                num_batches=int(critic_actor_ratio),
            ),
            iterations=int(iterations),
            warmup=int(warmup),
        )
    finally:
        _restore_replay_sample(
            replay_buffer,
            had_instance_sample=had_instance_sample,
            previous_sample=previous_sample,
        )

    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "batch_size": int(batch_size),
        "critic_actor_ratio": int(critic_actor_ratio),
        "legacy_single_batch_mean_s": float(legacy_single_s),
        "legacy_outer_separate_mean_s": float(legacy_outer_separate_s),
        "legacy_big_batch_mean_s": float(legacy_big_s),
        "fast_single_batch_mean_s": float(fast_single_s),
        "fast_big_batch_mean_s": float(fast_big_s),
        "big_batch_speedup_pct": (
            float((legacy_outer_separate_s - legacy_big_s) / legacy_outer_separate_s * 100.0)
            if legacy_outer_separate_s > 0
            else 0.0
        ),
        "fast_single_speedup_pct": (
            float((legacy_single_s - fast_single_s) / legacy_single_s * 100.0)
            if legacy_single_s > 0
            else 0.0
        ),
        "fast_big_vs_legacy_separate_speedup_pct": (
            float((legacy_outer_separate_s - fast_big_s) / legacy_outer_separate_s * 100.0)
            if legacy_outer_separate_s > 0
            else 0.0
        ),
    }


def _benchmark_payloads(
    *,
    cfg: Any,
    sample_obs: dict[str, np.ndarray],
    residual_action_spec: ResidualActionSpec,
    replay_buffer: Any,
    iterations: int,
) -> dict[str, Any]:
    agent = _make_agent(
        cfg=cfg,
        sample_obs=sample_obs,
        residual_action_spec=residual_action_spec,
    )
    _run_outer_update(agent=agent, cfg=cfg, replay_buffer=replay_buffer)

    full_snapshot_s, full_payload = _time_call(
        lambda: snapshot_agent_checkpoint_payload(agent, step=int(agent.state.step)),
        iterations=int(iterations),
    )
    actor_snapshot_s, actor_payload = _time_call(
        lambda: snapshot_actor_network_payload(agent, step=int(agent.state.step)),
        iterations=int(iterations),
    )

    full_actor = _make_agent(
        cfg=cfg,
        sample_obs=sample_obs,
        residual_action_spec=residual_action_spec,
    )
    full_apply_s, _ = _time_call(
        lambda: apply_checkpoint_payload_to_agent(
            full_actor,
            full_payload,
            load_optimizers=False,
        ),
        iterations=int(iterations),
    )

    actor_only_actor = _make_agent(
        cfg=cfg,
        sample_obs=sample_obs,
        residual_action_spec=residual_action_spec,
    )
    actor_apply_s, _ = _time_call(
        lambda: apply_checkpoint_payload_to_agent(
            actor_only_actor,
            actor_payload,
            load_optimizers=False,
        ),
        iterations=int(iterations),
    )

    full_payload_mb = float(_tree_tensor_bytes(full_payload) / 1024 / 1024)
    actor_payload_mb = float(_tree_tensor_bytes(actor_payload) / 1024 / 1024)
    del agent, full_actor, actor_only_actor, full_payload, actor_payload
    gc.collect()
    return {
        "iterations": int(iterations),
        "full_snapshot_mean_s": float(full_snapshot_s),
        "actor_only_snapshot_mean_s": float(actor_snapshot_s),
        "snapshot_speedup_pct": (
            float((full_snapshot_s - actor_snapshot_s) / full_snapshot_s * 100.0)
            if full_snapshot_s > 0
            else 0.0
        ),
        "full_apply_mean_s": float(full_apply_s),
        "actor_only_apply_mean_s": float(actor_apply_s),
        "apply_speedup_pct": (
            float((full_apply_s - actor_apply_s) / full_apply_s * 100.0)
            if full_apply_s > 0
            else 0.0
        ),
        "full_payload_tensor_mb": full_payload_mb,
        "actor_only_payload_tensor_mb": actor_payload_mb,
        "payload_reduction_pct": (
            float((full_payload_mb - actor_payload_mb) / full_payload_mb * 100.0)
            if full_payload_mb > 0
            else 0.0
        ),
    }


def _benchmark_scenario(
    *,
    name: str,
    cfg: Any,
    sample_obs: dict[str, np.ndarray],
    residual_action_spec: ResidualActionSpec,
    replay_buffer: Any,
    use_legacy_actor_update: bool,
    use_replay_prefetch: bool,
    use_big_batch_sample: bool,
    use_fast_step_window_sample: bool,
    trim_actor_temperature_batch: bool,
    torch_compile_target: str,
    torch_compile_mode: str,
    torch_compile_backend: str,
    torch_compile_fullgraph: bool,
    torch_compile_dynamic: bool,
    prefetch_queue_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    agent = _make_agent(
        cfg=cfg,
        sample_obs=sample_obs,
        residual_action_spec=residual_action_spec,
    )
    if use_legacy_actor_update:
        agent.update = types.MethodType(_legacy_update, agent)
    if trim_actor_temperature_batch:
        agent.update_high_utd = types.MethodType(
            _update_high_utd_trim_actor_batch,
            agent,
        )
    _compile_agent_modules(
        agent,
        target=str(torch_compile_target),
        mode=str(torch_compile_mode),
        backend=str(torch_compile_backend),
        fullgraph=bool(torch_compile_fullgraph),
        dynamic=bool(torch_compile_dynamic),
    )

    if bool(use_replay_prefetch) and bool(use_big_batch_sample):
        raise ValueError("replay prefetch and big-batch sample are separate scenarios")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    outer_times: list[float] = []
    sample_times: list[float] = []
    critic_only_times: list[float] = []
    high_utd_times: list[float] = []

    total_runs = int(warmup) + int(iterations)
    prefetcher = None
    sample_patch_state = None
    try:
        if use_fast_step_window_sample:
            sample_patch_state = _set_fast_replay_sample(replay_buffer)
        if use_replay_prefetch:
            prefetcher = _ReplayBatchPrefetcher(
                lambda: _sample_training_batch(cfg=cfg, replay_buffer=replay_buffer),
                queue_size=int(prefetch_queue_size),
            )
        for index in range(total_runs):
            if prefetcher is not None:
                timings = _run_outer_update_prefetch(
                    agent=agent,
                    cfg=cfg,
                    prefetcher=prefetcher,
                )
            elif use_big_batch_sample:
                timings = _run_outer_update_big_batch(
                    agent=agent,
                    cfg=cfg,
                    replay_buffer=replay_buffer,
                )
            else:
                timings = _run_outer_update(
                    agent=agent,
                    cfg=cfg,
                    replay_buffer=replay_buffer,
                )
            if index < int(warmup):
                continue
            outer_times.append(timings["outer_s"])
            sample_times.append(timings["sample_s"])
            critic_only_times.append(timings["critic_only_s"])
            high_utd_times.append(timings["high_utd_s"])
    finally:
        if prefetcher is not None:
            prefetcher.close()
        if sample_patch_state is not None:
            had_instance_sample, previous_sample = sample_patch_state
            _restore_replay_sample(
                replay_buffer,
                had_instance_sample=had_instance_sample,
                previous_sample=previous_sample,
            )

    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    outer_mean = _mean(outer_times)
    return {
        "name": name,
        "mixed_precision": bool(cfg.training.mixed_precision.enabled),
        "freeze_backbone": bool(cfg.encoder.resnet.freeze_backbone)
        if cfg.encoder.resnet is not None
        else False,
        "legacy_actor_update": bool(use_legacy_actor_update),
        "replay_prefetch": bool(use_replay_prefetch),
        "big_batch_sample": bool(use_big_batch_sample),
        "fast_step_window_sample": bool(use_fast_step_window_sample),
        "trim_actor_temperature_batch": bool(trim_actor_temperature_batch),
        "torch_compile_target": str(torch_compile_target),
        "torch_compile_mode": str(torch_compile_mode),
        "torch_compile_backend": str(torch_compile_backend),
        "torch_compile_fullgraph": bool(torch_compile_fullgraph),
        "torch_compile_dynamic": bool(torch_compile_dynamic),
        "prefetch_queue_size": int(prefetch_queue_size)
        if use_replay_prefetch
        else 0,
        "iterations": int(iterations),
        "warmup": int(warmup),
        "outer_mean_s": outer_mean,
        "outer_stdev_s": _stdev(outer_times),
        "updates_per_sec": float(1.0 / outer_mean) if outer_mean > 0 else 0.0,
        "sample_total_mean_s": _mean(sample_times),
        "critic_only_total_mean_s": _mean(critic_only_times),
        "high_utd_mean_s": _mean(high_utd_times),
        "peak_memory_mb": peak_memory_mb,
    }


def _print_markdown_table(results: list[dict[str, Any]]) -> None:
    baseline = results[0]["outer_mean_s"] if results else 0.0
    print()
    print("| scenario | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | peak mem (MB) | speedup vs baseline |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        speedup = (
            (baseline - float(result["outer_mean_s"])) / baseline * 100.0
            if baseline > 0
            else 0.0
        )
        print(
            "| {name} | {outer:.4f} | {ups:.4f} | {sample:.4f} | "
            "{critic:.4f} | {high_utd:.4f} | {mem:.1f} | {speedup:.2f}% |".format(
                name=result["name"],
                outer=float(result["outer_mean_s"]),
                ups=float(result["updates_per_sec"]),
                sample=float(result["sample_total_mean_s"]),
                critic=float(result["critic_only_total_mean_s"]),
                high_utd=float(result["high_utd_mean_s"]),
                mem=float(result["peak_memory_mb"]),
                speedup=speedup,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "examples/agibot_real/configs/train_residual.yaml",
    )
    parser.add_argument("--fake-steps", type=int, default=700)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--payload-iterations", type=int, default=3)
    parser.add_argument("--crop-iterations", type=int, default=10)
    parser.add_argument("--sample-iterations", type=int, default=10)
    parser.add_argument("--prefetch-queue-size", type=int, default=2)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument(
        "--scenario-filter",
        default="",
        help="Only run scenarios whose name contains this substring.",
    )
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip pretrained ResNet weights. The architecture is unchanged.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA because bf16 is CUDA-only here.")

    base_cfg = _load_cfg(
        config_path=args.config,
        mixed_precision=False,
        random_init=bool(args.random_init),
        freeze_backbone=False,
    )
    sample_obs, residual_action_spec = _make_sample_obs_and_spec(base_cfg)
    replay_buffer = _build_fake_replay(
        cfg=base_cfg,
        sample_obs=sample_obs,
        num_steps=int(args.fake_steps),
        episode_length=int(args.episode_length),
        seed=int(args.seed),
    )
    _validate_fast_step_window_sample(
        replay_buffer,
        batch_size=int(base_cfg.replay.batch_size),
    )

    scenario_specs = [
        {
            "name": "baseline_fp32_legacy",
            "mixed_precision": False,
            "freeze_backbone": False,
            "legacy_actor_update": True,
        },
        {
            "name": "bf16_legacy",
            "mixed_precision": True,
            "freeze_backbone": False,
            "legacy_actor_update": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update",
            "mixed_precision": True,
            "freeze_backbone": False,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone",
            "mixed_precision": True,
            "freeze_backbone": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_fast_sample",
            "mixed_precision": True,
            "freeze_backbone": True,
            "fast_step_window_sample": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_big_batch",
            "mixed_precision": True,
            "freeze_backbone": True,
            "big_batch_sample": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample",
            "mixed_precision": True,
            "freeze_backbone": True,
            "big_batch_sample": True,
            "fast_step_window_sample": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample_trim_actor_batch",
            "mixed_precision": True,
            "freeze_backbone": True,
            "big_batch_sample": True,
            "fast_step_window_sample": True,
            "trim_actor_temperature_batch": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_prefetch",
            "mixed_precision": True,
            "freeze_backbone": True,
            "replay_prefetch": True,
        },
        {
            "name": "bf16_freeze_critic_actor_update_freeze_backbone_fast_sample_prefetch",
            "mixed_precision": True,
            "freeze_backbone": True,
            "fast_step_window_sample": True,
            "replay_prefetch": True,
        },
    ]
    if bool(args.include_compile):
        compile_scenarios = [
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_compile_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "torch_compile_target": "critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_compile_actor_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "torch_compile_target": "actor_critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_fast_sample_compile_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "fast_step_window_sample": True,
                "torch_compile_target": "critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_fast_sample_compile_actor_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "fast_step_window_sample": True,
                "torch_compile_target": "actor_critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample_compile_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "big_batch_sample": True,
                "fast_step_window_sample": True,
                "torch_compile_target": "critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample_compile_actor_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "big_batch_sample": True,
                "fast_step_window_sample": True,
                "torch_compile_target": "actor_critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_prefetch_compile_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "replay_prefetch": True,
                "torch_compile_target": "critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_prefetch_compile_actor_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "replay_prefetch": True,
                "torch_compile_target": "actor_critic",
            },
            {
                "name": "bf16_freeze_critic_actor_update_freeze_backbone_fast_sample_prefetch_compile_actor_critic",
                "mixed_precision": True,
                "freeze_backbone": True,
                "fast_step_window_sample": True,
                "replay_prefetch": True,
                "torch_compile_target": "actor_critic",
            },
        ]
        insert_at = next(
            (
                index
                for index, scenario in enumerate(scenario_specs)
                if scenario["name"]
                == "bf16_freeze_critic_actor_update_freeze_backbone_prefetch"
            ),
            len(scenario_specs),
        )
        scenario_specs[insert_at:insert_at] = compile_scenarios

    scenario_filter = str(args.scenario_filter)
    if scenario_filter:
        scenario_specs = [
            scenario
            for scenario in scenario_specs
            if scenario_filter in str(scenario["name"])
        ]
        if not scenario_specs:
            raise ValueError(f"No benchmark scenarios matched filter={scenario_filter!r}")

    results: list[dict[str, Any]] = []
    for scenario in scenario_specs:
        cfg = _load_cfg(
            config_path=args.config,
            mixed_precision=bool(scenario.get("mixed_precision", False)),
            random_init=bool(args.random_init),
            freeze_backbone=bool(scenario.get("freeze_backbone", False)),
        )
        result = _benchmark_scenario(
            name=str(scenario["name"]),
            cfg=cfg,
            sample_obs=sample_obs,
            residual_action_spec=residual_action_spec,
            replay_buffer=replay_buffer,
            use_legacy_actor_update=bool(scenario.get("legacy_actor_update", False)),
            use_replay_prefetch=bool(scenario.get("replay_prefetch", False)),
            use_big_batch_sample=bool(scenario.get("big_batch_sample", False)),
            use_fast_step_window_sample=bool(
                scenario.get("fast_step_window_sample", False)
            ),
            trim_actor_temperature_batch=bool(
                scenario.get("trim_actor_temperature_batch", False)
            ),
            torch_compile_target=str(scenario.get("torch_compile_target", "none")),
            torch_compile_mode=str(args.torch_compile_mode),
            torch_compile_backend=str(args.torch_compile_backend),
            torch_compile_fullgraph=bool(args.torch_compile_fullgraph),
            torch_compile_dynamic=bool(args.torch_compile_dynamic),
            prefetch_queue_size=int(args.prefetch_queue_size),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        results.append(result)

    crop_benchmark = _benchmark_random_crop(
        iterations=int(args.crop_iterations),
        warmup=int(args.warmup),
    )
    replay_sample_benchmark = _benchmark_replay_sampling(
        cfg=base_cfg,
        replay_buffer=replay_buffer,
        iterations=int(args.sample_iterations),
        warmup=int(args.warmup),
    )

    payload_cfg = _load_cfg(
        config_path=args.config,
        mixed_precision=True,
        random_init=bool(args.random_init),
        freeze_backbone=False,
    )
    payload_benchmark = _benchmark_payloads(
        cfg=payload_cfg,
        sample_obs=sample_obs,
        residual_action_spec=residual_action_spec,
        replay_buffer=replay_buffer,
        iterations=int(args.payload_iterations),
    )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "config": str(args.config),
        "fake_steps": int(args.fake_steps),
        "episode_length": int(args.episode_length),
        "batch_size": int(base_cfg.replay.batch_size),
        "critic_actor_ratio": int(base_cfg.training.critic_actor_ratio),
        "utd_ratio": int(base_cfg.sac.utd_ratio),
        "random_init": bool(args.random_init),
        "include_compile": bool(args.include_compile),
        "torch_compile_mode": str(args.torch_compile_mode),
        "torch_compile_backend": str(args.torch_compile_backend),
        "torch_compile_fullgraph": bool(args.torch_compile_fullgraph),
        "torch_compile_dynamic": bool(args.torch_compile_dynamic),
        "results": results,
        "crop_benchmark": crop_benchmark,
        "replay_sample_benchmark": replay_sample_benchmark,
        "payload_benchmark": payload_benchmark,
    }

    _print_markdown_table(results)
    if crop_benchmark:
        print()
        print("| random crop benchmark | legacy loop | vectorized | improvement |")
        print("|---|---:|---:|---:|")
        print(
            "| mean s | {legacy:.6f} | {vectorized:.6f} | {speedup:.2f}% faster |".format(
                legacy=crop_benchmark["legacy_loop_mean_s"],
                vectorized=crop_benchmark["vectorized_mean_s"],
                speedup=crop_benchmark["speedup_pct"],
            )
        )
    print()
    print("| replay sample benchmark | mean s | improvement |")
    print("|---|---:|---:|")
    print(
        "| legacy single batch | {value:.6f} | - |".format(
            value=replay_sample_benchmark["legacy_single_batch_mean_s"],
        )
    )
    print(
        "| fast single batch | {value:.6f} | {speedup:.2f}% faster |".format(
            value=replay_sample_benchmark["fast_single_batch_mean_s"],
            speedup=replay_sample_benchmark["fast_single_speedup_pct"],
        )
    )
    print(
        "| legacy {ratio}x separate samples | {value:.6f} | - |".format(
            ratio=replay_sample_benchmark["critic_actor_ratio"],
            value=replay_sample_benchmark["legacy_outer_separate_mean_s"],
        )
    )
    print(
        "| legacy big batch + split | {value:.6f} | {speedup:.2f}% faster |".format(
            value=replay_sample_benchmark["legacy_big_batch_mean_s"],
            speedup=replay_sample_benchmark["big_batch_speedup_pct"],
        )
    )
    print(
        "| fast big batch + split | {value:.6f} | {speedup:.2f}% faster |".format(
            value=replay_sample_benchmark["fast_big_batch_mean_s"],
            speedup=replay_sample_benchmark[
                "fast_big_vs_legacy_separate_speedup_pct"
            ],
        )
    )
    print()
    print("| payload benchmark | full | actor-only | improvement |")
    print("|---|---:|---:|---:|")
    print(
        "| tensor payload MB | {full:.1f} | {actor:.1f} | {improve:.2f}% smaller |".format(
            full=payload_benchmark["full_payload_tensor_mb"],
            actor=payload_benchmark["actor_only_payload_tensor_mb"],
            improve=payload_benchmark["payload_reduction_pct"],
        )
    )
    print(
        "| snapshot mean s | {full:.4f} | {actor:.4f} | {improve:.2f}% faster |".format(
            full=payload_benchmark["full_snapshot_mean_s"],
            actor=payload_benchmark["actor_only_snapshot_mean_s"],
            improve=payload_benchmark["snapshot_speedup_pct"],
        )
    )
    print(
        "| actor apply mean s | {full:.4f} | {actor:.4f} | {improve:.2f}% faster |".format(
            full=payload_benchmark["full_apply_mean_s"],
            actor=payload_benchmark["actor_only_apply_mean_s"],
            improve=payload_benchmark["apply_speedup_pct"],
        )
    )
    print()
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
