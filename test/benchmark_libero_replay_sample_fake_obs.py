#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark LIBERO chunk replay sampling with fake observations only.

This isolates replay sampling from env rollout, policy inference, and learner
backprop. It uses the production StepWindowReplayBuffer implementation and the
same mixed online/offline sampling helper used by residual training.
"""

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for path in (REPO_ROOT, SERL_LAUNCHER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import sample_mixed_training_batch


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _make_observation_space(
    *,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
) -> gym.spaces.Dict:
    spaces: Dict[str, gym.Space] = {}
    for key in image_keys:
        spaces[key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(int(image_size), int(image_size), 3),
            dtype=np.uint8,
        )
    spaces["robot_proprio"] = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(proprio_dim),),
        dtype=np.float32,
    )
    spaces["base_action_chunk"] = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(chunk_horizon), int(action_dim)),
        dtype=np.float32,
    )
    spaces["alpha"] = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(1,),
        dtype=np.float32,
    )
    return gym.spaces.Dict(spaces)


def _fake_obs_batch(
    *,
    count: int,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    fill: int,
) -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {}
    for key in image_keys:
        obs[key] = np.full(
            (int(count), int(image_size), int(image_size), 3),
            int(fill) % 256,
            dtype=np.uint8,
        )
    obs["robot_proprio"] = np.zeros((int(count), int(proprio_dim)), dtype=np.float32)
    obs["base_action_chunk"] = np.zeros(
        (int(count), int(chunk_horizon), int(action_dim)),
        dtype=np.float32,
    )
    obs["alpha"] = np.zeros((int(count), 1), dtype=np.float32)
    return obs


def _fake_transition_batch(
    *,
    start_step: int,
    count: int,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    episode_length: int,
) -> Dict[str, Any]:
    step_ids = np.arange(int(start_step), int(start_step) + int(count), dtype=np.int64)
    episode_length = max(1, int(episode_length))
    episode_step = (step_ids % episode_length).astype(np.int32)
    dones = episode_step == (episode_length - 1)
    return {
        "episode_id": (step_ids // episode_length).astype(np.int64),
        "episode_step": episode_step,
        "observations": _fake_obs_batch(
            count=int(count),
            image_size=int(image_size),
            image_keys=image_keys,
            proprio_dim=int(proprio_dim),
            chunk_horizon=int(chunk_horizon),
            action_dim=int(action_dim),
            fill=int(start_step),
        ),
        "actions": np.zeros((int(count), int(action_dim)), dtype=np.float32),
        "next_observations": _fake_obs_batch(
            count=int(count),
            image_size=int(image_size),
            image_keys=image_keys,
            proprio_dim=int(proprio_dim),
            chunk_horizon=int(chunk_horizon),
            action_dim=int(action_dim),
            fill=int(start_step) + 1,
        ),
        "rewards": np.zeros((int(count),), dtype=np.float32),
        "masks": (~dones).astype(np.float32),
        "dones": dones.astype(bool),
    }


def _fill_replay(
    *,
    name: str,
    steps: int,
    capacity: int,
    chunk_size: int,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    episode_length: int,
) -> Any:
    observation_space = _make_observation_space(
        image_size=int(image_size),
        image_keys=image_keys,
        proprio_dim=int(proprio_dim),
        chunk_horizon=int(chunk_horizon),
        action_dim=int(action_dim),
    )
    replay = create_chunk_replay_buffer(
        observation_space=observation_space,
        action_dim=int(action_dim),
        chunk_horizon=int(chunk_horizon),
        discount=0.99,
        image_keys=tuple(image_keys),
        capacity=int(capacity),
    )
    start_time = time.perf_counter()
    inserted = 0
    while inserted < int(steps):
        count = min(int(chunk_size), int(steps) - inserted)
        replay.batch_insert(
            _fake_transition_batch(
                start_step=inserted,
                count=count,
                image_size=int(image_size),
                image_keys=image_keys,
                proprio_dim=int(proprio_dim),
                chunk_horizon=int(chunk_horizon),
                action_dim=int(action_dim),
                episode_length=int(episode_length),
            )
        )
        inserted += count
    print(
        json.dumps(
            {
                "event": "filled_replay",
                "name": str(name),
                "steps": int(steps),
                "capacity": int(capacity),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "num_windows": int(replay.num_windows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return replay


def _prefill_replay_metadata(
    *,
    name: str,
    steps: int,
    capacity: int,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    episode_length: int,
) -> Any:
    """Create a replay with valid metadata while leaving large image arrays untouched."""

    observation_space = _make_observation_space(
        image_size=int(image_size),
        image_keys=image_keys,
        proprio_dim=int(proprio_dim),
        chunk_horizon=int(chunk_horizon),
        action_dim=int(action_dim),
    )
    replay = create_chunk_replay_buffer(
        observation_space=observation_space,
        action_dim=int(action_dim),
        chunk_horizon=int(chunk_horizon),
        discount=0.99,
        image_keys=tuple(image_keys),
        capacity=int(capacity),
    )
    start_time = time.perf_counter()
    steps = min(int(steps), int(capacity))
    step_ids = np.arange(steps, dtype=np.int64)
    replay._episode_ids[:steps] = (step_ids // max(1, int(episode_length))).astype(
        np.int64
    )
    replay._episode_steps[:steps] = (step_ids % max(1, int(episode_length))).astype(
        np.int32
    )
    replay._step_ids[:steps] = step_ids
    replay.dataset_dict["actions"][:steps] = 0.0
    replay.dataset_dict["rewards"][:steps] = 0.0
    replay.dataset_dict["masks"][:steps] = 1.0
    replay.dataset_dict["dones"][:steps] = False
    replay._has_explicit_next_pixels[:steps] = False
    replay._size = int(steps)
    replay._insert_count = int(steps)
    replay._insert_index = int(steps % int(capacity))

    candidates: List[int] = []
    max_start = int(steps) - int(chunk_horizon) - 1
    if max_start >= 0:
        for step_id in range(0, max_start + 1):
            start_episode_step = int(replay._episode_steps[step_id])
            if start_episode_step + int(chunk_horizon) < int(episode_length):
                candidates.append(int(step_id))
    replay._candidate_start_step_ids.clear()
    replay._candidate_start_step_ids.extend(candidates)
    replay._candidate_start_step_set = set(candidates)
    print(
        json.dumps(
            {
                "event": "prefilled_replay_metadata",
                "name": str(name),
                "steps": int(steps),
                "capacity": int(capacity),
                "elapsed_sec": float(time.perf_counter() - start_time),
                "num_windows": int(replay.num_windows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return replay


def _create_replay(
    *,
    mode: str,
    name: str,
    steps: int,
    capacity: int,
    chunk_size: int,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    episode_length: int,
) -> Any:
    if str(mode) == "metadata":
        return _prefill_replay_metadata(
            name=name,
            steps=int(steps),
            capacity=int(capacity),
            image_size=int(image_size),
            image_keys=image_keys,
            proprio_dim=int(proprio_dim),
            chunk_horizon=int(chunk_horizon),
            action_dim=int(action_dim),
            episode_length=int(episode_length),
        )
    if str(mode) != "batch":
        raise ValueError(f"Unsupported prefill mode: {mode}")
    return _fill_replay(
        name=name,
        steps=int(steps),
        capacity=int(capacity),
        chunk_size=int(chunk_size),
        image_size=int(image_size),
        image_keys=image_keys,
        proprio_dim=int(proprio_dim),
        chunk_horizon=int(chunk_horizon),
        action_dim=int(action_dim),
        episode_length=int(episode_length),
    )


def _profile_once(
    *,
    online_replay: Any,
    offline_replay: Optional[Any],
    batch_size: int,
    offline_ratio: float,
    pack_obs_and_next_obs: bool,
    prefer_device_concat: bool,
    device: Any,
) -> Dict[str, Any]:
    profile: Dict[str, float] = {}
    start = time.perf_counter()
    _batch, batch_mix = sample_mixed_training_batch(
        online_replay_buffer=online_replay,
        offline_replay_buffer=offline_replay,
        batch_size=int(batch_size),
        offline_ratio=float(offline_ratio),
        profile=profile,
        pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
        prefer_device_concat=bool(prefer_device_concat),
        device=device,
    )
    return {
        "elapsed_sec": float(time.perf_counter() - start),
        "batch_mix": batch_mix,
        "profile": dict(profile),
    }


def _run_samples(
    *,
    online_replay: Any,
    offline_replay: Optional[Any],
    batch_size: int,
    offline_ratio: float,
    iterations: int,
    warmup: int,
    samples_per_iteration: int,
    pack_obs_and_next_obs: bool,
    prefer_device_concat: bool,
    device: Any,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    total_iterations = int(warmup) + int(iterations)
    for index in range(total_iterations):
        iter_start = time.perf_counter()
        merged_profile: Dict[str, float] = {}
        for _ in range(int(samples_per_iteration)):
            record = _profile_once(
                online_replay=online_replay,
                offline_replay=offline_replay,
                batch_size=int(batch_size),
                offline_ratio=float(offline_ratio),
                pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
                prefer_device_concat=bool(prefer_device_concat),
                device=device,
            )
            for key, value in record["profile"].items():
                if isinstance(value, (int, float)):
                    merged_profile[key] = float(merged_profile.get(key, 0.0)) + float(value)
        elapsed = float(time.perf_counter() - iter_start)
        if index >= int(warmup):
            records.append({"elapsed_sec": elapsed, "profile": merged_profile})

    profile_summary: Dict[str, Dict[str, float]] = {}
    all_keys = sorted({key for record in records for key in record["profile"]})
    for key in all_keys:
        profile_summary[key] = _summarize(
            record["profile"].get(key, 0.0) for record in records
        )
    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "samples_per_iteration": int(samples_per_iteration),
        "elapsed_sec": _summarize(record["elapsed_sec"] for record in records),
        "profile": profile_summary,
    }


def _start_writer(
    *,
    replay: Any,
    start_step: int,
    stop_event: threading.Event,
    chunk_size: int,
    sleep_sec: float,
    image_size: int,
    image_keys: List[str],
    proprio_dim: int,
    chunk_horizon: int,
    action_dim: int,
    episode_length: int,
) -> threading.Thread:
    def _worker() -> None:
        step = int(start_step)
        while not stop_event.is_set():
            replay.batch_insert(
                _fake_transition_batch(
                    start_step=step,
                    count=int(chunk_size),
                    image_size=int(image_size),
                    image_keys=image_keys,
                    proprio_dim=int(proprio_dim),
                    chunk_horizon=int(chunk_horizon),
                    action_dim=int(action_dim),
                    episode_length=int(episode_length),
                )
            )
            step += int(chunk_size)
            if sleep_sec > 0.0:
                time.sleep(float(sleep_sec))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-steps", type=int, default=20000)
    parser.add_argument("--offline-steps", type=int, default=7479)
    parser.add_argument("--capacity", type=int, default=250000)
    parser.add_argument("--offline-capacity", type=int, default=50000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-keys", default="image,wrist_image")
    parser.add_argument("--proprio-dim", type=int, default=9)
    parser.add_argument("--chunk-horizon", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--insert-chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offline-ratio", type=float, default=0.5)
    parser.add_argument("--critic-actor-ratio", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--prefill-mode",
        choices=("batch", "metadata"),
        default="batch",
        help=(
            "batch uses production batch_insert; metadata skips large fake image writes "
            "and constructs a valid replay state for sampling-only profiling"
        ),
    )
    parser.add_argument("--concurrent-writer", action="store_true")
    parser.add_argument("--writer-chunk-size", type=int, default=30)
    parser.add_argument("--writer-sleep-ms", type=float, default=20.0)
    parser.add_argument("--pack-obs-and-next-obs", action="store_true")
    parser.add_argument("--prefer-device-concat", action="store_true")
    parser.add_argument(
        "--device",
        default="none",
        help="Device passed to torch conversion when --prefer-device-concat is set.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    image_keys = [key for key in str(args.image_keys).split(",") if key]
    device = None if str(args.device).lower() in {"", "none", "null"} else str(args.device)
    online_replay = _create_replay(
        mode=str(args.prefill_mode),
        name="online",
        steps=int(args.online_steps),
        capacity=max(int(args.capacity), int(args.online_steps) + int(args.writer_chunk_size) + 32),
        chunk_size=int(args.insert_chunk_size),
        image_size=int(args.image_size),
        image_keys=image_keys,
        proprio_dim=int(args.proprio_dim),
        chunk_horizon=int(args.chunk_horizon),
        action_dim=int(args.action_dim),
        episode_length=int(args.episode_length),
    )
    offline_replay = None
    if int(args.offline_steps) > 0 and float(args.offline_ratio) > 0.0:
        offline_replay = _create_replay(
            mode=str(args.prefill_mode),
            name="offline",
            steps=int(args.offline_steps),
            capacity=max(int(args.offline_capacity), int(args.offline_steps) + 32),
            chunk_size=int(args.insert_chunk_size),
            image_size=int(args.image_size),
            image_keys=image_keys,
            proprio_dim=int(args.proprio_dim),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(args.action_dim),
            episode_length=int(args.episode_length),
        )

    stop_event = threading.Event()
    writer_thread = None
    if bool(args.concurrent_writer):
        writer_thread = _start_writer(
            replay=online_replay,
            start_step=int(args.online_steps),
            stop_event=stop_event,
            chunk_size=int(args.writer_chunk_size),
            sleep_sec=float(args.writer_sleep_ms) / 1000.0,
            image_size=int(args.image_size),
            image_keys=image_keys,
            proprio_dim=int(args.proprio_dim),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(args.action_dim),
            episode_length=int(args.episode_length),
        )

    try:
        result = {
            "config": {
                "online_steps": int(args.online_steps),
                "offline_steps": int(args.offline_steps),
                "capacity": int(args.capacity),
                "image_size": int(args.image_size),
                "image_keys": image_keys,
                "batch_size": int(args.batch_size),
                "offline_ratio": float(args.offline_ratio),
                "critic_actor_ratio": int(args.critic_actor_ratio),
                "prefill_mode": str(args.prefill_mode),
                "concurrent_writer": bool(args.concurrent_writer),
                "writer_chunk_size": int(args.writer_chunk_size),
                "writer_sleep_ms": float(args.writer_sleep_ms),
                "pack_obs_and_next_obs": bool(args.pack_obs_and_next_obs),
                "prefer_device_concat": bool(args.prefer_device_concat),
                "device": str(args.device),
            },
            "single_mixed_sample": _run_samples(
                online_replay=online_replay,
                offline_replay=offline_replay,
                batch_size=int(args.batch_size),
                offline_ratio=float(args.offline_ratio),
                iterations=int(args.iterations),
                warmup=int(args.warmup),
                samples_per_iteration=1,
                pack_obs_and_next_obs=bool(args.pack_obs_and_next_obs),
                prefer_device_concat=bool(args.prefer_device_concat),
                device=device,
            ),
            "learner_update_sample_pattern": _run_samples(
                online_replay=online_replay,
                offline_replay=offline_replay,
                batch_size=int(args.batch_size),
                offline_ratio=float(args.offline_ratio),
                iterations=int(args.iterations),
                warmup=int(args.warmup),
                samples_per_iteration=max(1, int(args.critic_actor_ratio)),
                pack_obs_and_next_obs=bool(args.pack_obs_and_next_obs),
                prefer_device_concat=bool(args.prefer_device_concat),
                device=device,
            ),
            "final_online_size": int(len(online_replay)),
            "final_online_windows": int(online_replay.num_windows),
        }
    finally:
        stop_event.set()
        if writer_thread is not None:
            writer_thread.join(timeout=5.0)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
