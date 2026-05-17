from __future__ import annotations

"""Measure pure LIBERO RPC throughput with fake actions and no policy inference.

Usage:

python test/benchmark_libero_rpc_fake_action.py --host 127.0.0.1 --port 30100
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.env.remote_task_env import RemoteLiberoTaskEnv


def _build_env(args: argparse.Namespace) -> RemoteLiberoTaskEnv:
    return RemoteLiberoTaskEnv(
        host=str(args.host),
        port=int(args.port),
        suite_name=str(args.suite_name),
        task_id=int(args.task_id),
        action_dim=int(args.action_dim),
        resolution=int(args.resolution),
        num_steps_wait=int(args.num_steps_wait),
        max_episode_steps=(
            None if args.max_episode_steps is None else int(args.max_episode_steps)
        ),
        libero_root=None if args.libero_root is None else str(args.libero_root),
        libero_config_dir=(
            None if args.libero_config_dir is None else str(args.libero_config_dir)
        ),
        libero_datasets_root=(
            None if args.libero_datasets_root is None else str(args.libero_datasets_root)
        ),
        env_seed=int(args.env_seed),
        timeout_sec=float(args.timeout_sec),
    )


def _reset_env(env: RemoteLiberoTaskEnv, *, episode_idx: int, seed: int) -> dict[str, Any]:
    return env.reset(seed=int(seed), init_episode_idx=int(episode_idx))


def _benchmark_step(
    env: RemoteLiberoTaskEnv,
    *,
    total_steps: int,
    action_dim: int,
    seed: int,
) -> dict[str, float]:
    action = np.zeros((int(action_dim),), dtype=np.float32)
    steps = 0
    episode_idx = 0
    _reset_env(env, episode_idx=episode_idx, seed=seed)
    start = time.perf_counter()
    while steps < int(total_steps):
        _obs, _reward, done, truncated, _info = env.step(action)
        steps += 1
        if bool(done or truncated):
            episode_idx += 1
            _reset_env(env, episode_idx=episode_idx, seed=seed)
    wall_time_sec = time.perf_counter() - start
    return {
        "steps": int(steps),
        "wall_time_sec": float(wall_time_sec),
        "steps_per_sec": float(steps) / max(1e-9, wall_time_sec),
        "wall_ms_per_step": 1000.0 * float(wall_time_sec) / max(1, int(steps)),
    }


def _benchmark_step_chunk(
    env: RemoteLiberoTaskEnv,
    *,
    total_steps: int,
    action_dim: int,
    chunk_horizon: int,
    seed: int,
) -> dict[str, float]:
    action_chunk = np.zeros((int(chunk_horizon), int(action_dim)), dtype=np.float32)
    steps = 0
    chunks = 0
    episode_idx = 0
    _reset_env(env, episode_idx=episode_idx, seed=seed)
    start = time.perf_counter()
    while steps < int(total_steps):
        result = env.step_chunk(action_chunk)
        executed_steps = int(result.get("num_steps", len(result.get("rewards", ()))))
        if executed_steps <= 0:
            raise RuntimeError(
                f"step_chunk returned no executed steps for horizon={chunk_horizon}"
            )
        steps += int(executed_steps)
        chunks += 1
        if bool(result.get("done", False) or result.get("truncated", False)):
            episode_idx += 1
            _reset_env(env, episode_idx=episode_idx, seed=seed)
    wall_time_sec = time.perf_counter() - start
    return {
        "steps": int(steps),
        "chunks": int(chunks),
        "wall_time_sec": float(wall_time_sec),
        "steps_per_sec": float(steps) / max(1e-9, wall_time_sec),
        "chunks_per_sec": float(chunks) / max(1e-9, wall_time_sec),
        "wall_ms_per_step": 1000.0 * float(wall_time_sec) / max(1, int(steps)),
        "wall_ms_per_chunk": 1000.0 * float(wall_time_sec) / max(1, int(chunks)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LIBERO remote env RPC throughput with fake actions",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--suite-name", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--libero-root", type=str, default=None)
    parser.add_argument("--libero-config-dir", type=str, default=None)
    parser.add_argument("--libero-datasets-root", type=str, default=None)
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--step-total", type=int, default=200)
    parser.add_argument("--chunk-total", type=int, default=300)
    parser.add_argument(
        "--chunk-horizons",
        type=int,
        nargs="+",
        default=(1, 5, 10, 20, 30),
    )
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = _build_env(args)
    try:
        step_result = _benchmark_step(
            env,
            total_steps=int(args.step_total),
            action_dim=int(args.action_dim),
            seed=int(args.env_seed),
        )
        chunk_results: dict[str, dict[str, float]] = {}
        for chunk_horizon in tuple(int(value) for value in args.chunk_horizons):
            chunk_results[str(chunk_horizon)] = _benchmark_step_chunk(
                env,
                total_steps=int(args.chunk_total),
                action_dim=int(args.action_dim),
                chunk_horizon=int(chunk_horizon),
                seed=int(args.env_seed),
            )
    finally:
        env.close(clear_cache=False)

    default_chunk_key = "5"
    best_chunk_key = max(
        chunk_results.keys(),
        key=lambda key: float(chunk_results[key]["steps_per_sec"]),
    )
    summary = {
        "step": step_result,
        "step_chunk": chunk_results,
        "default_chunk_horizon": int(default_chunk_key),
        "default_chunk": chunk_results.get(default_chunk_key),
        "best_chunk_horizon": int(best_chunk_key),
        "best_chunk": chunk_results[best_chunk_key],
    }

    print()
    print("=== LIBERO RPC Fake-Action Benchmark ===")
    print(
        "step(): "
        f"{step_result['steps_per_sec']:.2f} steps/s "
        f"({step_result['wall_ms_per_step']:.2f} ms/step)"
    )
    print("step_chunk():")
    for key in sorted(chunk_results.keys(), key=lambda value: int(value)):
        item = chunk_results[key]
        print(
            f"  H={int(key):<2} "
            f"{item['steps_per_sec']:.2f} steps/s "
            f"{item['chunks_per_sec']:.2f} chunks/s "
            f"({item['wall_ms_per_step']:.2f} ms/step, "
            f"{item['wall_ms_per_chunk']:.2f} ms/chunk)"
        )
    print(
        "best step_chunk: "
        f"H={int(best_chunk_key)} "
        f"{chunk_results[best_chunk_key]['steps_per_sec']:.2f} steps/s"
    )

    if args.json_out is not None:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"json written to {json_out}")


if __name__ == "__main__":
    main()
