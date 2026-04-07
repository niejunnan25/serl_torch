#!/usr/bin/env python3
"""Compute state/action normalization stats from LIBERO HDF5 demos."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.hdf5_utils import LiberoTaskSpec, resolve_task_specs


def _sorted_demo_names(dataset_file: h5py.File) -> list[str]:
    names = list(dataset_file["data"].keys())
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def _stack_task_arrays(task_spec: LiberoTaskSpec) -> tuple[np.ndarray, np.ndarray, int]:
    all_states = []
    all_actions = []
    total_episodes = 0

    with h5py.File(task_spec.dataset_path, "r") as dataset_file:
        demo_names = _sorted_demo_names(dataset_file)
        demo_iter = tqdm(
            demo_names,
            desc=f"{task_spec.task_key} demos",
            unit="ep",
            dynamic_ncols=True,
            leave=False,
        )
        for demo_name in demo_iter:
            demo = dataset_file["data"][demo_name]
            obs = demo["obs"]
            states = np.concatenate(
                [
                    np.asarray(obs["ee_pos"], dtype=np.float64),
                    np.asarray(obs["ee_ori"], dtype=np.float64),
                    np.asarray(obs["gripper_states"], dtype=np.float64),
                ],
                axis=-1,
            )
            actions = np.asarray(demo["actions"], dtype=np.float64)
            all_states.append(states)
            all_actions.append(actions)
            total_episodes += 1

    if not all_states:
        raise ValueError(f"No demos found in {task_spec.dataset_path}")

    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_actions, axis=0),
        total_episodes,
    )


def _compute_stats(states: np.ndarray, actions: np.ndarray, task_spec: LiberoTaskSpec, total_episodes: int) -> dict:
    eps = 1e-6
    state_std = np.maximum(states.std(axis=0), eps)
    action_std = np.maximum(actions.std(axis=0), eps)
    return {
        "task_key": task_spec.task_key,
        "suite_name": task_spec.suite_name,
        "task_id": int(task_spec.task_id),
        "task_name": task_spec.task_name,
        "task_description": task_spec.task_description,
        "dataset_path": str(task_spec.dataset_path),
        "total_episodes": int(total_episodes),
        "total_frames": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "state_mean": states.mean(axis=0).astype(np.float32).tolist(),
        "state_std": state_std.astype(np.float32).tolist(),
        "state_min": states.min(axis=0).astype(np.float32).tolist(),
        "state_max": states.max(axis=0).astype(np.float32).tolist(),
        "action_mean": actions.mean(axis=0).astype(np.float32).tolist(),
        "action_std": action_std.astype(np.float32).tolist(),
        "action_min": actions.min(axis=0).astype(np.float32).tolist(),
        "action_max": actions.max(axis=0).astype(np.float32).tolist(),
    }


def _task_id_list(args: argparse.Namespace) -> Iterable[int] | None:
    if args.all or args.task_id is None:
        return None
    return [int(args.task_id)]


def _log_line(message: str) -> None:
    tqdm.write(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute normalization stats from LIBERO HDF5 demos")
    parser.add_argument("--suite_name", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=None, help="Single task id to process")
    parser.add_argument("--all", action="store_true", help="Process all tasks in the suite")
    parser.add_argument("--libero_root", type=str, default=None)
    parser.add_argument("--libero_config_dir", type=str, default=None)
    parser.add_argument("--libero_datasets_root", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "stats"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = resolve_task_specs(
        suite_name=args.suite_name,
        task_ids=_task_id_list(args),
        libero_root=args.libero_root,
        libero_config_dir=args.libero_config_dir,
        libero_datasets_root=args.libero_datasets_root,
    )

    task_iter = specs
    if len(specs) > 1:
        task_iter = tqdm(specs, desc="Tasks", unit="task", dynamic_ncols=True)

    for task_spec in task_iter:
        _log_line(f"Computing stats for {task_spec.task_key}: {task_spec.dataset_path}")
        states, actions, total_episodes = _stack_task_arrays(task_spec)
        stats = _compute_stats(states, actions, task_spec, total_episodes)
        output_path = output_dir / f"{task_spec.task_key}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        _log_line(
            f"  saved={output_path} episodes={stats['total_episodes']} "
            f"frames={stats['total_frames']} state_dim={stats['state_dim']} action_dim={stats['action_dim']}"
        )


if __name__ == "__main__":
    main()
