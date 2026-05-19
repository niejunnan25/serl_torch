#!/usr/bin/env python3
from __future__ import annotations

"""Analyze gripper-change timing in LIBERO HDF5 demonstration files."""

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


QUANTILES = (10, 25, 50, 75, 90)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary(values: list[int | None]) -> dict[str, Any]:
    present = np.asarray([value for value in values if value is not None], dtype=float)
    missing = int(len(values) - int(present.size))
    result: dict[str, Any] = {
        "count": int(present.size),
        "missing": int(missing),
    }
    if present.size <= 0:
        result.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                **{f"q{q}": None for q in QUANTILES},
            }
        )
        return result
    result.update(
        {
            "mean": float(np.mean(present)),
            "std": float(np.std(present)),
            "min": int(np.min(present)),
            "max": int(np.max(present)),
        }
    )
    for quantile in QUANTILES:
        result[f"q{quantile}"] = float(np.percentile(present, quantile))
    return result


def _align_floor(value: float, *, align_to: int) -> int:
    clipped = max(0.0, float(value))
    if int(align_to) <= 1:
        return int(np.floor(clipped))
    return int(np.floor(clipped / float(align_to)) * int(align_to))


def _discover_action_datasets(h5_file: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if Path(str(name)).name != "actions":
            return
        if len(obj.shape) != 2:
            return
        if int(obj.shape[0]) <= 0 or int(obj.shape[1]) <= 0:
            return
        paths.append(str(name))

    h5_file.visititems(visitor)
    return sorted(paths, key=_dataset_sort_key)


def _dataset_sort_key(path: str) -> tuple[str, int, str]:
    parts = str(path).split("/")
    parent = parts[-2] if len(parts) >= 2 else ""
    if parent.startswith("demo_"):
        try:
            return ("/".join(parts[:-2]), int(parent.split("_")[-1]), path)
        except ValueError:
            pass
    return ("/".join(parts[:-1]), 10**9, path)


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size <= 0:
        return None
    return int(indices[0] + 1)


def _analyze_actions(
    actions: np.ndarray,
    *,
    gripper_dim: int,
    change_threshold: float,
) -> dict[str, Any]:
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape={actions.shape}")
    resolved_dim = int(gripper_dim)
    if resolved_dim < 0:
        resolved_dim = int(actions.shape[1]) + resolved_dim
    if resolved_dim < 0 or resolved_dim >= int(actions.shape[1]):
        raise ValueError(
            f"gripper_dim={gripper_dim} is out of range for action shape={actions.shape}"
        )
    gripper = np.asarray(actions[:, resolved_dim], dtype=float)
    delta = np.diff(gripper)
    return {
        "episode_length": int(actions.shape[0]),
        "action_shape": [int(value) for value in actions.shape],
        "gripper_initial": float(gripper[0]),
        "gripper_final": float(gripper[-1]),
        "gripper_min": float(np.min(gripper)),
        "gripper_max": float(np.max(gripper)),
        "first_gripper_change_step": _first_index(
            np.abs(delta) > float(change_threshold)
        ),
        "first_close_like_change_step": _first_index(
            delta > float(change_threshold)
        ),
        "first_open_like_change_step": _first_index(
            delta < -float(change_threshold)
        ),
    }


def analyze_file(
    path: Path,
    *,
    gripper_dim: int,
    change_threshold: float,
    pre_window: int,
    align_to: int,
) -> dict[str, Any]:
    demos: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5_file:
        action_paths = _discover_action_datasets(h5_file)
        for action_path in action_paths:
            actions = np.asarray(h5_file[action_path])
            demo_stats = _analyze_actions(
                actions,
                gripper_dim=gripper_dim,
                change_threshold=change_threshold,
            )
            demo_stats["dataset_path"] = str(action_path)
            demos.append(demo_stats)

    episode_lengths = [int(item["episode_length"]) for item in demos]
    first_changes = [item["first_gripper_change_step"] for item in demos]
    first_close = [item["first_close_like_change_step"] for item in demos]
    first_open = [item["first_open_like_change_step"] for item in demos]
    change_summary = _summary(first_changes)
    q10 = change_summary.get("q10", None)
    suggested_start_step = None
    if q10 is not None:
        suggested_start_step = _align_floor(
            float(q10) - float(pre_window),
            align_to=int(align_to),
        )

    return {
        "file": str(path),
        "file_name": path.name,
        "num_demos": int(len(demos)),
        "action_datasets": [
            {
                "path": str(item["dataset_path"]),
                "shape": list(item["action_shape"]),
            }
            for item in demos
        ],
        "episode_length": _summary(episode_lengths),
        "first_gripper_change_step": change_summary,
        "first_close_like_change_step": _summary(first_close),
        "first_open_like_change_step": _summary(first_open),
        "suggested_start_step": suggested_start_step,
        "demos": demos,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    keys = ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")
    values = []
    for key in keys:
        value = summary.get(key, None)
        if value is None:
            values.append("None")
        elif isinstance(value, float):
            values.append(f"{value:.2f}")
        else:
            values.append(str(value))
    return " ".join(f"{key}={value}" for key, value in zip(keys, values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LIBERO gripper action change steps from HDF5 demos.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing .hdf5 files.")
    parser.add_argument("--gripper-dim", type=int, default=-1)
    parser.add_argument("--change-threshold", type=float, default=0.05)
    parser.add_argument("--pre-window", type=int, default=10)
    parser.add_argument("--align-to", type=int, default=5)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(data_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {data_dir}")

    result = {
        "data_dir": str(data_dir),
        "gripper_dim": int(args.gripper_dim),
        "change_threshold": float(args.change_threshold),
        "pre_window": int(args.pre_window),
        "align_to": int(args.align_to),
        "files": [],
        "file_sorted_index": [],
    }

    for index, file_path in enumerate(files):
        file_stats = analyze_file(
            file_path,
            gripper_dim=int(args.gripper_dim),
            change_threshold=float(args.change_threshold),
            pre_window=int(args.pre_window),
            align_to=int(args.align_to),
        )
        result["files"].append(file_stats)
        result["file_sorted_index"].append(
            {
                "index_0_based": int(index),
                "index_1_based": int(index + 1),
                "file_name": file_path.name,
                "suggested_start_step": file_stats["suggested_start_step"],
            }
        )

        print(f"\n[{index}] {file_path.name}")
        print(f"  num_demos={file_stats['num_demos']}")
        print(f"  first action dataset paths:")
        for dataset in file_stats["action_datasets"][:3]:
            print(f"    {dataset['path']} shape={dataset['shape']}")
        if len(file_stats["action_datasets"]) > 3:
            print(f"    ... {len(file_stats['action_datasets']) - 3} more")
        print(f"  episode_length: {_format_summary(file_stats['episode_length'])}")
        print(
            "  first_gripper_change_step: "
            f"{_format_summary(file_stats['first_gripper_change_step'])}"
        )
        print(
            "  first_close_like_change_step: "
            f"{_format_summary(file_stats['first_close_like_change_step'])}"
        )
        print(
            "  first_open_like_change_step: "
            f"{_format_summary(file_stats['first_open_like_change_step'])}"
        )
        print(f"  suggested_start_step={file_stats['suggested_start_step']}")

    print("\nFile sorted index mapping:")
    for entry in result["file_sorted_index"]:
        print(
            f"  0-based={entry['index_0_based']} "
            f"1-based={entry['index_1_based']} "
            f"suggested={entry['suggested_start_step']} "
            f"{entry['file_name']}"
        )

    if args.out_json is not None:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(_jsonable(result), fp, indent=2)
        print(f"\nWrote JSON stats to {out_path}")


if __name__ == "__main__":
    main()
