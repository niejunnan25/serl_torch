#!/usr/bin/env python3
"""Convert LIBERO HDF5 demos into compact offline episode PKLs."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "h5py is required for convert_hdf5_to_offline.py. "
        "Please run this script in an environment with h5py installed."
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.data.hdf5_utils import resolve_task_specs
from serl_torch.examples.libero.env_wrappers import resolve_openpi_root, setup_openpi_client_pythonpath
from serl_torch.examples.libero.policy import OpenPIChunkClient, select_action_chunk_window

_T = TypeVar("_T")


def _progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Iterable[_T]:
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )


def _sorted_demo_names(dataset_file: h5py.File) -> list[str]:
    names = list(dataset_file["data"].keys())
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def _build_frame_obs(payload: dict, frame_idx: int) -> dict:
    return {
        "agentview_rgb": payload["agentview_rgb"][frame_idx],
        "eye_in_hand_rgb": payload["eye_in_hand_rgb"][frame_idx],
        "ee_pos": payload["ee_pos"][frame_idx],
        "ee_ori": payload["ee_ori"][frame_idx],
        "gripper_states": payload["gripper_states"][frame_idx],
    }


def _precompute_base_chunks(
    payload: dict,
    *,
    openpi_client: OpenPIChunkClient,
    chunk_horizon: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    num_frames = int(payload["actions"].shape[0])
    chunks = []
    chunk_starts: Iterable[int] = range(0, num_frames, chunk_horizon)
    total_chunks = (num_frames + chunk_horizon - 1) // chunk_horizon
    if progress_desc is not None:
        chunk_starts = _progress(
            chunk_starts,
            total=total_chunks,
            desc=progress_desc,
            unit="chunk",
            leave=False,
        )
    for chunk_start in chunk_starts:
        obs_raw = _build_frame_obs(payload, chunk_start)
        action_chunk, _ = openpi_client.infer_chunk(obs_raw, str(payload["task_description"]))
        chunks.append(select_action_chunk_window(action_chunk, horizon=chunk_horizon))
    return np.asarray(chunks, dtype=np.float32)


def _convert_demo(
    demo,
    *,
    suite_name: str,
    task_id: int,
    task_name: str,
    task_description: str,
    dataset_path: Path,
    episode_index: int,
    chunk_horizon: int,
    openpi_client: OpenPIChunkClient | None,
    demo_name: str,
) -> dict:
    obs = demo["obs"]
    payload = {
        "format": "libero_offline_episode_v1",
        "suite_name": suite_name,
        "task_id": int(task_id),
        "task_name": task_name,
        "task_description": task_description,
        "dataset_path": str(dataset_path),
        "episode_index": int(episode_index),
        "demo_name": str(demo_name),
        "chunk_horizon": int(chunk_horizon),
        "agentview_rgb": np.asarray(obs["agentview_rgb"], dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8),
        "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float32),
        "ee_ori": np.asarray(obs["ee_ori"], dtype=np.float32),
        "gripper_states": np.asarray(obs["gripper_states"], dtype=np.float32),
        "actions": np.asarray(demo["actions"], dtype=np.float32),
        "rewards": np.asarray(demo.get("rewards", np.zeros((len(demo["actions"]),))), dtype=np.float32),
        "dones": np.asarray(demo.get("dones", np.zeros((len(demo["actions"]),))), dtype=bool),
    }
    if payload["dones"].shape[0] > 0:
        payload["dones"][-1] = True
    if openpi_client is not None:
        payload["base_chunks"] = _precompute_base_chunks(
            payload,
            openpi_client=openpi_client,
            chunk_horizon=chunk_horizon,
            progress_desc=f"{task_name} ep={episode_index:03d}",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 demos into offline episode PKLs")
    parser.add_argument("--suite_name", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=None, help="Single task id to convert")
    parser.add_argument("--all", action="store_true", help="Convert all tasks in the suite")
    parser.add_argument("--libero_root", type=str, default=None)
    parser.add_argument("--openpi_root", type=str, default=None)
    parser.add_argument("--libero_config_dir", type=str, default=None)
    parser.add_argument("--libero_datasets_root", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "offline"),
    )
    parser.add_argument("--chunk_horizon", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--openpi_host", type=str, default=None)
    parser.add_argument("--openpi_port", type=int, default=30001)
    args = parser.parse_args()

    if args.openpi_host is not None:
        openpi_root = resolve_openpi_root(args.openpi_root)
        setup_openpi_client_pythonpath(openpi_root)
        openpi_client = OpenPIChunkClient(host=args.openpi_host, port=args.openpi_port)
    else:
        openpi_client = None

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_ids = None if (args.all or args.task_id is None) else [int(args.task_id)]
    specs = resolve_task_specs(
        suite_name=args.suite_name,
        task_ids=task_ids,
        libero_root=args.libero_root,
        openpi_root=args.openpi_root,
        libero_config_dir=args.libero_config_dir,
        libero_datasets_root=args.libero_datasets_root,
    )

    task_iter: Iterable = _progress(specs, total=len(specs), desc="Tasks", unit="task", leave=True)
    for task_spec in task_iter:
        if not task_spec.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {task_spec.dataset_path}")

        task_output_dir = output_root / task_spec.task_key
        task_output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        manifest_files = []
        total_frames = 0
        with h5py.File(task_spec.dataset_path, "r") as dataset_file:
            demo_names = _sorted_demo_names(dataset_file)
            if args.max_episodes is not None:
                demo_names = demo_names[: int(args.max_episodes)]

            print(
                f"[task] {task_spec.task_key}: episodes={len(demo_names)} "
                f"openpi_precompute={openpi_client is not None}"
            )
            episode_iter: Iterator[tuple[int, str]] | Iterable[tuple[int, str]]
            episode_iter = enumerate(demo_names)
            episode_iter = _progress(
                episode_iter,
                total=len(demo_names),
                desc=f"{task_spec.task_key} episodes",
                unit="ep",
                leave=True,
            )
            for episode_index, demo_name in episode_iter:
                payload = _convert_demo(
                    dataset_file["data"][demo_name],
                    suite_name=task_spec.suite_name,
                    task_id=task_spec.task_id,
                    task_name=task_spec.task_name,
                    task_description=task_spec.task_description,
                    dataset_path=task_spec.dataset_path,
                    episode_index=episode_index,
                    chunk_horizon=int(args.chunk_horizon),
                    openpi_client=openpi_client,
                    demo_name=demo_name,
                )
                total_frames += int(payload["actions"].shape[0])
                episode_path = task_output_dir / f"episode_{episode_index:06d}.pkl"
                with open(episode_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                manifest_files.append(str(episode_path))

        manifest = {
            "format": "libero_offline_manifest_v1",
            "task_key": task_spec.task_key,
            "suite_name": task_spec.suite_name,
            "task_id": int(task_spec.task_id),
            "task_name": task_spec.task_name,
            "task_description": task_spec.task_description,
            "dataset_path": str(task_spec.dataset_path),
            "chunk_horizon": int(args.chunk_horizon),
            "openpi_precomputed": bool(openpi_client is not None),
            "openpi_host": args.openpi_host,
            "openpi_port": int(args.openpi_port),
            "num_episodes": len(manifest_files),
            "total_frames": int(total_frames),
            "episode_files": manifest_files,
            "elapsed_sec": float(time.time() - t0),
        }
        manifest_path = task_output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(
            f"Converted {task_spec.task_key}: episodes={manifest['num_episodes']} "
            f"frames={manifest['total_frames']} manifest={manifest_path}"
        )


if __name__ == "__main__":
    main()
