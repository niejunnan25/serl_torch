#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


VIEW_DATASETS = {
    "agentview": "agentview_rgb",
    "wrist": "eye_in_hand_rgb",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export step-indexed LIBERO HDF5 episode images for manual milestone "
            "annotation."
        )
    )
    parser.add_argument("--hdf5", required=True, help="Input LIBERO demo HDF5 file.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for exported episode images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return (int(name.split("_", 1)[1]), name)
        except ValueError:
            pass
    return (10**9, name)


def _ensure_uint8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected image as HWC/CHW array, got shape={arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _restore_libero_view(image: np.ndarray) -> np.ndarray:
    # LIBERO camera arrays are stored flipped relative to the rendered view used
    # by the training/eval pipeline.
    arr = _ensure_uint8_hwc(image)
    return np.ascontiguousarray(arr[::-1, ::-1])


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_restore_libero_view(image)).save(path)


def _dataset_or_none(group: h5py.Group, name: str) -> h5py.Dataset | None:
    obs = group.get("obs", None)
    if not isinstance(obs, h5py.Group):
        return None
    ds = obs.get(name, None)
    return ds if isinstance(ds, h5py.Dataset) else None


def main() -> None:
    args = _parse_args()
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {hdf5_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Pass --overwrite to continue."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    demo_summaries: list[dict[str, Any]] = []
    total_frames = 0

    with h5py.File(hdf5_path, "r") as h5:
        data = h5.get("data", None)
        if not isinstance(data, h5py.Group):
            raise KeyError(f"{hdf5_path} does not contain a top-level 'data' group")

        demo_names = sorted(data.keys(), key=_demo_sort_key)
        for demo_name in demo_names:
            demo_group = data[demo_name]
            if not isinstance(demo_group, h5py.Group):
                continue

            datasets: dict[str, h5py.Dataset] = {}
            for view, dataset_name in VIEW_DATASETS.items():
                ds = _dataset_or_none(demo_group, dataset_name)
                if ds is None:
                    raise KeyError(
                        f"Missing data/{demo_name}/obs/{dataset_name}. "
                        f"Available obs keys: {list(demo_group.get('obs', {}).keys())}"
                    )
                if ds.ndim != 4:
                    raise ValueError(
                        f"Expected data/{demo_name}/obs/{dataset_name} to be rank 4, "
                        f"got shape={ds.shape}"
                    )
                datasets[view] = ds

            lengths = {view: int(ds.shape[0]) for view, ds in datasets.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"View lengths do not match for {demo_name}: {lengths}")
            num_steps = next(iter(lengths.values()))

            actions = demo_group.get("actions", None)
            rewards = demo_group.get("rewards", None)
            dones = demo_group.get("dones", None)
            action_len = int(actions.shape[0]) if isinstance(actions, h5py.Dataset) else None
            if action_len is not None and action_len != num_steps:
                raise ValueError(
                    f"Image/action length mismatch for {demo_name}: "
                    f"images={num_steps} actions={action_len}"
                )

            for step in range(num_steps):
                row: dict[str, Any] = {
                    "demo": demo_name,
                    "step": int(step),
                }
                if isinstance(actions, h5py.Dataset):
                    action = np.asarray(actions[step]).reshape(-1)
                    row["action_gripper"] = float(action[-1]) if action.size else ""
                if isinstance(rewards, h5py.Dataset):
                    row["reward"] = float(np.asarray(rewards[step]).reshape(()))
                if isinstance(dones, h5py.Dataset):
                    row["done"] = int(np.asarray(dones[step]).reshape(()))

                for view, ds in datasets.items():
                    image = _ensure_uint8_hwc(ds[step])
                    image_path = (
                        out_dir
                        / demo_name
                        / view
                        / f"step_{int(step):04d}_{view}.png"
                    )
                    _write_png(image_path, image)
                    row[f"{view}_image"] = str(image_path)

                rows.append(row)
                total_frames += 1

            demo_summaries.append(
                {
                    "demo": demo_name,
                    "num_steps": int(num_steps),
                    "views": dict(lengths),
                }
            )
            print(f"exported {demo_name}: steps={num_steps}")

    manifest = {
        "source_hdf5": str(hdf5_path),
        "output_dir": str(out_dir),
        "num_demos": int(len(demo_summaries)),
        "total_steps": int(total_frames),
        "views": sorted(VIEW_DATASETS),
        "images_per_step": int(len(VIEW_DATASETS)),
        "orientation": "rotated_180_to_match_libero_training_view",
        "demos": demo_summaries,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    csv_path = out_dir / "annotation_index.csv"
    fieldnames = [
        "demo",
        "step",
        "action_gripper",
        "reward",
        "done",
        "agentview_image",
        "wrist_image",
        "stage1_done",
        "notes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(out_row)

    print("done")
    print(f"source_hdf5={hdf5_path}")
    print(f"output_dir={out_dir}")
    print(f"num_demos={len(demo_summaries)}")
    print(f"total_steps={total_frames}")
    print(f"annotation_index={csv_path}")
    print(f"manifest={out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
