#!/usr/bin/env python3
"""Materialize unified residual-training episode PKLs from LIBERO HDF5 demos."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, TypeVar

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.policy.base import PolicyClient
from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.action_spec import resolve_control_indices
from serl_launcher.residual.data.materialize import build_residual_training_manifest
from serl_launcher.residual.data.materialize import materialize_with_config
from serl_torch.examples.libero.hdf5_utils import resolve_task_specs
from serl_torch.examples.libero.training_config import LIBERO_OFFLINE_TRAINING_CONFIG
from serl_torch.examples.libero.runtime.policy_adapter import build_libero_policy_input

_T = TypeVar("_T")


def _progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Iterable[_T]:
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


def _parse_csv_floats(value: Optional[str]) -> Optional[Sequence[float]]:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    return [float(part) for part in parts if part]


def _parse_csv_bools(value: Optional[str]) -> Optional[Sequence[bool]]:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    parsed = []
    for part in parts:
        if not part:
            continue
        token = part.lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            parsed.append(True)
            continue
        if token in {"0", "false", "f", "no", "n", "off"}:
            parsed.append(False)
            continue
        raise ValueError(
            "action_mask entries must be booleans like true/false or 1/0, "
            f"got {part!r}"
        )
    return parsed


def _build_frame_obs(payload: dict, frame_idx: int) -> dict:
    return {
        "agentview_rgb": payload["agentview_rgb"][frame_idx],
        "eye_in_hand_rgb": payload["eye_in_hand_rgb"][frame_idx],
        "ee_pos": payload["ee_pos"][frame_idx],
        "ee_ori": payload["ee_ori"][frame_idx],
        "gripper_states": payload["gripper_states"][frame_idx],
    }


def _build_policy_cfg_from_args(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "policy": {
                "type": str(args.policy_type),
                "id": (
                    str(args.policy_id).strip()
                    if args.policy_id is not None
                    else str(args.policy_type)
                ),
            },
            "openpi": {
                "host": str(args.openpi_host),
                "port": int(args.openpi_port),
            },
        }
    )


def _precompute_base_chunks(
    payload: dict,
    *,
    policy_client: PolicyClient,
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
        action_chunk, _ = policy_client.infer_chunk(
            build_libero_policy_input(
                obs_raw,
                str(payload["task_description"]),
            )
        )
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
    policy_client: PolicyClient,
    policy_backend_info: dict,
    demo_name: str,
    residual_alpha: float,
    action_mask: np.ndarray,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
) -> dict:
    obs = demo["obs"]
    agentview_rgb = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
    eye_in_hand_rgb = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
    ee_ori = np.asarray(obs["ee_ori"], dtype=np.float32)
    gripper_states = np.asarray(obs["gripper_states"], dtype=np.float32)
    expert_actions = np.asarray(demo["actions"], dtype=np.float32)
    num_steps = int(expert_actions.shape[0])
    rewards = np.asarray(
        demo.get("rewards", np.zeros((num_steps,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if rewards.shape[0] < num_steps:
        padded_rewards = np.zeros((num_steps,), dtype=np.float32)
        if rewards.shape[0] > 0:
            padded_rewards[: rewards.shape[0]] = rewards
        rewards = padded_rewards
    dones = np.asarray(
        demo.get("dones", np.zeros((num_steps,), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if dones.shape[0] < num_steps:
        padded_dones = np.zeros((num_steps,), dtype=bool)
        if dones.shape[0] > 0:
            padded_dones[: dones.shape[0]] = dones
        dones = padded_dones
    if num_steps > 0:
        rewards[-1] = 1.0
        dones[-1] = True

    raw_payload = {
        "task_description": task_description,
        "agentview_rgb": agentview_rgb,
        "eye_in_hand_rgb": eye_in_hand_rgb,
        "ee_pos": ee_pos,
        "ee_ori": ee_ori,
        "gripper_states": gripper_states,
        "actions": expert_actions,
    }
    base_chunks = _precompute_base_chunks(
        raw_payload,
        policy_client=policy_client,
        chunk_horizon=chunk_horizon,
        progress_desc=f"{task_name} ep={episode_index:03d}",
    )

    task_key = f"{suite_name}_task_{int(task_id)}"
    return materialize_with_config(
        {
            "source": "offline",
            "suite_name": suite_name,
            "task_id": int(task_id),
            "task_key": task_key,
            "task_description": task_description,
            "prompt": task_description,
            "alpha": float(residual_alpha),
            "agentview_rgb": agentview_rgb,
            "eye_in_hand_rgb": eye_in_hand_rgb,
            "ee_pos": ee_pos,
            "ee_ori": ee_ori,
            "gripper_states": gripper_states,
            "base_chunks": base_chunks,
            "actions": expert_actions,
            "rewards": rewards,
            "dones": dones,
            "episode_index": int(episode_index),
            "episode_steps": int(num_steps),
            "episode_return": float(np.sum(rewards)) if rewards.size > 0 else 0.0,
            "episode_success": bool(num_steps > 0),
            "metadata": {
                "source_episode_format": "libero_hdf5_demo",
                "base_policy_type": str(policy_backend_info["type"]),
                "base_policy_id": str(policy_backend_info["id"]),
                "task_name": str(task_name),
                "dataset_path": str(dataset_path),
                "demo_name": str(demo_name),
                "projection": {
                    "action_mask": [
                        bool(v) for v in np.asarray(action_mask, dtype=bool).tolist()
                    ],
                    "control_indices": [
                        int(v)
                        for v in np.asarray(control_indices, dtype=np.int64).reshape(-1)
                    ],
                    "residual_limits": [
                        float(v)
                        for v in np.asarray(residual_limits, dtype=np.float32).reshape(-1)
                    ],
                    "expert_reference_scale": float(expert_reference_scale),
                    "clip_residual_to_unit": bool(clip_residual_to_unit),
                },
            },
        },
        data_config=LIBERO_OFFLINE_TRAINING_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 demos into unified residual-training episode PKLs"
    )
    parser.add_argument("--suite_name", type=str, default="libero_10")
    parser.add_argument(
        "--task_id", type=int, default=None, help="Single task id to convert"
    )
    parser.add_argument("--all", action="store_true", help="Convert all tasks in the suite")
    parser.add_argument("--libero_root", type=str, default=None)
    parser.add_argument("--libero_config_dir", type=str, default=None)
    parser.add_argument("--libero_datasets_root", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "residual_training"
            / "offline"
        ),
    )
    parser.add_argument("--chunk_horizon", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--policy_type",
        type=str,
        default="openpi",
        help="Chunk policy backend type used to materialize base_chunks.",
    )
    parser.add_argument(
        "--policy_id",
        type=str,
        default=None,
        help="Semantic base-policy label recorded in exported metadata.",
    )
    parser.add_argument(
        "--openpi_host",
        type=str,
        required=True,
        help="OpenPI host used when policy_type=openpi.",
    )
    parser.add_argument("--openpi_port", type=int, default=30001)
    parser.add_argument(
        "--residual_alpha",
        type=float,
        required=True,
        help="Residual alpha baked into the exported training payload",
    )
    parser.add_argument(
        "--action_mask",
        type=str,
        default=None,
        help=(
            "Comma-separated boolean residual control mask, e.g. "
            "true,true,true,true,true,true,false. Defaults to all env action dims."
        ),
    )
    parser.add_argument(
        "--action_limits",
        type=str,
        default=None,
        help="Comma-separated residual limits. Defaults to 1.0 for every env action dim.",
    )
    parser.add_argument("--expert_reference_scale", type=float, default=1.0)
    parser.add_argument(
        "--clip_residual_to_unit",
        dest="clip_residual_to_unit",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_clip_residual_to_unit",
        dest="clip_residual_to_unit",
        action="store_false",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_materialize_residual_training_offline")

    policy_cfg = _build_policy_cfg_from_args(args)
    policy_backend_info = build_policy_backend_info(policy_cfg)
    policy_client = build_policy_client(policy_cfg, logger=logger)
    logger.info(
        "Chunk policy backend: type=%s id=%s",
        policy_backend_info["type"],
        policy_backend_info["id"],
    )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_ids = None if (args.all or args.task_id is None) else [int(args.task_id)]
    specs = resolve_task_specs(
        suite_name=args.suite_name,
        task_ids=task_ids,
        libero_root=args.libero_root,
        libero_config_dir=args.libero_config_dir,
        libero_datasets_root=args.libero_datasets_root,
    )

    task_iter: Iterable = _progress(
        specs, total=len(specs), desc="Tasks", unit="task", leave=True
    )
    parsed_action_mask = _parse_csv_bools(args.action_mask)
    for task_spec in task_iter:
        if not task_spec.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {task_spec.dataset_path}")

        task_output_dir = output_root / task_spec.task_key
        task_output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        manifest_files = []
        total_frames = 0
        action_dim_for_manifest = None
        with h5py.File(task_spec.dataset_path, "r") as dataset_file:
            demo_names = _sorted_demo_names(dataset_file)
            if args.max_episodes is not None:
                demo_names = demo_names[: int(args.max_episodes)]

            print(
                f"[task] {task_spec.task_key}: episodes={len(demo_names)} "
                f"policy_precompute={policy_backend_info['type']}"
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
                demo = dataset_file["data"][demo_name]
                demo_actions = np.asarray(demo["actions"], dtype=np.float32)
                full_action_dim = int(demo_actions.shape[1])
                control_indices = resolve_control_indices(
                    full_action_dim=full_action_dim,
                    action_mask=parsed_action_mask,
                )
                resolved_action_mask = np.isin(
                    np.arange(full_action_dim, dtype=np.int64),
                    np.asarray(control_indices, dtype=np.int64),
                )
                action_limits = _parse_csv_floats(args.action_limits)
                if action_limits is None:
                    action_limits = [1.0] * full_action_dim
                residual_limits = build_residual_limits(
                    np.asarray(control_indices, dtype=np.int64),
                    full_action_dim=full_action_dim,
                    action_limits=action_limits,
                )
                payload = _convert_demo(
                    demo,
                    suite_name=task_spec.suite_name,
                    task_id=task_spec.task_id,
                    task_name=task_spec.task_name,
                    task_description=task_spec.task_description,
                    dataset_path=task_spec.dataset_path,
                    episode_index=episode_index,
                    chunk_horizon=int(args.chunk_horizon),
                    policy_client=policy_client,
                    policy_backend_info=policy_backend_info,
                    demo_name=demo_name,
                    residual_alpha=float(args.residual_alpha),
                    action_mask=np.asarray(resolved_action_mask, dtype=bool),
                    control_indices=np.asarray(control_indices, dtype=np.int64),
                    residual_limits=np.asarray(residual_limits, dtype=np.float32),
                    expert_reference_scale=float(args.expert_reference_scale),
                    clip_residual_to_unit=bool(args.clip_residual_to_unit),
                )
                total_frames += int(payload["action"]["final"].shape[0])
                action_dim_for_manifest = int(payload["action"]["final"].shape[1])
                episode_path = task_output_dir / f"episode_{episode_index:06d}.pkl"
                with open(episode_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                manifest_files.append(str(episode_path))

        manifest = build_residual_training_manifest(
            schema=LIBERO_OFFLINE_TRAINING_CONFIG.schema,
            source="offline",
            task_key=task_spec.task_key,
            suite_name=task_spec.suite_name,
            task_id=int(task_spec.task_id),
            task_description=task_spec.task_description,
            chunk_horizon=int(args.chunk_horizon),
            action_dim=(
                int(action_dim_for_manifest) if action_dim_for_manifest is not None else 0
            ),
            num_episodes=len(manifest_files),
            total_frames=int(total_frames),
            episode_files=manifest_files,
            metadata={
                "base_policy_type": str(policy_backend_info["type"]),
                "base_policy_id": str(policy_backend_info["id"]),
                "task_name": task_spec.task_name,
                "dataset_path": str(task_spec.dataset_path),
                "residual_alpha": float(args.residual_alpha),
                "expert_reference_scale": float(args.expert_reference_scale),
                "clip_residual_to_unit": bool(args.clip_residual_to_unit),
                "elapsed_sec": float(time.time() - t0),
            },
        )
        if str(policy_backend_info["type"]) == "openpi":
            manifest["metadata"]["openpi_host"] = args.openpi_host
            manifest["metadata"]["openpi_port"] = int(args.openpi_port)
        manifest_path = task_output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(
            f"Converted {task_spec.task_key}: episodes={manifest['num_episodes']} "
            f"frames={manifest['total_frames']} manifest={manifest_path}"
        )


if __name__ == "__main__":
    main()
