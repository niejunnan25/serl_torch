#!/usr/bin/env python3
"""Prepare demo-derived residual-training episode PKLs from AgiBot demo PKLs."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[5]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[4] / "serl_launcher"
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.policy.base import PolicyClient
from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.action_spec import resolve_control_indices
from serl_launcher.residual.data.materialize import build_residual_training_manifest
from serl_launcher.residual.data.materialize import materialize_with_config
from serl_torch.examples.agibot_real.config import resolve_agibot_cfg_task_key
from serl_torch.examples.agibot_real.runtime.policy_adapter import build_agibot_policy_input
from serl_torch.examples.agibot_real.training_config import AGIBOT_OFFLINE_TRAINING_CONFIG

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
        raise ValueError(f"action_mask entries must be booleans, got {part!r}")
    return parsed


def _build_policy_cfg_from_args(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "policy": {
                "type": str(args.policy_type),
                "id": str(args.policy_id).strip() if args.policy_id is not None else str(args.policy_type),
            },
            "openpi": {
                "host": str(args.openpi_host),
                "port": int(args.openpi_port),
            },
        }
    )


def _find_first_key(mapping: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing keys {tuple(candidates)} in {list(mapping.keys())}")


def _canonicalize_obs_frame(obs_raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    pose = np.asarray(_find_first_key(obs_raw, ("state/pose", "observation/state", "pose")), dtype=np.float32).reshape(-1)
    if pose.shape[0] != 14:
        raise ValueError(f"AgiBot offline observation pose must be 14D, got {pose.shape}")
    return {
        "image/head": np.asarray(
            _find_first_key(obs_raw, ("image/head", "head_image", "observation/image")),
            dtype=np.uint8,
        ).copy(),
        "image/left_wrist": np.asarray(
            _find_first_key(
                obs_raw,
                ("image/left_wrist", "left_wrist_image", "observation/wrist_left_image"),
            ),
            dtype=np.uint8,
        ).copy(),
        "image/right_wrist": np.asarray(
            _find_first_key(
                obs_raw,
                ("image/right_wrist", "right_wrist_image", "observation/wrist_right_image"),
            ),
            dtype=np.uint8,
        ).copy(),
        "state/pose": pose.copy(),
    }


def _canonicalize_action(action: Any) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] == 18:
        action_arr = action_arr[:14]
    if action_arr.shape[0] != 14:
        raise ValueError(f"AgiBot offline action must be 14D or 18D, got {action_arr.shape}")
    return action_arr.astype(np.float32)


def _precompute_base_chunks(
    episode_frames: Sequence[Dict[str, np.ndarray]],
    *,
    prompt: str,
    policy_client: PolicyClient,
    chunk_horizon: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    chunks = []
    chunk_starts: Iterable[int] = range(0, len(episode_frames), chunk_horizon)
    total_chunks = (len(episode_frames) + chunk_horizon - 1) // chunk_horizon
    if progress_desc is not None:
        chunk_starts = _progress(
            chunk_starts,
            total=total_chunks,
            desc=progress_desc,
            unit="chunk",
            leave=False,
        )
    for chunk_start in chunk_starts:
        obs_raw = episode_frames[chunk_start]
        action_chunk, _ = policy_client.infer_chunk(
            build_agibot_policy_input(
                obs_raw,
                prompt,
            )
        )
        chunks.append(select_action_chunk_window(action_chunk, horizon=chunk_horizon))
    return np.asarray(chunks, dtype=np.float32)


def _iter_input_files(
    *,
    demo_paths: Sequence[str],
    input_dir: Optional[str],
    file_glob: str,
) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw_path in demo_paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        yield path
    if input_dir is not None:
        root = Path(input_dir).expanduser().resolve()
        for path in sorted(root.glob(file_glob)):
            if path in seen:
                continue
            seen.add(path)
            yield path


def _load_transition_list(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list) or (payload and not isinstance(payload[0], dict)):
        raise TypeError(f"Expected a list of transition dicts in {path}")
    return payload


def _split_episodes(transitions: Sequence[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    current: list[dict[str, Any]] = []
    for transition in transitions:
        current.append(dict(transition))
        done = bool(transition.get("dones", False))
        truncated = bool(transition.get("truncated", False))
        if done or truncated:
            yield current
            current = []
    if current:
        yield current


def _convert_episode(
    episode_transitions: Sequence[dict[str, Any]],
    *,
    task_key: str,
    task_name: str,
    task_description: str,
    dataset_path: Path,
    episode_index: int,
    chunk_horizon: int,
    policy_client: PolicyClient,
    policy_backend_info: dict,
    residual_alpha: float,
    action_mask: np.ndarray,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
) -> dict:
    episode_frames: list[dict[str, np.ndarray]] = []
    expert_actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    for transition in episode_transitions:
        episode_frames.append(
            _canonicalize_obs_frame(dict(transition["observations"]))
        )
        expert_actions.append(_canonicalize_action(transition["actions"]))
        rewards.append(float(transition.get("rewards", 0.0)))
        dones.append(bool(transition.get("dones", False)))
    if not expert_actions:
        raise ValueError(f"Episode {episode_index} from {dataset_path} is empty")

    base_chunks = _precompute_base_chunks(
        episode_frames,
        prompt=task_description,
        policy_client=policy_client,
        chunk_horizon=chunk_horizon,
        progress_desc=f"{task_name} ep={episode_index:03d}",
    )

    head_images = np.stack([frame["image/head"] for frame in episode_frames], axis=0)
    left_wrist_images = np.stack([frame["image/left_wrist"] for frame in episode_frames], axis=0)
    right_wrist_images = np.stack([frame["image/right_wrist"] for frame in episode_frames], axis=0)
    pose = np.stack([frame["state/pose"] for frame in episode_frames], axis=0)
    expert_actions_arr = np.stack(expert_actions, axis=0).astype(np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)
    dones_arr[-1] = True

    return materialize_with_config(
        {
            "source": "offline",
            "suite_name": "agibot_real",
            "task_id": 0,
            "task_key": task_key,
            "task_description": task_description,
            "prompt": task_description,
            "alpha": float(residual_alpha),
            "head_image": head_images,
            "left_wrist_image": left_wrist_images,
            "right_wrist_image": right_wrist_images,
            "pose": pose,
            "base_chunks": base_chunks,
            "actions": expert_actions_arr,
            "rewards": rewards_arr,
            "dones": dones_arr,
            "episode_index": int(episode_index),
            "episode_steps": int(expert_actions_arr.shape[0]),
            "episode_return": float(np.sum(rewards_arr)),
            "episode_success": bool(np.any(rewards_arr > 0.0) or np.any(dones_arr)),
            "metadata": {
                "source_episode_format": "agibot_transition_pkl",
                "base_policy_type": str(policy_backend_info["type"]),
                "base_policy_id": str(policy_backend_info["id"]),
                "task_name": str(task_name),
                "dataset_path": str(dataset_path),
                "projection": {
                    "action_mask": [bool(v) for v in np.asarray(action_mask, dtype=bool).tolist()],
                    "control_indices": [int(v) for v in np.asarray(control_indices, dtype=np.int64).reshape(-1)],
                    "residual_limits": [float(v) for v in np.asarray(residual_limits, dtype=np.float32).reshape(-1)],
                    "expert_reference_scale": float(expert_reference_scale),
                    "clip_residual_to_unit": bool(clip_residual_to_unit),
                },
            },
        },
        data_config=AGIBOT_OFFLINE_TRAINING_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare AgiBot offline demo PKLs into residual-training episode files",
    )
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_key", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--demo_paths", type=str, nargs="*", default=[])
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--glob", type=str, default="*.pkl")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--chunk_horizon", type=int, default=5)
    parser.add_argument("--policy_type", type=str, default="openpi")
    parser.add_argument("--policy_id", type=str, default="pi05_agibot")
    parser.add_argument("--openpi_host", type=str, default="127.0.0.1")
    parser.add_argument("--openpi_port", type=int, default=30001)
    parser.add_argument("--residual_alpha", type=float, default=0.2)
    parser.add_argument("--action_mask", type=str, default=None)
    parser.add_argument("--action_limits", type=str, default=None)
    parser.add_argument("--expert_reference_scale", type=float, default=1.0)
    parser.add_argument("--clip_residual_to_unit", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("agibot_prepare_offline_demos")

    input_files = list(
        _iter_input_files(
            demo_paths=args.demo_paths,
            input_dir=args.input_dir,
            file_glob=str(args.glob),
        )
    )
    if not input_files:
        raise ValueError("No input demo PKL files were found")

    policy_cfg = _build_policy_cfg_from_args(args)
    policy_client = build_policy_client(policy_cfg, logger=logger)
    policy_backend_info = build_policy_backend_info(policy_cfg)

    action_dim = 14
    action_mask = np.asarray(
        _parse_csv_bools(args.action_mask)
        if args.action_mask is not None
        else [True] * action_dim,
        dtype=bool,
    )
    control_indices = resolve_control_indices(
        full_action_dim=action_dim,
        action_mask=[bool(v) for v in action_mask.tolist()],
    )
    residual_limits = build_residual_limits(
        control_indices,
        action_limits=_parse_csv_floats(args.action_limits),
        full_action_dim=action_dim,
    )

    task_cfg = OmegaConf.create(
        {
            "task": {
                "name": str(args.task_name),
                "task_key": args.task_key,
            }
        }
    )
    task_key = resolve_agibot_cfg_task_key(task_cfg)
    task_description = str(args.prompt or args.task_name)

    output_root = Path(args.output_dir).expanduser().resolve()
    task_output_dir = output_root / task_key
    task_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[str] = []
    total_frames = 0
    total_episodes = 0
    global_episode_index = 0

    file_iter = _progress(input_files, total=len(input_files), desc="files", unit="file")
    for dataset_path in file_iter:
        transitions = _load_transition_list(dataset_path)
        for episode_transitions in _split_episodes(transitions):
            if args.max_episodes is not None and total_episodes >= int(args.max_episodes):
                break
            payload = _convert_episode(
                episode_transitions,
                task_key=task_key,
                task_name=str(args.task_name),
                task_description=task_description,
                dataset_path=dataset_path,
                episode_index=global_episode_index,
                chunk_horizon=int(args.chunk_horizon),
                policy_client=policy_client,
                policy_backend_info=policy_backend_info,
                residual_alpha=float(args.residual_alpha),
                action_mask=action_mask,
                control_indices=control_indices,
                residual_limits=residual_limits,
                expert_reference_scale=float(args.expert_reference_scale),
                clip_residual_to_unit=bool(args.clip_residual_to_unit),
            )
            file_name = f"episode_{global_episode_index:06d}.pkl"
            episode_path = task_output_dir / file_name
            with episode_path.open("wb") as f:
                pickle.dump(payload, f)
            manifest_files.append(file_name)
            episode_steps = int(payload["episode"]["steps"])
            total_frames += episode_steps
            total_episodes += 1
            global_episode_index += 1
        if args.max_episodes is not None and total_episodes >= int(args.max_episodes):
            break

    manifest = build_residual_training_manifest(
        schema=AGIBOT_OFFLINE_TRAINING_CONFIG.schema,
        source="offline",
        task_key=task_key,
        suite_name="agibot_real",
        task_id=0,
        task_description=task_description,
        chunk_horizon=int(args.chunk_horizon),
        action_dim=14,
        num_episodes=int(total_episodes),
        total_frames=int(total_frames),
        episode_files=manifest_files,
        metadata={
            "task_name": str(args.task_name),
            "base_policy_type": str(policy_backend_info["type"]),
            "base_policy_id": str(policy_backend_info["id"]),
        },
    )
    manifest_path = task_output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(
        "Converted offline demos: task_key=%s episodes=%s frames=%s manifest=%s",
        task_key,
        total_episodes,
        total_frames,
        manifest_path,
    )


if __name__ == "__main__":
    main()

