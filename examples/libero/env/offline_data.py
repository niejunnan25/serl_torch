from __future__ import annotations

"""LIBERO-specific offline data preparation and loading helpers."""

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from serl_launcher.data.offline_prepared import EPISODE_FILE_GLOB
from serl_launcher.data.offline_prepared import MANIFEST_FILENAME
from serl_launcher.data.offline_prepared import OfflinePreparedResolution
from serl_launcher.data.offline_prepared import build_residual_prepared_fingerprint
from serl_launcher.data.offline_prepared import build_residual_training_signature
from serl_launcher.data.offline_prepared import extract_residual_manifest_signature
from serl_launcher.data.offline_prepared import (
    load_prepared_offline_replay as _load_prepared_offline_replay,
)
from serl_launcher.data.offline_prepared import read_manifest
from serl_launcher.data.offline_prepared import (
    resolve_prepared_episode_files as _resolve_prepared_episode_files,
)
from serl_launcher.data.offline_prepared import resolve_prepared_path_value
from serl_launcher.data.offline_prepared import resolve_residual_prepared_dir
from serl_launcher.data.offline_prepared import validate_prepared_paths
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_id
from serl_launcher.policy.typed_factory import resolve_policy_backend_type
from serl_launcher.residual.expert_projection import project_expert_action
from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.path_utils import resolve_original_cwd
from serl_launcher.utils.path_utils import resolve_path
from serl_launcher.utils.serialization import to_jsonable

from ..config import LiberoTrainConfig
from .observation import build_libero_state
from .observation import extract_libero_images
from .policy_input import build_libero_policy_input
from .setup import resolve_libero_config_dir
from .setup import resolve_libero_datasets_root
from .setup import resolve_libero_root
from .setup import setup_libero_pythonpath

OFFLINE_FORMAT_VERSION = "libero_offline_step_transitions_v1"


@dataclasses.dataclass(frozen=True, slots=True)
class LiberoTaskSpec:
    suite_name: str
    task_id: int
    task_name: str
    task_description: str
    dataset_path: Path

    @property
    def task_key(self) -> str:
        return f"{self.suite_name}_task_{self.task_id}"


def _candidate_dataset_paths(
    datasets_root: Path,
    suite_name: str,
    task_name: str,
) -> Iterator[Path]:
    filename = f"{task_name}_demo.hdf5"
    yield (datasets_root / suite_name / filename).resolve()
    yield (datasets_root / filename).resolve()


def resolve_task_spec(cfg: LiberoTrainConfig) -> LiberoTaskSpec:
    resolved_libero_root = resolve_libero_root(cfg.libero_root)
    resolved_config_dir = resolve_libero_config_dir(cfg.libero_config_dir)
    resolved_datasets_root = resolve_libero_datasets_root(
        cfg.libero_datasets_root,
        libero_root=resolved_libero_root,
    )
    setup_libero_pythonpath(
        resolved_libero_root,
        resolved_config_dir,
        datasets_root=resolved_datasets_root,
    )

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[str(cfg.task.suite_name)]()
    task = suite.get_task(int(cfg.task.task_id))

    dataset_override = cfg.offline.prepare.raw_dataset_path
    if dataset_override is not None:
        dataset_path = resolve_path(dataset_override, base=resolve_original_cwd())
    else:
        dataset_path = None
        for candidate in _candidate_dataset_paths(
            resolved_datasets_root,
            str(cfg.task.suite_name),
            str(task.name),
        ):
            if candidate.exists():
                dataset_path = candidate
                break
        if dataset_path is None:
            dataset_path = next(
                _candidate_dataset_paths(
                    resolved_datasets_root,
                    str(cfg.task.suite_name),
                    str(task.name),
                )
            )

    return LiberoTaskSpec(
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        task_name=str(task.name),
        task_description=str(task.language),
        dataset_path=dataset_path.resolve(),
    )


def prepare_fingerprint(
    cfg: LiberoTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
    offline_format_version: str,
) -> dict[str, Any]:
    return build_residual_prepared_fingerprint(
        format_version=offline_format_version,
        task_key=str(task_spec.task_key),
        task_description=str(task_spec.task_description),
        policy_backend_type=resolve_policy_backend_type(cfg),
        policy_backend_id=resolve_policy_backend_id(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        action_dim=int(cfg.env.action_dim),
        alpha=float(cfg.residual.alpha),
        action_mask=cfg.residual.action_mask,
        action_limits=cfg.residual.action_limits,
        clip_gripper=bool(cfg.residual.clip_gripper),
        expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
        clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
        filter_unrepresentable_steps=bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        image_keys=cfg.obs.image_keys,
        vector_obs_keys=cfg.obs.vector_obs_keys,
        raw_dataset_path=task_spec.dataset_path,
    )


def training_compatibility_signature(cfg: LiberoTrainConfig) -> dict[str, Any]:
    return build_residual_training_signature(
        task_key=f"{cfg.task.suite_name}_task_{cfg.task.task_id}",
        policy_backend_type=resolve_policy_backend_type(cfg),
        policy_backend_id=resolve_policy_backend_id(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        action_dim=int(cfg.env.action_dim),
        alpha=float(cfg.residual.alpha),
        action_mask=cfg.residual.action_mask,
        action_limits=cfg.residual.action_limits,
        clip_gripper=bool(cfg.residual.clip_gripper),
        expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
        clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
        filter_unrepresentable_steps=bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        image_keys=cfg.obs.image_keys,
        vector_obs_keys=cfg.obs.vector_obs_keys,
    )


def prepared_dir_for_cfg(
    cfg: LiberoTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
) -> Path:
    return resolve_residual_prepared_dir(
        output_root=cfg.offline.prepare.output_root,
        task_key=str(task_spec.task_key),
        policy_backend=describe_policy_backend(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        alpha=float(cfg.residual.alpha),
    )


def build_frame_obs(
    payload: dict[str, np.ndarray],
    frame_idx: int,
) -> dict[str, Any]:
    return {
        "agentview_rgb": payload["agentview_rgb"][frame_idx],
        "eye_in_hand_rgb": payload["eye_in_hand_rgb"][frame_idx],
        "ee_pos": payload["ee_pos"][frame_idx],
        "ee_ori": payload["ee_ori"][frame_idx],
        "gripper_states": payload["gripper_states"][frame_idx],
    }


def pad_or_truncate_1d(
    values: Any,
    *,
    length: int,
    dtype: Any,
    fill_value: float | bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape[0] == length:
        return arr
    out = np.full((int(length),), fill_value, dtype=dtype)
    if arr.shape[0] > 0:
        copy_len = min(int(arr.shape[0]), int(length))
        out[:copy_len] = arr[:copy_len]
    return out


def _precompute_base_chunks_for_steps(
    payload: dict[str, np.ndarray],
    *,
    task_prompt: str,
    policy_client: Any,
    chunk_horizon: int,
) -> np.ndarray:
    num_steps = int(payload["actions"].shape[0])
    base_chunks: list[np.ndarray] = []
    step_iter: Iterable[int] = tqdm(
        range(num_steps),
        total=num_steps,
        desc="offline base chunks",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    for step_idx in step_iter:
        obs_raw = build_frame_obs(payload, step_idx)
        robot_state = build_libero_state(obs_raw)
        image_observations = extract_libero_images(obs_raw)
        policy_input = build_libero_policy_input(
            prompt=task_prompt,
            state=robot_state,
            images=image_observations,
        )
        action_chunk, _ = policy_client.infer(policy_input)
        base_chunks.append(
            prepare_base_actions_chunk(
                base_actions=action_chunk,
                chunk_horizon=chunk_horizon,
                source="offline base policy",
            )
        )
    return np.asarray(base_chunks, dtype=np.float32)


def _zero_like_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs.items()}


def _load_demo_payload(demo: Any) -> dict[str, np.ndarray]:
    obs = demo["obs"]
    expert_actions = np.asarray(demo["actions"], dtype=np.float32)
    if expert_actions.ndim != 2 or expert_actions.shape[0] <= 0:
        raise ValueError(f"invalid demo action shape: {expert_actions.shape}")
    return {
        "agentview_rgb": np.asarray(obs["agentview_rgb"], dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8),
        "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float32),
        "ee_ori": np.asarray(obs["ee_ori"], dtype=np.float32),
        "gripper_states": np.asarray(obs["gripper_states"], dtype=np.float32),
        "actions": expert_actions,
    }


def prepare_demo_transitions(
    *,
    demo: Any,
    episode_id: int,
    task_prompt: str,
    action_spec: ResidualActionSpec,
    image_keys: tuple[str, ...],
    policy_client: Any,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    filter_unrepresentable_steps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_demo_payload(demo)
    expert_actions = payload["actions"]
    num_steps = int(expert_actions.shape[0])
    rewards_present = "rewards" in demo
    rewards = pad_or_truncate_1d(
        demo.get("rewards", np.zeros((num_steps,), dtype=np.float32)),
        length=num_steps,
        dtype=np.float32,
        fill_value=0.0,
    )
    if (not rewards_present) and num_steps > 0:
        rewards[-1] = 1.0
    dones = pad_or_truncate_1d(
        demo.get("dones", np.zeros((num_steps,), dtype=bool)),
        length=num_steps,
        dtype=bool,
        fill_value=False,
    )
    if num_steps > 0:
        dones[-1] = True

    base_chunks_per_step = _precompute_base_chunks_for_steps(
        payload,
        task_prompt=task_prompt,
        policy_client=policy_client,
        chunk_horizon=int(action_spec.chunk_horizon),
    )
    if base_chunks_per_step.shape[:2] != (
        int(num_steps),
        int(action_spec.chunk_horizon),
    ):
        raise ValueError(
            "Unexpected precomputed base chunk shape: "
            f"{base_chunks_per_step.shape} vs ({num_steps}, {int(action_spec.chunk_horizon)}, *)"
        )

    transitions: list[dict[str, Any]] = []
    unrepresentable_values = 0
    steps_unrepresentable = 0
    steps_filtered = 0
    episode_return = 0.0
    for step_idx in range(num_steps):
        obs_raw = build_frame_obs(payload, step_idx)
        base_chunk = np.asarray(base_chunks_per_step[step_idx], dtype=np.float32)
        residual_obs = build_chunk_residual_obs(
            robot_state=build_libero_state(obs_raw),
            images=extract_libero_images(obs_raw),
            image_keys=image_keys,
            base_actions=base_chunk,
            residual_alpha=float(action_spec.alpha),
        )

        final_action, step_unrepresentable_values, step_unrepresentable = project_expert_action(
            expert_action=expert_actions[step_idx],
            base_action=base_chunk[0],
            action_spec=action_spec,
            expert_reference_scale=expert_reference_scale,
            clip_residual_to_unit=clip_residual_to_unit,
        )
        unrepresentable_values += int(step_unrepresentable_values)
        if step_unrepresentable:
            steps_unrepresentable += 1
            if filter_unrepresentable_steps:
                steps_filtered += 1
                continue
        reward = float(rewards[step_idx])
        episode_return += reward
        done = bool(dones[step_idx]) or bool(step_idx >= (num_steps - 1))

        if done:
            next_residual_obs = _zero_like_obs(residual_obs)
            mask = 0.0
        else:
            next_obs_raw = build_frame_obs(payload, step_idx + 1)
            next_base_chunk = np.asarray(base_chunks_per_step[step_idx + 1], dtype=np.float32)
            next_residual_obs = build_chunk_residual_obs(
                robot_state=build_libero_state(next_obs_raw),
                images=extract_libero_images(next_obs_raw),
                image_keys=image_keys,
                base_actions=next_base_chunk,
                residual_alpha=float(action_spec.alpha),
            )
            mask = 1.0

        transitions.append(
            {
                "episode_id": int(episode_id),
                "episode_step": int(step_idx),
                "observations": residual_obs,
                "actions": np.asarray(final_action, dtype=np.float32).reshape(-1),
                "next_observations": next_residual_obs,
                "rewards": float(reward),
                "masks": float(mask),
                "dones": bool(done),
            }
        )

    episode_stats = {
        "episode_id": int(episode_id),
        "steps_total": int(num_steps),
        "steps_written": int(len(transitions)),
        "steps_unrepresentable": int(steps_unrepresentable),
        "steps_filtered": int(steps_filtered),
        "episode_return": float(episode_return),
        "success": bool(num_steps > 0),
        "unrepresentable_values": int(unrepresentable_values),
    }
    return transitions, episode_stats


def _manifest_signature(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return extract_residual_manifest_signature(manifest)


def resolve_configured_prepared_paths(cfg: LiberoTrainConfig) -> tuple[Path, ...]:
    return resolve_prepared_path_value(cfg.offline.prepared_path)


def resolve_and_validate_prepared_paths(
    cfg: LiberoTrainConfig,
    *,
    logger: logging.Logger,
) -> OfflinePreparedResolution:
    del logger
    return validate_prepared_paths(
        resolve_configured_prepared_paths(cfg),
        expected_signature=training_compatibility_signature(cfg),
        manifest_signature_fn=_manifest_signature,
        manifest_filename=MANIFEST_FILENAME,
    )


def write_manifest(
    *,
    manifest_path: Path,
    task_spec: LiberoTaskSpec,
    cfg: LiberoTrainConfig,
    fingerprint: dict[str, Any],
    episode_files: Sequence[Path],
    prepare_stats: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "format_version": OFFLINE_FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "task": {
            "suite_name": str(task_spec.suite_name),
            "task_id": int(task_spec.task_id),
            "task_key": str(task_spec.task_key),
            "task_name": str(task_spec.task_name),
            "task_description": str(task_spec.task_description),
            "dataset_path": str(task_spec.dataset_path),
        },
        "policy": {
            "type": resolve_policy_backend_type(cfg),
            "id": resolve_policy_backend_id(cfg),
            "backend": describe_policy_backend(cfg),
        },
        "residual": {
            "alpha": float(cfg.residual.alpha),
            "chunk_horizon": int(cfg.residual.chunk_horizon),
            "action_mask": (
                None
                if cfg.residual.action_mask is None
                else [bool(v) for v in cfg.residual.action_mask]
            ),
            "action_limits": [float(v) for v in cfg.residual.action_limits],
            "clip_gripper": bool(cfg.residual.clip_gripper),
        },
        "offline": {
            "expert_reference_scale": float(cfg.offline.prepare.expert_reference_scale),
            "clip_residual_to_unit": bool(cfg.offline.prepare.clip_residual_to_unit),
            "filter_unrepresentable_steps": bool(
                cfg.offline.prepare.filter_unrepresentable_steps
            ),
        },
        "fingerprint": to_jsonable(fingerprint),
        "prepare_stats": to_jsonable(prepare_stats),
        "episode_files": [str(path.name) for path in episode_files],
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    return manifest


def resolve_prepared_episode_files(paths: Sequence[Path]) -> list[Path]:
    return _resolve_prepared_episode_files(
        paths,
        manifest_filename=MANIFEST_FILENAME,
        episode_file_glob=EPISODE_FILE_GLOB,
        read_manifest_fn=read_manifest,
    )


def load_prepared_offline_replay(
    *,
    replay_buffer: Any,
    prepared_paths: Sequence[Path],
    logger: logging.Logger,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
) -> dict[str, Any]:
    return _load_prepared_offline_replay(
        replay_buffer=replay_buffer,
        prepared_paths=prepared_paths,
        logger=logger,
        max_episodes=max_episodes,
        max_transitions=max_transitions,
        manifest_filename=MANIFEST_FILENAME,
        episode_file_glob=EPISODE_FILE_GLOB,
        read_manifest_fn=read_manifest,
    )


__all__ = [
    "LiberoTaskSpec",
    "OfflinePreparedResolution",
    "OFFLINE_FORMAT_VERSION",
    "load_prepared_offline_replay",
    "prepare_demo_transitions",
    "prepare_fingerprint",
    "prepared_dir_for_cfg",
    "resolve_and_validate_prepared_paths",
    "resolve_configured_prepared_paths",
    "resolve_prepared_episode_files",
    "resolve_task_spec",
    "write_manifest",
]
