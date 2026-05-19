from __future__ import annotations

"""Shared prepared-offline dataset helpers."""

import dataclasses
import json
import logging
import pickle
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Sequence

from tqdm.auto import tqdm

from serl_launcher.utils.path_utils import resolve_original_cwd
from serl_launcher.utils.path_utils import resolve_path
from serl_launcher.utils.serialization import to_jsonable

MANIFEST_FILENAME = "manifest.json"
EPISODE_FILE_GLOB = "episode_*.pkl"
ActiveStepRange = tuple[int, int | None]
RESIDUAL_PREPARED_SIGNATURE_KEYS = (
    "task_key",
    "policy_backend_type",
    "policy_backend_id",
    "chunk_horizon",
    "action_dim",
    "alpha",
    "action_mask",
    "action_limits",
    "clip_gripper",
    "expert_reference_scale",
    "clip_residual_to_unit",
    "filter_unrepresentable_steps",
    "image_keys",
    "vector_obs_keys",
)


@dataclasses.dataclass(frozen=True, slots=True)
class OfflinePreparedInputs:
    prepared_paths: tuple[Path, ...]
    prepare_stats: dict[str, Any] | None
    manifest_paths: tuple[Path, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class OfflinePreparedResolution:
    prepared_paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    validation_stats: dict[str, Any]


def format_residual_alpha_token(alpha: float) -> str:
    return f"{float(alpha):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_residual_prepared_fingerprint(
    *,
    format_version: str,
    task_key: str,
    task_description: str,
    policy_backend_type: str,
    policy_backend_id: str,
    chunk_horizon: int,
    action_dim: int,
    alpha: float,
    action_mask: Sequence[bool] | None,
    action_limits: Sequence[float],
    clip_gripper: bool,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    filter_unrepresentable_steps: bool,
    image_keys: Sequence[str],
    vector_obs_keys: Sequence[str] | None,
    raw_dataset_path: str | Path,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    signature = build_residual_training_signature(
        task_key=task_key,
        policy_backend_type=policy_backend_type,
        policy_backend_id=policy_backend_id,
        chunk_horizon=chunk_horizon,
        action_dim=action_dim,
        alpha=alpha,
        action_mask=action_mask,
        action_limits=action_limits,
        clip_gripper=clip_gripper,
        expert_reference_scale=expert_reference_scale,
        clip_residual_to_unit=clip_residual_to_unit,
        filter_unrepresentable_steps=filter_unrepresentable_steps,
        image_keys=image_keys,
        vector_obs_keys=vector_obs_keys,
    )
    fingerprint = {
        "format_version": str(format_version),
        "task_key": signature["task_key"],
        "task_description": str(task_description),
    }
    fingerprint.update(
        {key: value for key, value in signature.items() if key != "task_key"}
    )
    fingerprint["raw_dataset_path"] = str(raw_dataset_path)
    if extra_fields is not None:
        fingerprint.update(dict(extra_fields))
    return fingerprint


def build_residual_training_signature(
    *,
    task_key: str,
    policy_backend_type: str,
    policy_backend_id: str,
    chunk_horizon: int,
    action_dim: int,
    alpha: float,
    action_mask: Sequence[bool] | None,
    action_limits: Sequence[float],
    clip_gripper: bool,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    filter_unrepresentable_steps: bool,
    image_keys: Sequence[str],
    vector_obs_keys: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "task_key": str(task_key),
        "policy_backend_type": str(policy_backend_type),
        "policy_backend_id": str(policy_backend_id),
        "chunk_horizon": int(chunk_horizon),
        "action_dim": int(action_dim),
        "alpha": float(alpha),
        "action_mask": (
            None if action_mask is None else [bool(value) for value in action_mask]
        ),
        "action_limits": [float(value) for value in action_limits],
        "clip_gripper": bool(clip_gripper),
        "expert_reference_scale": float(expert_reference_scale),
        "clip_residual_to_unit": bool(clip_residual_to_unit),
        "filter_unrepresentable_steps": bool(filter_unrepresentable_steps),
        "image_keys": [str(value) for value in image_keys],
        "vector_obs_keys": (
            None
            if vector_obs_keys is None
            else [str(value) for value in vector_obs_keys]
        ),
    }


def extract_residual_manifest_signature(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    fingerprint = manifest.get("fingerprint", None)
    if not isinstance(fingerprint, dict):
        return None
    signature = {
        key: fingerprint.get(key, None) for key in RESIDUAL_PREPARED_SIGNATURE_KEYS
    }
    signature["filter_unrepresentable_steps"] = bool(
        fingerprint.get("filter_unrepresentable_steps", False)
    )
    return signature


def resolve_residual_prepared_dir(
    *,
    output_root: str | Path,
    task_key: str,
    policy_backend: str,
    chunk_horizon: int,
    alpha: float,
) -> Path:
    output_root_path = resolve_path(str(output_root), base=resolve_original_cwd())
    resolved_task_key = str(task_key)
    task_root = (
        output_root_path
        if output_root_path.name == resolved_task_key
        else (output_root_path / resolved_task_key)
    )
    backend = str(policy_backend).replace(":", "_")
    alpha_token = format_residual_alpha_token(alpha)
    return (
        task_root / f"{backend}_chunk{int(chunk_horizon)}_alpha{alpha_token}"
    ).resolve()


def resolve_prepared_path_value(prepared_path: str | None) -> tuple[Path, ...]:
    if prepared_path is None:
        return tuple()
    return (resolve_path(prepared_path, base=resolve_original_cwd()),)


def read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def validate_prepared_paths(
    prepared_paths: Sequence[Path],
    *,
    expected_signature: dict[str, Any],
    manifest_signature_fn: Callable[[dict[str, Any] | None], dict[str, Any] | None],
    manifest_filename: str = MANIFEST_FILENAME,
) -> OfflinePreparedResolution:
    if not prepared_paths:
        raise ValueError(
            "offline.enabled=true requires offline.prepared_path to point to prepared data"
        )

    manifest_paths: list[Path] = []
    stats: dict[str, Any] = {
        "paths_total": int(len(prepared_paths)),
        "manifests_checked": 0,
        "manifests_matched": 0,
        "paths_without_manifest": 0,
        "paths_without_validation": 0,
    }

    for path in prepared_paths:
        manifest_path: Path | None = None
        if path.is_dir():
            candidate = path / manifest_filename
            if candidate.exists():
                manifest_path = candidate.resolve()
            else:
                raise ValueError(
                    f"prepared offline directory must contain {manifest_filename} for "
                    f"compatibility validation: {path}"
                )
        elif path.name == manifest_filename:
            manifest_path = path.resolve()
        elif path.suffix == ".pkl":
            raise ValueError(
                "prepared offline episode file without manifest is no longer "
                f"supported: {path}"
            )
        else:
            raise ValueError(f"Unsupported offline prepared path: {path}")

        manifest_paths.append(manifest_path)
        manifest = read_manifest(manifest_path)
        if manifest is None:
            raise ValueError(f"Invalid offline manifest: {manifest_path}")
        manifest_signature = manifest_signature_fn(manifest)
        if manifest_signature is None:
            raise ValueError(
                "offline manifest missing compatibility fingerprint fields: "
                f"{manifest_path}"
            )

        mismatches: dict[str, dict[str, Any]] = {}
        for key, expected_value in expected_signature.items():
            actual_value = manifest_signature.get(key, None)
            if to_jsonable(actual_value) != to_jsonable(expected_value):
                mismatches[str(key)] = {
                    "expected": to_jsonable(expected_value),
                    "actual": to_jsonable(actual_value),
                }
        stats["manifests_checked"] = int(stats["manifests_checked"]) + 1
        if mismatches:
            raise ValueError(
                "prepared offline manifest does not match current training config: "
                f"{manifest_path} mismatches={json.dumps(mismatches, ensure_ascii=False)}"
            )
        stats["manifests_matched"] = int(stats["manifests_matched"]) + 1

    return OfflinePreparedResolution(
        prepared_paths=tuple(Path(path).resolve() for path in prepared_paths),
        manifest_paths=tuple(manifest_paths),
        validation_stats=stats,
    )


def _episode_files_from_manifest(
    manifest_path: Path,
    *,
    read_manifest_fn: Callable[[Path], dict[str, Any] | None] = read_manifest,
) -> list[Path]:
    manifest = read_manifest_fn(manifest_path)
    if manifest is None:
        raise ValueError(f"Invalid offline manifest: {manifest_path}")
    episode_files = manifest.get("episode_files", ())
    resolved: list[Path] = []
    for entry in episode_files:
        resolved.append((manifest_path.parent / str(entry)).resolve())
    return resolved


def resolve_prepared_episode_files(
    paths: Sequence[Path],
    *,
    manifest_filename: str = MANIFEST_FILENAME,
    episode_file_glob: str = EPISODE_FILE_GLOB,
    read_manifest_fn: Callable[[Path], dict[str, Any] | None] = read_manifest,
) -> list[Path]:
    episode_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            manifest_path = path / manifest_filename
            if manifest_path.exists():
                episode_files.extend(
                    _episode_files_from_manifest(
                        manifest_path,
                        read_manifest_fn=read_manifest_fn,
                    )
                )
                continue
            episode_files.extend(sorted(path.glob(episode_file_glob)))
            continue
        if path.name == manifest_filename:
            episode_files.extend(
                _episode_files_from_manifest(
                    path,
                    read_manifest_fn=read_manifest_fn,
                )
            )
            continue
        if path.suffix == ".pkl":
            episode_files.append(path.resolve())
            continue
        raise ValueError(f"Unsupported offline prepared path: {path}")

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in episode_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(resolved)
    return unique_files


def _episode_step_in_active_ranges(
    episode_step: int,
    active_step_ranges: Sequence[ActiveStepRange] | None,
) -> bool:
    if active_step_ranges is None:
        return True
    step = int(episode_step)
    for start_step, end_step in active_step_ranges:
        if step < int(start_step):
            continue
        if end_step is not None and step >= int(end_step):
            continue
        return True
    return False


def _terminalize_if_next_step_inactive(
    transition: dict[str, Any],
    *,
    episode_step: int,
    active_step_ranges: Sequence[ActiveStepRange] | None,
) -> tuple[dict[str, Any], bool]:
    if active_step_ranges is None:
        return transition, False
    if _episode_step_in_active_ranges(
        int(episode_step) + 1,
        active_step_ranges,
    ):
        return transition, False

    existing_mask = transition.get("masks", 1.0)
    try:
        already_terminal = float(existing_mask) == 0.0
    except Exception:  # noqa: BLE001
        already_terminal = False
    patched_transition = dict(transition)
    patched_transition["masks"] = 0.0
    return patched_transition, not bool(already_terminal)


def load_prepared_offline_replay(
    *,
    replay_buffer: Any,
    prepared_paths: Sequence[Path],
    logger: logging.Logger,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
    min_episode_step: int | None = None,
    active_step_ranges: Sequence[ActiveStepRange] | None = None,
    manifest_filename: str = MANIFEST_FILENAME,
    episode_file_glob: str = EPISODE_FILE_GLOB,
    read_manifest_fn: Callable[[Path], dict[str, Any] | None] = read_manifest,
) -> dict[str, Any]:
    episode_files = resolve_prepared_episode_files(
        prepared_paths,
        manifest_filename=manifest_filename,
        episode_file_glob=episode_file_glob,
        read_manifest_fn=read_manifest_fn,
    )
    stats: dict[str, Any] = {
        "files_total": int(len(episode_files)),
        "episodes_loaded": 0,
        "steps_loaded": 0,
        "steps_skipped_min_episode_step": 0,
        "steps_skipped_active_step_ranges": 0,
        "steps_terminalized_active_step_ranges": 0,
        "load_errors": 0,
    }
    if not episode_files:
        logger.warning("Prepared offline dataset paths resolved to zero episode files")
        return stats

    episode_iter: Iterable[Path] = tqdm(
        episode_files,
        total=len(episode_files),
        desc="load offline replay",
        unit="episode",
        dynamic_ncols=True,
        leave=False,
    )
    for episode_path in episode_iter:
        if max_episodes is not None and stats["episodes_loaded"] >= int(max_episodes):
            break
        if max_transitions is not None and stats["steps_loaded"] >= int(max_transitions):
            break
        try:
            with open(episode_path, "rb") as fp:
                transitions = pickle.load(fp)
            if not isinstance(transitions, list):
                raise ValueError(f"prepared episode must be a list, got {type(transitions)}")
            for transition in transitions:
                if max_transitions is not None and stats["steps_loaded"] >= int(max_transitions):
                    break
                if not isinstance(transition, dict):
                    raise ValueError(
                        f"prepared transition must be a dict, got {type(transition)}"
                    )
                if min_episode_step is not None:
                    episode_step_raw = transition.get("episode_step", None)
                    if episode_step_raw is None:
                        stats["steps_skipped_min_episode_step"] += 1
                        continue
                    if int(episode_step_raw) < int(min_episode_step):
                        stats["steps_skipped_min_episode_step"] += 1
                        continue
                if active_step_ranges is not None:
                    episode_step_raw = transition.get("episode_step", None)
                    if episode_step_raw is None:
                        stats["steps_skipped_active_step_ranges"] += 1
                        continue
                    if not _episode_step_in_active_ranges(
                        int(episode_step_raw),
                        active_step_ranges,
                    ):
                        stats["steps_skipped_active_step_ranges"] += 1
                        continue
                    transition, terminalized = _terminalize_if_next_step_inactive(
                        transition,
                        episode_step=int(episode_step_raw),
                        active_step_ranges=active_step_ranges,
                    )
                    if terminalized:
                        stats["steps_terminalized_active_step_ranges"] += 1
                replay_buffer.insert(transition)
                stats["steps_loaded"] += 1
            stats["episodes_loaded"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["load_errors"] += 1
            logger.warning("failed to load prepared offline episode %s: %s", episode_path, exc)

    manifest_paths: list[Path] = []
    seen_manifest_paths: set[Path] = set()
    for prepared_path in prepared_paths:
        manifest_path: Path | None = None
        if prepared_path.is_dir():
            candidate = prepared_path / manifest_filename
            if candidate.exists():
                manifest_path = candidate.resolve()
        elif prepared_path.name == manifest_filename:
            manifest_path = prepared_path.resolve()
        if manifest_path is None or manifest_path in seen_manifest_paths:
            continue
        seen_manifest_paths.add(manifest_path)
        manifest_paths.append(manifest_path)

    dataset_stats = {
        "steps_total": 0,
        "steps_unrepresentable": 0,
        "steps_filtered": 0,
        "steps_written": 0,
    }
    for manifest_path in manifest_paths:
        manifest = read_manifest_fn(manifest_path)
        if manifest is None:
            continue
        prepare_stats = manifest.get("prepare_stats", None)
        if not isinstance(prepare_stats, dict):
            continue
        dataset_stats["steps_total"] += int(prepare_stats.get("steps_total", 0) or 0)
        dataset_stats["steps_unrepresentable"] += int(
            prepare_stats.get("steps_unrepresentable", 0) or 0
        )
        dataset_stats["steps_filtered"] += int(
            prepare_stats.get("steps_filtered", 0) or 0
        )
        dataset_stats["steps_written"] += int(
            prepare_stats.get(
                "steps_written",
                prepare_stats.get("transitions_written", 0),
            )
            or 0
        )

    logger.info(
        "Offline replay loaded: files_total=%s episodes_loaded=%s steps_loaded=%s "
        "steps_skipped_min_episode_step=%s steps_skipped_active_step_ranges=%s "
        "steps_terminalized_active_step_ranges=%s load_errors=%s "
        "dataset_steps_total=%s dataset_steps_filtered=%s dataset_steps_written=%s",
        int(stats["files_total"]),
        int(stats["episodes_loaded"]),
        int(stats["steps_loaded"]),
        int(stats["steps_skipped_min_episode_step"]),
        int(stats["steps_skipped_active_step_ranges"]),
        int(stats["steps_terminalized_active_step_ranges"]),
        int(stats["load_errors"]),
        int(dataset_stats["steps_total"]),
        int(dataset_stats["steps_filtered"]),
        int(dataset_stats["steps_written"]),
    )
    return stats


__all__ = [
    "EPISODE_FILE_GLOB",
    "MANIFEST_FILENAME",
    "OfflinePreparedInputs",
    "OfflinePreparedResolution",
    "RESIDUAL_PREPARED_SIGNATURE_KEYS",
    "build_residual_prepared_fingerprint",
    "build_residual_training_signature",
    "extract_residual_manifest_signature",
    "format_residual_alpha_token",
    "load_prepared_offline_replay",
    "read_manifest",
    "resolve_prepared_episode_files",
    "resolve_prepared_path_value",
    "resolve_residual_prepared_dir",
    "validate_prepared_paths",
]
