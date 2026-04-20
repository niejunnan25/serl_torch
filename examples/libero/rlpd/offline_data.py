from __future__ import annotations

"""LIBERO-specific offline data preparation and loading helpers for direct RLPD."""

import dataclasses
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from serl_launcher.utils.serialization import to_jsonable

from .config import LiberoRLPDTrainConfig
from ..offline_data import EPISODE_FILE_GLOB
from ..offline_data import MANIFEST_FILENAME
from ..offline_data import LiberoTaskSpec
from ..offline_data import _build_frame_obs
from ..offline_data import _load_demo_payload
from ..offline_data import _pad_or_truncate_1d
from ..offline_data import _read_manifest
from ..offline_data import _resolve_original_cwd
from ..offline_data import _resolve_path
from ..offline_data import _resolve_task_spec
from ..offline_data import resolve_prepared_episode_files
from .observation import build_rlpd_obs

OFFLINE_FORMAT_VERSION = "libero_rlpd_step_transitions_v1"


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


def _prepare_fingerprint(
    cfg: LiberoRLPDTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
) -> dict[str, Any]:
    return {
        "format_version": OFFLINE_FORMAT_VERSION,
        "task_key": task_spec.task_key,
        "task_description": task_spec.task_description,
        "observation_type": "direct",
        "action_source": "expert_raw",
        "action_dim": int(cfg.env.action_dim),
        "image_keys": [str(v) for v in cfg.obs.image_keys],
        "vector_obs_keys": (
            None
            if cfg.obs.vector_obs_keys is None
            else [str(v) for v in cfg.obs.vector_obs_keys]
        ),
        "raw_dataset_path": str(task_spec.dataset_path),
    }


def _training_compatibility_signature(cfg: LiberoRLPDTrainConfig) -> dict[str, Any]:
    return {
        "task_key": f"{cfg.task.suite_name}_task_{cfg.task.task_id}",
        "observation_type": "direct",
        "action_source": "expert_raw",
        "action_dim": int(cfg.env.action_dim),
        "image_keys": [str(v) for v in cfg.obs.image_keys],
        "vector_obs_keys": (
            None
            if cfg.obs.vector_obs_keys is None
            else [str(v) for v in cfg.obs.vector_obs_keys]
        ),
    }


def _prepared_dir_for_cfg(
    cfg: LiberoRLPDTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
) -> Path:
    output_root = _resolve_path(
        cfg.offline.prepare.output_root,
        base=_resolve_original_cwd(),
    )
    task_root = (
        output_root
        if output_root.name == str(task_spec.task_key)
        else (output_root / task_spec.task_key)
    )
    image_token = "-".join(str(key) for key in cfg.obs.image_keys)
    proprio_token = (
        "proprio" if cfg.obs.vector_obs_keys is not None else "vision_only"
    )
    return (task_root / f"direct_{image_token}_{proprio_token}").resolve()


def _manifest_signature(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    manifest_fingerprint = manifest.get("fingerprint", None)
    if not isinstance(manifest_fingerprint, dict):
        return None
    return {
        "task_key": manifest_fingerprint.get("task_key", None),
        "observation_type": manifest_fingerprint.get("observation_type", None),
        "action_source": manifest_fingerprint.get("action_source", None),
        "action_dim": manifest_fingerprint.get("action_dim", None),
        "image_keys": manifest_fingerprint.get("image_keys", None),
        "vector_obs_keys": manifest_fingerprint.get("vector_obs_keys", None),
    }


def resolve_configured_prepared_paths(
    cfg: LiberoRLPDTrainConfig,
) -> tuple[Path, ...]:
    prepared_path = cfg.offline.prepared_path
    if prepared_path is None:
        return tuple()
    return (_resolve_path(prepared_path, base=_resolve_original_cwd()),)


def resolve_and_validate_prepared_paths(
    cfg: LiberoRLPDTrainConfig,
    *,
    logger: logging.Logger,
) -> OfflinePreparedResolution:
    del logger
    prepared_paths = resolve_configured_prepared_paths(cfg)
    if not prepared_paths:
        raise ValueError(
            "offline.enabled=true requires offline.prepared_path to point to prepared data"
        )

    expected_signature = _training_compatibility_signature(cfg)
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
            candidate = path / MANIFEST_FILENAME
            if candidate.exists():
                manifest_path = candidate.resolve()
            else:
                raise ValueError(
                    "prepared offline directory must contain manifest.json for "
                    f"compatibility validation: {path}"
                )
        elif path.name == MANIFEST_FILENAME:
            manifest_path = path.resolve()
        elif path.suffix == ".pkl":
            raise ValueError(
                "prepared offline episode file without manifest is no longer "
                f"supported: {path}"
            )
        else:
            raise ValueError(f"Unsupported offline prepared path: {path}")

        manifest_paths.append(manifest_path)
        manifest = _read_manifest(manifest_path)
        if manifest is None:
            raise ValueError(f"Invalid offline manifest: {manifest_path}")
        manifest_signature = _manifest_signature(manifest)
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
        prepared_paths=prepared_paths,
        manifest_paths=tuple(manifest_paths),
        validation_stats=stats,
    )


def _zero_like_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs.items()}


def _prepare_demo_transitions(
    *,
    demo: Any,
    episode_id: int,
    image_keys: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_demo_payload(demo)
    expert_actions = payload["actions"]
    num_steps = int(expert_actions.shape[0])
    rewards_present = "rewards" in demo
    rewards = _pad_or_truncate_1d(
        demo.get("rewards", np.zeros((num_steps,), dtype=np.float32)),
        length=num_steps,
        dtype=np.float32,
        fill_value=0.0,
    )
    if (not rewards_present) and num_steps > 0:
        rewards[-1] = 1.0
    dones = _pad_or_truncate_1d(
        demo.get("dones", np.zeros((num_steps,), dtype=bool)),
        length=num_steps,
        dtype=bool,
        fill_value=False,
    )
    if num_steps > 0:
        dones[-1] = True

    transitions: list[dict[str, Any]] = []
    episode_return = 0.0
    for step_idx in range(num_steps):
        obs_raw = _build_frame_obs(payload, step_idx)
        rlpd_obs = build_rlpd_obs(
            obs=obs_raw,
            image_keys=image_keys,
        )
        reward = float(rewards[step_idx])
        episode_return += reward
        done = bool(dones[step_idx]) or bool(step_idx >= (num_steps - 1))

        if done:
            next_rlpd_obs = _zero_like_obs(rlpd_obs)
            mask = 0.0
        else:
            next_obs_raw = _build_frame_obs(payload, step_idx + 1)
            next_rlpd_obs = build_rlpd_obs(
                obs=next_obs_raw,
                image_keys=image_keys,
            )
            mask = 1.0

        transitions.append(
            {
                "episode_id": int(episode_id),
                "episode_step": int(step_idx),
                "observations": rlpd_obs,
                "actions": np.asarray(expert_actions[step_idx], dtype=np.float32).reshape(
                    -1
                ),
                "next_observations": next_rlpd_obs,
                "rewards": float(reward),
                "masks": float(mask),
                "dones": bool(done),
            }
        )

    episode_stats = {
        "episode_id": int(episode_id),
        "steps_total": int(num_steps),
        "steps_written": int(len(transitions)),
        "steps_unrepresentable": 0,
        "steps_filtered": 0,
        "episode_return": float(episode_return),
        "success": bool(num_steps > 0),
    }
    return transitions, episode_stats


def _write_manifest(
    *,
    manifest_path: Path,
    task_spec: LiberoTaskSpec,
    cfg: LiberoRLPDTrainConfig,
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
        "rlpd": {
            "observation_type": "direct",
            "action_source": "expert_raw",
            "action_dim": int(cfg.env.action_dim),
        },
        "fingerprint": to_jsonable(fingerprint),
        "prepare_stats": to_jsonable(prepare_stats),
        "episode_files": [str(path.name) for path in episode_files],
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    return manifest


def prepare_current_task_offline_data(
    cfg: LiberoRLPDTrainConfig,
    *,
    logger: logging.Logger,
) -> OfflinePreparedInputs:
    if not cfg.offline.enabled:
        return OfflinePreparedInputs(
            prepared_paths=tuple(),
            prepare_stats=None,
            manifest_paths=tuple(),
        )

    prepared_paths = resolve_configured_prepared_paths(cfg)
    if prepared_paths:
        return OfflinePreparedInputs(
            prepared_paths=prepared_paths,
            prepare_stats=None,
            manifest_paths=tuple(
                path / MANIFEST_FILENAME if path.is_dir() else path
                for path in prepared_paths
                if path.name == MANIFEST_FILENAME or path.is_dir()
            ),
        )

    task_spec = _resolve_task_spec(cfg)
    prepared_dir = _prepared_dir_for_cfg(cfg, task_spec=task_spec)
    manifest_path = prepared_dir / MANIFEST_FILENAME
    fingerprint = _prepare_fingerprint(cfg, task_spec=task_spec)

    if not task_spec.dataset_path.exists():
        raise FileNotFoundError(f"Raw LIBERO demo dataset not found: {task_spec.dataset_path}")

    import h5py

    prepared_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in prepared_dir.glob(EPISODE_FILE_GLOB):
        stale_path.unlink()

    image_keys = cfg.obs.image_keys
    prepare_stats: dict[str, Any] = {
        "raw_dataset_path": str(task_spec.dataset_path),
        "prepared_dir": str(prepared_dir),
        "episodes_total": 0,
        "steps_total": 0,
        "steps_unrepresentable": 0,
        "steps_filtered": 0,
        "episodes_written": 0,
        "steps_written": 0,
        "elapsed_sec": 0.0,
    }
    start_time = time.time()
    episode_files: list[Path] = []

    logger.info(
        "Preparing direct offline dataset: task=%s raw=%s output=%s",
        task_spec.task_key,
        task_spec.dataset_path,
        prepared_dir,
    )

    with h5py.File(task_spec.dataset_path, "r") as dataset_file:
        demo_names = sorted(
            list(dataset_file["data"].keys()),
            key=lambda name: int(str(name).split("_")[-1]),
        )

        episode_iter: Iterable[tuple[int, str]] = tqdm(
            enumerate(demo_names),
            total=len(demo_names),
            desc=f"prepare {task_spec.task_key}",
            unit="episode",
            dynamic_ncols=True,
            leave=True,
        )
        for episode_index, demo_name in episode_iter:
            transitions, episode_stats = _prepare_demo_transitions(
                demo=dataset_file["data"][demo_name],
                episode_id=int(episode_index),
                image_keys=image_keys,
            )
            prepare_stats["episodes_total"] = int(prepare_stats["episodes_total"]) + 1
            prepare_stats["steps_total"] = int(prepare_stats["steps_total"]) + int(
                episode_stats["steps_total"]
            )
            prepare_stats["steps_written"] = int(prepare_stats["steps_written"]) + int(
                episode_stats["steps_written"]
            )
            if transitions:
                episode_path = prepared_dir / f"episode_{int(episode_index):06d}.pkl"
                with open(episode_path, "wb") as fp:
                    pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)
                episode_files.append(episode_path)
                prepare_stats["episodes_written"] = int(
                    prepare_stats["episodes_written"]
                ) + 1

    prepare_stats["elapsed_sec"] = float(time.time() - start_time)
    _write_manifest(
        manifest_path=manifest_path,
        task_spec=task_spec,
        cfg=cfg,
        fingerprint=fingerprint,
        episode_files=episode_files,
        prepare_stats=prepare_stats,
    )
    logger.info(
        "Direct offline prepare complete: episodes_total=%s steps_total=%s "
        "episodes_written=%s steps_written=%s manifest=%s",
        int(prepare_stats["episodes_total"]),
        int(prepare_stats["steps_total"]),
        int(prepare_stats["episodes_written"]),
        int(prepare_stats["steps_written"]),
        manifest_path,
    )
    return OfflinePreparedInputs(
        prepared_paths=(prepared_dir,),
        prepare_stats=prepare_stats,
        manifest_paths=(manifest_path,),
    )


def load_prepared_offline_replay(
    *,
    replay_buffer: Any,
    prepared_paths: Sequence[Path],
    logger: logging.Logger,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
) -> dict[str, Any]:
    episode_files = resolve_prepared_episode_files(prepared_paths)
    stats: dict[str, Any] = {
        "files_total": int(len(episode_files)),
        "episodes_loaded": 0,
        "steps_loaded": 0,
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
            candidate = prepared_path / MANIFEST_FILENAME
            if candidate.exists():
                manifest_path = candidate.resolve()
        elif prepared_path.name == MANIFEST_FILENAME:
            manifest_path = prepared_path.resolve()
        if manifest_path is None or manifest_path in seen_manifest_paths:
            continue
        seen_manifest_paths.add(manifest_path)
        manifest_paths.append(manifest_path)

    dataset_stats = {
        "steps_total": 0,
        "steps_written": 0,
    }
    for manifest_path in manifest_paths:
        manifest = _read_manifest(manifest_path)
        if manifest is None:
            continue
        prepare_stats = manifest.get("prepare_stats", None)
        if not isinstance(prepare_stats, dict):
            continue
        dataset_stats["steps_total"] += int(prepare_stats.get("steps_total", 0) or 0)
        dataset_stats["steps_written"] += int(
            prepare_stats.get("steps_written", 0) or 0
        )

    logger.info(
        "Offline replay loaded: files_total=%s episodes_loaded=%s steps_loaded=%s "
        "load_errors=%s dataset_steps_total=%s dataset_steps_written=%s",
        int(stats["files_total"]),
        int(stats["episodes_loaded"]),
        int(stats["steps_loaded"]),
        int(stats["load_errors"]),
        int(dataset_stats["steps_total"]),
        int(dataset_stats["steps_written"]),
    )
    return stats


__all__ = [
    "OFFLINE_FORMAT_VERSION",
    "OfflinePreparedInputs",
    "OfflinePreparedResolution",
    "load_prepared_offline_replay",
    "prepare_current_task_offline_data",
    "resolve_and_validate_prepared_paths",
    "resolve_configured_prepared_paths",
]
