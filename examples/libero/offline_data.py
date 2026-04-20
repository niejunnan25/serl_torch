from __future__ import annotations

"""LIBERO-specific offline data preparation and loading helpers."""

import dataclasses
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np
from hydra.utils import get_original_cwd
from tqdm.auto import tqdm

from serl_launcher.policy.typed_factory import build_policy_client
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_id
from serl_launcher.policy.typed_factory import resolve_policy_backend_type
from serl_launcher.utils.serialization import to_jsonable

from .config import LiberoTrainConfig
from .env.policy_input import build_libero_policy_input
from .env.setup import resolve_libero_config_dir
from .env.setup import resolve_libero_datasets_root
from .env.setup import resolve_libero_root
from .env.setup import setup_libero_pythonpath
from .residual_observation import build_chunk_residual_obs
from .residual_observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec

MANIFEST_FILENAME = "manifest.json"
EPISODE_FILE_GLOB = "episode_*.pkl"
OFFLINE_FORMAT_VERSION = "libero_offline_step_transitions_v1"
UNIT_RESIDUAL_EPS = 1.0e-6


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


def _resolve_original_cwd() -> Path:
    try:
        return Path(get_original_cwd()).resolve()
    except Exception:  # noqa: BLE001
        return Path.cwd().resolve()


def _resolve_path(raw_path: str, *, base: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _candidate_dataset_paths(
    datasets_root: Path,
    suite_name: str,
    task_name: str,
) -> Iterator[Path]:
    filename = f"{task_name}_demo.hdf5"
    yield (datasets_root / suite_name / filename).resolve()
    yield (datasets_root / filename).resolve()


def _resolve_task_spec(cfg: LiberoTrainConfig) -> LiberoTaskSpec:
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
        dataset_path = _resolve_path(dataset_override, base=_resolve_original_cwd())
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


def _format_alpha(alpha: float) -> str:
    return f"{float(alpha):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _prepare_fingerprint(
    cfg: LiberoTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
) -> dict[str, Any]:
    return {
        "format_version": OFFLINE_FORMAT_VERSION,
        "task_key": task_spec.task_key,
        "task_description": task_spec.task_description,
        "policy_backend_type": resolve_policy_backend_type(cfg),
        "policy_backend_id": resolve_policy_backend_id(cfg),
        "chunk_horizon": int(cfg.residual.chunk_horizon),
        "action_dim": int(cfg.env.action_dim),
        "alpha": float(cfg.residual.alpha),
        "action_mask": (
            None
            if cfg.residual.action_mask is None
            else [bool(v) for v in cfg.residual.action_mask]
        ),
        "action_limits": [float(v) for v in cfg.residual.action_limits],
        "clip_gripper": bool(cfg.residual.clip_gripper),
        "expert_reference_scale": float(cfg.offline.prepare.expert_reference_scale),
        "clip_residual_to_unit": bool(cfg.offline.prepare.clip_residual_to_unit),
        "filter_unrepresentable_steps": bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        "image_keys": [str(v) for v in cfg.obs.image_keys],
        "vector_obs_keys": (
            None
            if cfg.obs.vector_obs_keys is None
            else [str(v) for v in cfg.obs.vector_obs_keys]
        ),
        "raw_dataset_path": str(task_spec.dataset_path),
    }


def _training_compatibility_signature(cfg: LiberoTrainConfig) -> dict[str, Any]:
    return {
        "task_key": f"{cfg.task.suite_name}_task_{cfg.task.task_id}",
        "policy_backend_type": resolve_policy_backend_type(cfg),
        "policy_backend_id": resolve_policy_backend_id(cfg),
        "chunk_horizon": int(cfg.residual.chunk_horizon),
        "action_dim": int(cfg.env.action_dim),
        "alpha": float(cfg.residual.alpha),
        "action_mask": (
            None
            if cfg.residual.action_mask is None
            else [bool(v) for v in cfg.residual.action_mask]
        ),
        "action_limits": [float(v) for v in cfg.residual.action_limits],
        "clip_gripper": bool(cfg.residual.clip_gripper),
        "expert_reference_scale": float(cfg.offline.prepare.expert_reference_scale),
        "clip_residual_to_unit": bool(cfg.offline.prepare.clip_residual_to_unit),
        "filter_unrepresentable_steps": bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        "image_keys": [str(v) for v in cfg.obs.image_keys],
        "vector_obs_keys": (
            None
            if cfg.obs.vector_obs_keys is None
            else [str(v) for v in cfg.obs.vector_obs_keys]
        ),
    }


def _prepared_dir_for_cfg(
    cfg: LiberoTrainConfig,
    *,
    task_spec: LiberoTaskSpec,
) -> Path:
    output_root = _resolve_path(
        cfg.offline.prepare.output_root,
        base=_resolve_original_cwd(),
    )
    backend = describe_policy_backend(cfg).replace(":", "_")
    alpha_token = _format_alpha(cfg.residual.alpha)
    task_root = (
        output_root
        if output_root.name == str(task_spec.task_key)
        else (output_root / task_spec.task_key)
    )
    return (
        task_root / f"{backend}_chunk{int(cfg.residual.chunk_horizon)}_alpha{alpha_token}"
    ).resolve()


def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


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
        "policy_backend_type": manifest_fingerprint.get("policy_backend_type", None),
        "policy_backend_id": manifest_fingerprint.get("policy_backend_id", None),
        "chunk_horizon": manifest_fingerprint.get("chunk_horizon", None),
        "action_dim": manifest_fingerprint.get("action_dim", None),
        "alpha": manifest_fingerprint.get("alpha", None),
        "action_mask": manifest_fingerprint.get("action_mask", None),
        "action_limits": manifest_fingerprint.get("action_limits", None),
        "clip_gripper": manifest_fingerprint.get("clip_gripper", None),
        "expert_reference_scale": manifest_fingerprint.get("expert_reference_scale", None),
        "clip_residual_to_unit": manifest_fingerprint.get("clip_residual_to_unit", None),
        "filter_unrepresentable_steps": bool(
            manifest_fingerprint.get("filter_unrepresentable_steps", False)
        ),
        "image_keys": manifest_fingerprint.get("image_keys", None),
        "vector_obs_keys": manifest_fingerprint.get("vector_obs_keys", None),
    }


def resolve_configured_prepared_paths(cfg: LiberoTrainConfig) -> tuple[Path, ...]:
    prepared_path = cfg.offline.prepared_path
    if prepared_path is None:
        return tuple()
    return (_resolve_path(prepared_path, base=_resolve_original_cwd()),)


def resolve_and_validate_prepared_paths(
    cfg: LiberoTrainConfig,
    *,
    logger: logging.Logger,
) -> OfflinePreparedResolution:
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


def _build_frame_obs(payload: dict[str, np.ndarray], frame_idx: int) -> dict[str, Any]:
    return {
        "agentview_rgb": payload["agentview_rgb"][frame_idx],
        "eye_in_hand_rgb": payload["eye_in_hand_rgb"][frame_idx],
        "ee_pos": payload["ee_pos"][frame_idx],
        "ee_ori": payload["ee_ori"][frame_idx],
        "gripper_states": payload["gripper_states"][frame_idx],
    }


def _pad_or_truncate_1d(
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
        obs_raw = _build_frame_obs(payload, step_idx)
        policy_input = build_libero_policy_input(obs_raw, task_prompt)
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


def _project_expert_action(
    *,
    expert_action: np.ndarray,
    base_action: np.ndarray,
    action_spec: ResidualActionSpec,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
) -> tuple[np.ndarray, int, bool]:
    expert_arr = np.asarray(expert_action, dtype=np.float32).reshape(-1)
    base_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if expert_arr.shape != base_arr.shape:
        raise ValueError(
            f"expert/base action shape mismatch: {expert_arr.shape} != {base_arr.shape}"
        )

    projected = np.asarray(base_arr, dtype=np.float32).copy()
    if action_spec.alpha <= 0.0:
        step_unrepresentable = bool(
            np.any(
                np.abs(
                    expert_arr[
                        np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
                    ]
                    - base_arr[
                        np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
                    ]
                )
                > UNIT_RESIDUAL_EPS
            )
        )
        if action_spec.clip_gripper and projected.shape[0] > 0:
            projected[-1] = np.clip(projected[-1], -1.0, 1.0)
        return projected, 0, step_unrepresentable

    clipped_values = 0
    step_unrepresentable = False
    limits = np.asarray(action_spec.residual_limits, dtype=np.float32).reshape(-1)
    control_indices = np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
    denom = limits * float(action_spec.alpha) * float(expert_reference_scale)
    for local_idx, action_idx in enumerate(control_indices):
        scale = float(denom[local_idx])
        if (not np.isfinite(scale)) or scale <= 0.0:
            if abs(float(expert_arr[action_idx] - base_arr[action_idx])) > UNIT_RESIDUAL_EPS:
                clipped_values += 1
                step_unrepresentable = True
            continue
        residual_value = float(expert_arr[action_idx] - base_arr[action_idx]) / scale
        if abs(residual_value) > (1.0 + UNIT_RESIDUAL_EPS):
            clipped_values += 1
            step_unrepresentable = True
        if clip_residual_to_unit:
            clipped_residual = float(np.clip(residual_value, -1.0, 1.0))
            residual_value = clipped_residual
        projected[action_idx] = base_arr[action_idx] + (residual_value * scale)

    if action_spec.clip_gripper and projected.shape[0] > 0:
        projected[-1] = np.clip(projected[-1], -1.0, 1.0)
    return projected.astype(np.float32, copy=False), int(clipped_values), bool(
        step_unrepresentable
    )


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


def _prepare_demo_transitions(
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
        obs_raw = _build_frame_obs(payload, step_idx)
        base_chunk = np.asarray(base_chunks_per_step[step_idx], dtype=np.float32)
        residual_obs = build_chunk_residual_obs(
            obs=obs_raw,
            base_actions=base_chunk,
            image_keys=image_keys,
            residual_alpha=float(action_spec.alpha),
        )

        final_action, step_unrepresentable_values, step_unrepresentable = _project_expert_action(
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
            next_obs_raw = _build_frame_obs(payload, step_idx + 1)
            next_base_chunk = np.asarray(base_chunks_per_step[step_idx + 1], dtype=np.float32)
            next_residual_obs = build_chunk_residual_obs(
                obs=next_obs_raw,
                base_actions=next_base_chunk,
                image_keys=image_keys,
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


def _write_manifest(
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


def prepare_current_task_offline_data(
    cfg: LiberoTrainConfig,
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

    action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=cfg.env.action_dim)
    image_keys = cfg.obs.image_keys
    policy_client = build_policy_client(cfg, logger=logger)
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
        "Preparing offline dataset: task=%s backend=%s raw=%s output=%s",
        task_spec.task_key,
        describe_policy_backend(cfg),
        task_spec.dataset_path,
        prepared_dir,
    )

    try:
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
                    task_prompt=task_spec.task_description,
                    action_spec=action_spec,
                    image_keys=image_keys,
                    policy_client=policy_client,
                    expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
                    clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
                    filter_unrepresentable_steps=bool(
                        cfg.offline.prepare.filter_unrepresentable_steps
                    ),
                )
                prepare_stats["episodes_total"] = int(prepare_stats["episodes_total"]) + 1
                prepare_stats["steps_total"] = int(prepare_stats["steps_total"]) + int(
                    episode_stats["steps_total"]
                )
                prepare_stats["steps_unrepresentable"] = int(
                    prepare_stats["steps_unrepresentable"]
                ) + int(episode_stats["steps_unrepresentable"])
                prepare_stats["steps_filtered"] = int(
                    prepare_stats["steps_filtered"]
                ) + int(episode_stats["steps_filtered"])
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
    finally:
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:  # noqa: BLE001
                pass

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
        "Offline prepare complete: episodes_total=%s steps_total=%s "
        "steps_unrepresentable=%s steps_filtered=%s episodes_written=%s "
        "steps_written=%s manifest=%s",
        int(prepare_stats["episodes_total"]),
        int(prepare_stats["steps_total"]),
        int(prepare_stats["steps_unrepresentable"]),
        int(prepare_stats["steps_filtered"]),
        int(prepare_stats["episodes_written"]),
        int(prepare_stats["steps_written"]),
        manifest_path,
    )
    return OfflinePreparedInputs(
        prepared_paths=(prepared_dir,),
        prepare_stats=prepare_stats,
        manifest_paths=(manifest_path,),
    )


def _episode_files_from_manifest(manifest_path: Path) -> list[Path]:
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Invalid offline manifest: {manifest_path}")
    episode_files = manifest.get("episode_files", ())
    resolved: list[Path] = []
    for entry in episode_files:
        resolved.append((manifest_path.parent / str(entry)).resolve())
    return resolved


def resolve_prepared_episode_files(paths: Sequence[Path]) -> list[Path]:
    episode_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            manifest_path = path / MANIFEST_FILENAME
            if manifest_path.exists():
                episode_files.extend(_episode_files_from_manifest(manifest_path))
                continue
            episode_files.extend(sorted(path.glob(EPISODE_FILE_GLOB)))
            continue
        if path.name == MANIFEST_FILENAME:
            episode_files.extend(_episode_files_from_manifest(path))
            continue
        if path.suffix == ".pkl":
            episode_files.append(path.resolve())
            continue
        raise ValueError(f"Unsupported offline prepared path: {path}")
    unique_files: list[Path] = []
    seen = set()
    for path in episode_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(resolved)
    return unique_files


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
        "steps_unrepresentable": 0,
        "steps_filtered": 0,
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
        "load_errors=%s dataset_steps_total=%s dataset_steps_filtered=%s "
        "dataset_steps_written=%s",
        int(stats["files_total"]),
        int(stats["episodes_loaded"]),
        int(stats["steps_loaded"]),
        int(stats["load_errors"]),
        int(dataset_stats["steps_total"]),
        int(dataset_stats["steps_filtered"]),
        int(dataset_stats["steps_written"]),
    )
    return stats


__all__ = [
    "LiberoTaskSpec",
    "OfflinePreparedInputs",
    "OfflinePreparedResolution",
    "OFFLINE_FORMAT_VERSION",
    "load_prepared_offline_replay",
    "prepare_current_task_offline_data",
    "resolve_and_validate_prepared_paths",
    "resolve_configured_prepared_paths",
    "resolve_prepared_episode_files",
]
