from __future__ import annotations

"""AgiBot offline-data helpers aligned to the LIBERO train flow.

This module intentionally mirrors the shape of ``examples/libero/env/offline_data.py``
for the first alignment pass. The prepared replay contract is stable; raw inputs
can be either the legacy reference pickle episodes or LeRobot v2.x episode
parquet files.
"""

import dataclasses
import io
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import TYPE_CHECKING

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

from .observation import build_agibot_state
from .observation import extract_agibot_residual_images

if TYPE_CHECKING:
    from ..config import AgiBotTrainConfig

OFFLINE_FORMAT_VERSION = "agibot_real_offline_step_transitions_v1"
PICKLE_REFERENCE_SOURCE_FORMAT = "agibot_reference_episode_pickle_v1"
LEROBOT_REFERENCE_SOURCE_FORMAT = "agibot_lerobot_episode_parquet_v1"
REFERENCE_SOURCE_FORMAT = "agibot_reference_episode_raw_v2"
REFERENCE_NOTE = (
    "AgiBot residual offline pipeline modeled after examples/libero. "
    "Raw sources may be legacy reference pickle episodes or LeRobot v2.x datasets."
)
LEROBOT_INFO_RELATIVE_PATH = Path("meta") / "info.json"
LEROBOT_DATA_FILE_GLOB = "data/chunk-*/episode_*.parquet"
LEROBOT_STATE_COLUMN = "observation.state"
LEROBOT_JOYRA_ACTION_COLUMN = "action"
LEROBOT_OPENPI_ACTION_COLUMN = "actions"
LEROBOT_TARGET_STATE_ACTION_DIM = 14
LEROBOT_IMAGE_KEY_MAP = {
    "observation.images.head_color": "image/head",
    "observation.images.hand_left": "image/left_wrist",
    "observation.images.hand_right": "image/right_wrist",
}


@dataclasses.dataclass(frozen=True, slots=True)
class AgiBotTaskSpec:
    task_name: str
    task_key: str
    task_description: str
    dataset_path: Path


@dataclasses.dataclass(slots=True)
class _LazyLerobotImageBytes:
    data: bytes
    _array: np.ndarray | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    def as_array(self) -> np.ndarray:
        if self._array is None:
            from PIL import Image

            image = Image.open(io.BytesIO(self.data)).convert("RGB")
            self._array = np.asarray(image, dtype=np.uint8)
        return self._array

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        array = self.as_array()
        if dtype is not None:
            return np.array(array, dtype=dtype, copy=True if copy is None else copy)
        if copy:
            return np.array(array, copy=True)
        return array


@dataclasses.dataclass(slots=True)
class _LazyLerobotVideoFrame:
    video_path: Path
    frame_index: int
    _array: np.ndarray | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    def as_array(self) -> np.ndarray:
        if self._array is None:
            import imageio.v3 as iio

            frame = iio.imread(self.video_path, index=int(self.frame_index))
            self._array = np.asarray(frame, dtype=np.uint8)
        return self._array

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        array = self.as_array()
        if dtype is not None:
            return np.array(array, dtype=dtype, copy=True if copy is None else copy)
        if copy:
            return np.array(array, copy=True)
        return array


def resolve_task_spec(cfg: AgiBotTrainConfig) -> AgiBotTaskSpec:
    dataset_override = cfg.offline.prepare.raw_dataset_path
    if dataset_override is None:
        raise ValueError(
            "Temporary AgiBot offline prepare requires offline.prepare.raw_dataset_path"
        )
    dataset_path = resolve_path(dataset_override, base=resolve_original_cwd())
    return AgiBotTaskSpec(
        task_name=str(cfg.task.name),
        task_key=str(cfg.task.task_key),
        task_description=str(cfg.task.prompt),
        dataset_path=dataset_path,
    )


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / LEROBOT_INFO_RELATIVE_PATH).is_file()


def _find_lerobot_dataset_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _is_lerobot_dataset_dir(candidate):
            return candidate
    return None


def _resolve_lerobot_episode_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file() and dataset_path.suffix == ".parquet":
        dataset_root = _find_lerobot_dataset_root(dataset_path)
        if dataset_root is None:
            raise ValueError(
                f"LeRobot parquet file is not under a dataset root with meta/info.json: {dataset_path}"
            )
        return [dataset_path.resolve()]

    if not dataset_path.is_dir():
        return []

    if _is_lerobot_dataset_dir(dataset_path):
        episode_files = sorted(dataset_path.glob(LEROBOT_DATA_FILE_GLOB))
        return [path.resolve() for path in episode_files]

    dataset_roots = sorted(
        path.parent.parent
        for path in dataset_path.glob(f"**/{LEROBOT_INFO_RELATIVE_PATH.as_posix()}")
        if path.is_file()
    )
    episode_files: list[Path] = []
    for dataset_root in dataset_roots:
        episode_files.extend(sorted(dataset_root.glob(LEROBOT_DATA_FILE_GLOB)))
    return [path.resolve() for path in episode_files]


def resolve_reference_source_format(dataset_path: Path) -> str:
    if dataset_path.is_file() and dataset_path.suffix == ".pkl":
        return PICKLE_REFERENCE_SOURCE_FORMAT
    if dataset_path.is_file() and dataset_path.suffix == ".parquet":
        return LEROBOT_REFERENCE_SOURCE_FORMAT
    if dataset_path.is_dir():
        if sorted(dataset_path.glob(EPISODE_FILE_GLOB)):
            return PICKLE_REFERENCE_SOURCE_FORMAT
        if _resolve_lerobot_episode_files(dataset_path):
            return LEROBOT_REFERENCE_SOURCE_FORMAT
    raise ValueError(
        "AgiBot offline prepare expects offline.prepare.raw_dataset_path to be "
        f"a directory of {EPISODE_FILE_GLOB} files, a single .pkl file, or a LeRobot v2.x "
        f"dataset containing {LEROBOT_DATA_FILE_GLOB}; got {dataset_path}"
    )


def prepare_fingerprint(
    cfg: AgiBotTrainConfig,
    *,
    task_spec: AgiBotTaskSpec,
) -> dict[str, Any]:
    return build_residual_prepared_fingerprint(
        format_version=OFFLINE_FORMAT_VERSION,
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
        extra_fields={
            "raw_source_format": resolve_reference_source_format(
                task_spec.dataset_path,
            )
        },
    )


def training_compatibility_signature(cfg: AgiBotTrainConfig) -> dict[str, Any]:
    return build_residual_training_signature(
        task_key=str(cfg.task.task_key),
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
    cfg: AgiBotTrainConfig,
    *,
    task_spec: AgiBotTaskSpec,
) -> Path:
    return resolve_residual_prepared_dir(
        output_root=cfg.offline.prepare.output_root,
        task_key=str(task_spec.task_key),
        policy_backend=describe_policy_backend(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        alpha=float(cfg.residual.alpha),
    )


def _manifest_signature(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return extract_residual_manifest_signature(manifest)


def resolve_configured_prepared_paths(cfg: AgiBotTrainConfig) -> tuple[Path, ...]:
    return resolve_prepared_path_value(cfg.offline.prepared_path)


def resolve_and_validate_prepared_paths(
    cfg: AgiBotTrainConfig,
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


def _coerce_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _coerce_obs_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_obs_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_coerce_obs_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def normalize_episode_steps(payload: Any, *, source_path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        steps = payload
    elif isinstance(payload, tuple):
        steps = list(payload)
    elif isinstance(payload, dict):
        for key in ("steps", "transitions", "episode"):
            candidate = payload.get(key, None)
            if isinstance(candidate, list):
                steps = candidate
                break
        else:
            raise ValueError(
                f"Unsupported raw offline episode payload keys for {source_path}: "
                f"{sorted(payload.keys())}"
            )
    else:
        raise ValueError(
            f"Raw offline episode must be list/tuple/dict, got {type(payload)} from {source_path}"
        )
    if not steps:
        raise ValueError(f"Raw offline episode has no steps: {source_path}")
    normalized: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(
                f"Raw offline step must be a dict, got {type(step)} in {source_path}"
            )
        normalized.append(dict(step))
    return normalized


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_lerobot_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / LEROBOT_INFO_RELATIVE_PATH
    with open(info_path, encoding="utf-8") as fp:
        info = json.load(fp)
    if not isinstance(info, dict):
        raise ValueError(f"LeRobot info.json must contain an object: {info_path}")
    return info


def _episode_index_from_lerobot_path(parquet_path: Path) -> int:
    stem = parquet_path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected LeRobot episode parquet name: {parquet_path}")
    return int(stem[len(prefix) :])


def _episode_chunk_from_lerobot_path(
    parquet_path: Path,
    *,
    episode_index: int,
    info: dict[str, Any],
) -> int:
    chunk_name = parquet_path.parent.name
    prefix = "chunk-"
    if chunk_name.startswith(prefix):
        return int(chunk_name[len(prefix) :])
    return int(episode_index // int(info.get("chunks_size", 1000)))


def _patch_fastparquet_dotted_schema(parquet_file: Any) -> None:
    schema_helper = parquet_file.schema
    original_schema_element = schema_helper.schema_element

    def normalize_parts(parts: str | Sequence[str]) -> list[str]:
        raw_parts = parts.split(".") if isinstance(parts, str) else list(parts)
        if not raw_parts:
            return raw_parts
        root_children = schema_helper.root["children"]
        if raw_parts[0] in root_children:
            return raw_parts
        for end_idx in range(len(raw_parts), 0, -1):
            candidate = ".".join(raw_parts[:end_idx])
            if candidate in root_children:
                return [candidate, *raw_parts[end_idx:]]
        return raw_parts

    def schema_element(name: str | Sequence[str]) -> Any:
        return original_schema_element(normalize_parts(name))

    def is_required(name: str | Sequence[str]) -> bool:
        from fastparquet import parquet_thrift

        required = True
        parts = normalize_parts(name)
        for idx in range(len(parts)):
            element = schema_element(parts[: idx + 1])
            if (
                element.repetition_type
                != parquet_thrift.FieldRepetitionType.REQUIRED
            ):
                required = False
                break
        return required

    def max_definition_level(parts: str | Sequence[str]) -> int:
        from fastparquet import parquet_thrift

        max_level = 0
        normalized_parts = normalize_parts(parts)
        for idx in range(len(normalized_parts)):
            element = schema_element(normalized_parts[: idx + 1])
            if (
                element.repetition_type
                != parquet_thrift.FieldRepetitionType.REQUIRED
            ):
                max_level += 1
        return max_level

    def max_repetition_level(parts: str | Sequence[str]) -> int:
        from fastparquet import parquet_thrift

        max_level = 0
        normalized_parts = normalize_parts(parts)
        for idx in range(len(normalized_parts)):
            element = schema_element(normalized_parts[: idx + 1])
            if (
                element.repetition_type
                == parquet_thrift.FieldRepetitionType.REPEATED
            ):
                max_level += 1
        return max_level

    schema_helper.schema_element = schema_element
    schema_helper.is_required = is_required
    schema_helper.max_definition_level = max_definition_level
    schema_helper.max_repetition_level = max_repetition_level


def _read_pyarrow_parquet_column(parquet_path: Path, column: str) -> list[Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[column])
    if column not in table.column_names:
        raise KeyError(f"Parquet column {column!r} not found in {parquet_path}")
    column_array = table.column(column)
    return [column_array[idx].as_py() for idx in range(table.num_rows)]


def _read_fastparquet_column(
    parquet_path: Path,
    column: str,
    *,
    patch_dotted_schema: bool,
) -> list[Any]:
    from fastparquet import ParquetFile

    parquet_file = ParquetFile(parquet_path)
    if patch_dotted_schema:
        _patch_fastparquet_dotted_schema(parquet_file)
    frame = parquet_file.to_pandas(columns=[column])
    if column not in frame.columns:
        raise KeyError(f"Parquet column {column!r} not found in {parquet_path}")
    return frame[column].tolist()


def _read_parquet_column_values(parquet_path: Path, column: str) -> list[Any]:
    errors: list[str] = []
    for reader_name, reader_fn in (
        ("pyarrow", lambda: _read_pyarrow_parquet_column(parquet_path, column)),
        (
            "fastparquet",
            lambda: _read_fastparquet_column(
                parquet_path,
                column,
                patch_dotted_schema=False,
            ),
        ),
        (
            "fastparquet patched dotted schema",
            lambda: _read_fastparquet_column(
                parquet_path,
                column,
                patch_dotted_schema=True,
            ),
        ),
    ):
        try:
            return reader_fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reader_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Unable to read LeRobot parquet column {column!r} from {parquet_path}. "
        "Install compatible pyarrow/fastparquet versions or rewrite the parquet. "
        f"Reader errors: {' | '.join(errors)}"
    )


def _coerce_lerobot_state_action_vector(
    value: Any,
    *,
    column: str,
    source_path: Path,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(vector.shape[0]) == LEROBOT_TARGET_STATE_ACTION_DIM:
        return vector
    if int(vector.shape[0]) == 30:
        return np.asarray(vector[-LEROBOT_TARGET_STATE_ACTION_DIM:], dtype=np.float32)
    raise ValueError(
        f"LeRobot {column} in {source_path} must be 14D OpenPI data or 30D JoyRA data, "
        f"got {vector.shape}"
    )


def _lerobot_action_column(info: dict[str, Any], *, source_path: Path) -> str:
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError(f"LeRobot info.json has invalid features: {source_path}")
    if LEROBOT_OPENPI_ACTION_COLUMN in features:
        return LEROBOT_OPENPI_ACTION_COLUMN
    if LEROBOT_JOYRA_ACTION_COLUMN in features:
        return LEROBOT_JOYRA_ACTION_COLUMN
    raise ValueError(
        f"LeRobot dataset missing action/actions feature in {source_path}"
    )


def _lerobot_video_path(
    dataset_root: Path,
    *,
    info: dict[str, Any],
    video_key: str,
    episode_index: int,
    episode_chunk: int,
) -> Path:
    template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    return dataset_root / template.format(
        episode_chunk=int(episode_chunk),
        video_key=str(video_key),
        episode_index=int(episode_index),
    )


def _lerobot_image_values(
    *,
    dataset_root: Path,
    parquet_path: Path,
    info: dict[str, Any],
    episode_index: int,
    episode_chunk: int,
    frame_indices: Sequence[int],
) -> dict[str, list[Any]]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError(f"LeRobot info.json has invalid features: {parquet_path}")

    images: dict[str, list[Any]] = {}
    for lerobot_key, agibot_key in LEROBOT_IMAGE_KEY_MAP.items():
        feature = features.get(lerobot_key, None)
        if not isinstance(feature, dict):
            raise ValueError(
                f"LeRobot dataset missing required image feature {lerobot_key}: {parquet_path}"
            )
        dtype = str(feature.get("dtype", ""))
        if dtype == "video":
            video_path = _lerobot_video_path(
                dataset_root,
                info=info,
                video_key=lerobot_key,
                episode_index=episode_index,
                episode_chunk=episode_chunk,
            )
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"LeRobot video file for {lerobot_key} is missing: {video_path}"
                )
            images[agibot_key] = [
                _LazyLerobotVideoFrame(video_path=video_path, frame_index=int(frame_idx))
                for frame_idx in frame_indices
            ]
            continue
        if dtype == "image":
            byte_column = f"{lerobot_key}.bytes"
            byte_values = _read_parquet_column_values(parquet_path, byte_column)
            if len(byte_values) != len(frame_indices):
                raise ValueError(
                    f"LeRobot image/state length mismatch for {byte_column} in {parquet_path}: "
                    f"{len(byte_values)} vs {len(frame_indices)}"
                )
            image_refs: list[Any] = []
            for value in byte_values:
                if isinstance(value, (bytes, bytearray, memoryview)):
                    image_refs.append(_LazyLerobotImageBytes(bytes(value)))
                else:
                    raise ValueError(
                        f"LeRobot image column {byte_column} must contain bytes, "
                        f"got {type(value)} in {parquet_path}"
                    )
            images[agibot_key] = image_refs
            continue
        raise ValueError(
            f"Unsupported LeRobot image dtype for {lerobot_key} in {parquet_path}: {dtype!r}"
        )
    return images


def _lerobot_task_prompts(dataset_root: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for record in _read_jsonl_records(dataset_root / "meta" / "tasks.jsonl"):
        if "task_index" in record and "task" in record:
            prompts[int(record["task_index"])] = str(record["task"])
    return prompts


def _lerobot_episode_tasks(dataset_root: Path) -> dict[int, str]:
    episode_tasks: dict[int, str] = {}
    for record in _read_jsonl_records(dataset_root / "meta" / "episodes.jsonl"):
        tasks = record.get("tasks", None)
        if "episode_index" in record and isinstance(tasks, list) and tasks:
            episode_tasks[int(record["episode_index"])] = str(tasks[0])
    return episode_tasks


def load_lerobot_episode_steps(parquet_path: Path) -> list[dict[str, Any]]:
    dataset_root = _find_lerobot_dataset_root(parquet_path)
    if dataset_root is None:
        raise ValueError(
            f"LeRobot parquet file is not under a dataset root with meta/info.json: {parquet_path}"
        )

    info = _read_lerobot_info(dataset_root)
    episode_index = _episode_index_from_lerobot_path(parquet_path)
    episode_chunk = _episode_chunk_from_lerobot_path(
        parquet_path,
        episode_index=episode_index,
        info=info,
    )
    action_column = _lerobot_action_column(info, source_path=parquet_path)

    state_values = _read_parquet_column_values(parquet_path, LEROBOT_STATE_COLUMN)
    action_values = _read_parquet_column_values(parquet_path, action_column)
    frame_values = _read_parquet_column_values(parquet_path, "frame_index")
    task_values: list[Any] | None
    try:
        task_values = _read_parquet_column_values(parquet_path, "task_index")
    except RuntimeError:
        task_values = None

    if len(state_values) != len(action_values):
        raise ValueError(
            f"LeRobot state/action length mismatch in {parquet_path}: "
            f"{len(state_values)} vs {len(action_values)}"
        )
    if len(frame_values) != len(state_values):
        raise ValueError(
            f"LeRobot frame/state length mismatch in {parquet_path}: "
            f"{len(frame_values)} vs {len(state_values)}"
        )

    frame_indices = [int(value) for value in frame_values]
    image_values = _lerobot_image_values(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        info=info,
        episode_index=episode_index,
        episode_chunk=episode_chunk,
        frame_indices=frame_indices,
    )
    task_prompts = _lerobot_task_prompts(dataset_root)
    episode_tasks = _lerobot_episode_tasks(dataset_root)
    fallback_prompt = episode_tasks.get(episode_index, "")

    steps: list[dict[str, Any]] = []
    num_steps = len(state_values)
    for step_idx in range(num_steps):
        task_prompt = fallback_prompt
        if task_values is not None:
            task_prompt = task_prompts.get(int(task_values[step_idx]), task_prompt)

        observations: dict[str, Any] = {
            "state/pose": _coerce_lerobot_state_action_vector(
                state_values[step_idx],
                column=LEROBOT_STATE_COLUMN,
                source_path=parquet_path,
            ),
        }
        for agibot_key, values in image_values.items():
            observations[agibot_key] = values[step_idx]

        is_last = bool(step_idx >= num_steps - 1)
        steps.append(
            {
                "observations": observations,
                "expert_action": _coerce_lerobot_state_action_vector(
                    action_values[step_idx],
                    column=action_column,
                    source_path=parquet_path,
                ),
                "reward": 1.0 if is_last else 0.0,
                "done": is_last,
                "success": is_last,
                "task_prompt": task_prompt,
                "metadata": {
                    "source_format": LEROBOT_REFERENCE_SOURCE_FORMAT,
                    "dataset_root": str(dataset_root),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_indices[step_idx]),
                    "action_column": action_column,
                },
            }
        )
    return steps


def load_reference_raw_episode_steps(source_path: Path) -> list[dict[str, Any]]:
    if source_path.suffix == ".pkl":
        with open(source_path, "rb") as fp:
            raw_payload = pickle.load(fp)
        return normalize_episode_steps(raw_payload, source_path=source_path)
    if source_path.suffix == ".parquet":
        return load_lerobot_episode_steps(source_path)
    raise ValueError(
        f"Unsupported AgiBot raw offline episode file: {source_path}. "
        "Expected .pkl or LeRobot .parquet."
    )


def _raw_obs_from_step(step: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    obs = step.get("observations", None)
    if obs is None:
        obs = step.get("obs", None)
    if not isinstance(obs, dict):
        raise ValueError(f"Raw offline step missing observations in {source_path}")
    return _coerce_obs_tree(obs)


def _raw_next_obs_from_step(step: dict[str, Any]) -> dict[str, Any] | None:
    next_obs = step.get("next_observations", None)
    if next_obs is None:
        next_obs = step.get("next_obs", None)
    if next_obs is None:
        return None
    if not isinstance(next_obs, dict):
        raise ValueError("Raw offline next_observations must be a dict when provided")
    return _coerce_obs_tree(next_obs)


def _expert_action_from_step(step: dict[str, Any], *, action_dim: int) -> np.ndarray:
    for key in ("expert_action", "action", "actions"):
        value = step.get(key, None)
        if value is None:
            continue
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(action.shape[0]) != int(action_dim):
            raise ValueError(
                f"Raw offline action must be {int(action_dim)}D, got {action.shape}"
            )
        return action
    raise ValueError("Raw offline step missing expert action")


def _reward_from_step(step: dict[str, Any]) -> float:
    value = step.get("rewards", step.get("reward", 0.0))
    return float(value)


def _done_from_step(step: dict[str, Any], *, is_last: bool) -> bool:
    return bool(step.get("dones", step.get("done", False))) or bool(is_last)


def _truncated_from_step(step: dict[str, Any]) -> bool:
    return bool(step.get("truncated", False))


def _success_from_step(step: dict[str, Any]) -> bool:
    return bool(step.get("success", False))


def _base_chunk_from_step(
    step: dict[str, Any],
    *,
    chunk_horizon: int,
) -> np.ndarray | None:
    for key in ("base_chunk", "base_action_chunk"):
        value = step.get(key, None)
        if value is None:
            continue
        return prepare_base_actions_chunk(
            base_actions=np.asarray(value, dtype=np.float32),
            chunk_horizon=chunk_horizon,
            source=f"raw offline {key}",
        )
    return None


def _precompute_base_chunks_for_steps(
    steps: Sequence[dict[str, Any]],
    *,
    task_prompt: str,
    base_policy: Any,
    chunk_horizon: int,
    action_dim: int,
    source_path: Path,
) -> np.ndarray:
    precomputed_chunks: list[np.ndarray] = []
    missing_indices: list[int] = []

    for step_idx, step in enumerate(steps):
        maybe_base_chunk = _base_chunk_from_step(
            step,
            chunk_horizon=int(chunk_horizon),
        )
        if maybe_base_chunk is None:
            missing_indices.append(int(step_idx))
            precomputed_chunks.append(
                np.zeros((int(chunk_horizon), int(action_dim)), dtype=np.float32)
            )
        else:
            precomputed_chunks.append(np.asarray(maybe_base_chunk, dtype=np.float32))

    if not missing_indices:
        return np.asarray(precomputed_chunks, dtype=np.float32)

    step_iter: Iterable[int] = tqdm(
        missing_indices,
        total=len(missing_indices),
        desc=f"offline base chunks {source_path.name}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    for step_idx in step_iter:
        obs_raw = _raw_obs_from_step(steps[step_idx], source_path=source_path)
        step_task_prompt = str(steps[step_idx].get("task_prompt") or task_prompt)
        action_chunk, _ = base_policy.infer(obs_raw, prompt=step_task_prompt)
        precomputed_chunks[step_idx] = prepare_base_actions_chunk(
            base_actions=action_chunk,
            chunk_horizon=chunk_horizon,
            source="offline base policy",
        )
    return np.asarray(precomputed_chunks, dtype=np.float32)


def _zero_like_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs.items()}


def prepare_reference_episode_transitions(
    *,
    raw_steps: Sequence[dict[str, Any]],
    episode_id: int,
    task_prompt: str,
    action_spec: ResidualActionSpec,
    image_keys: tuple[str, ...],
    base_policy: Any,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    filter_unrepresentable_steps: bool,
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_steps = int(len(raw_steps))
    if num_steps <= 0:
        raise ValueError(f"Raw offline episode has no steps: {source_path}")

    base_chunks_per_step = _precompute_base_chunks_for_steps(
        raw_steps,
        task_prompt=task_prompt,
        base_policy=base_policy,
        chunk_horizon=int(action_spec.chunk_horizon),
        action_dim=int(action_spec.action_dim),
        source_path=source_path,
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
    episode_success = False

    for step_idx, raw_step in enumerate(raw_steps):
        obs_raw = _raw_obs_from_step(raw_step, source_path=source_path)
        base_chunk = np.asarray(base_chunks_per_step[step_idx], dtype=np.float32)
        residual_obs = build_chunk_residual_obs(
            robot_state=build_agibot_state(obs_raw),
            images=extract_agibot_residual_images(
                obs_raw,
                image_keys=image_keys,
            ),
            image_keys=image_keys,
            base_actions=base_chunk,
            residual_alpha=float(action_spec.alpha),
        )

        final_action, step_unrepresentable_values, step_unrepresentable = project_expert_action(
            expert_action=_expert_action_from_step(
                raw_step,
                action_dim=int(action_spec.action_dim),
            ),
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

        is_last = bool(step_idx >= (num_steps - 1))
        reward = _reward_from_step(raw_step)
        done = _done_from_step(raw_step, is_last=is_last)
        truncated = _truncated_from_step(raw_step)
        episode_done = bool(done or truncated)
        episode_return += float(reward)
        episode_success = bool(episode_success or _success_from_step(raw_step))

        if episode_done:
            next_residual_obs = _zero_like_obs(residual_obs)
            mask = 0.0
        else:
            next_obs_raw = _raw_next_obs_from_step(raw_step)
            if next_obs_raw is None:
                next_obs_raw = _raw_obs_from_step(
                    raw_steps[step_idx + 1],
                    source_path=source_path,
                )
            next_base_chunk = np.asarray(base_chunks_per_step[step_idx + 1], dtype=np.float32)
            next_residual_obs = build_chunk_residual_obs(
                robot_state=build_agibot_state(next_obs_raw),
                images=extract_agibot_residual_images(
                    next_obs_raw,
                    image_keys=image_keys,
                ),
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
                "dones": bool(episode_done),
            }
        )

    episode_stats = {
        "episode_id": int(episode_id),
        "steps_total": int(num_steps),
        "steps_written": int(len(transitions)),
        "steps_unrepresentable": int(steps_unrepresentable),
        "steps_filtered": int(steps_filtered),
        "episode_return": float(episode_return),
        "success": bool(episode_success),
        "unrepresentable_values": int(unrepresentable_values),
    }
    return transitions, episode_stats


def write_manifest(
    *,
    manifest_path: Path,
    task_spec: AgiBotTaskSpec,
    cfg: AgiBotTrainConfig,
    fingerprint: dict[str, Any],
    episode_files: Sequence[Path],
    prepare_stats: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "format_version": OFFLINE_FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "reference_only": True,
        "reference_note": REFERENCE_NOTE,
        "task": {
            "task_name": str(task_spec.task_name),
            "task_key": str(task_spec.task_key),
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
            "raw_source_format": resolve_reference_source_format(
                task_spec.dataset_path,
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


def resolve_reference_raw_episode_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_dir():
        episode_files = sorted(dataset_path.glob(EPISODE_FILE_GLOB))
        if episode_files:
            return [path.resolve() for path in episode_files]
        lerobot_episode_files = _resolve_lerobot_episode_files(dataset_path)
        if lerobot_episode_files:
            return lerobot_episode_files
        raise FileNotFoundError(
            f"AgiBot raw offline directory has no {EPISODE_FILE_GLOB} files and no "
            f"LeRobot {LEROBOT_DATA_FILE_GLOB} files: {dataset_path}"
        )
    if dataset_path.is_file() and dataset_path.suffix == ".pkl":
        return [dataset_path.resolve()]
    if dataset_path.is_file() and dataset_path.suffix == ".parquet":
        return _resolve_lerobot_episode_files(dataset_path)
    raise ValueError(
        "AgiBot offline prepare expects offline.prepare.raw_dataset_path "
        f"to be a directory of {EPISODE_FILE_GLOB} files, a single .pkl file, or "
        f"a LeRobot dataset containing {LEROBOT_DATA_FILE_GLOB}; got {dataset_path}"
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
    "AgiBotTaskSpec",
    "OfflinePreparedResolution",
    "OFFLINE_FORMAT_VERSION",
    "LEROBOT_REFERENCE_SOURCE_FORMAT",
    "PICKLE_REFERENCE_SOURCE_FORMAT",
    "REFERENCE_NOTE",
    "REFERENCE_SOURCE_FORMAT",
    "load_lerobot_episode_steps",
    "load_prepared_offline_replay",
    "load_reference_raw_episode_steps",
    "normalize_episode_steps",
    "prepare_fingerprint",
    "prepare_reference_episode_transitions",
    "prepared_dir_for_cfg",
    "resolve_and_validate_prepared_paths",
    "resolve_configured_prepared_paths",
    "resolve_prepared_episode_files",
    "resolve_reference_raw_episode_files",
    "resolve_reference_source_format",
    "resolve_task_spec",
    "write_manifest",
]
