from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import pickle
import time
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np

from .processor_protocol import normalize_chunk_result
from ..env.observation import build_libero_state

RAW_ROLLOUT_FORMAT_VERSION = "libero_raw_rollout_sequence_v1"
RAW_ROLLOUT_MANIFEST_FILENAME = "manifest.json"


@dataclass
class _EpisodeBuffer:
    episode_id: int
    task_prompt: str
    observations: list[dict[str, Any]]
    states: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[bool]
    truncations: list[bool]
    infos: list[dict[str, Any]]
    chunk_seqs: list[int]
    seen_chunk_seqs: set[int]
    episode_return: float
    success: bool
    tainted: bool


def _deepcopy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(obs))


def _structures_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except Exception:  # noqa: BLE001
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        if left_keys != right_keys:
            return False
        return all(_structures_equal(left[key], right[key]) for key in left_keys)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_structures_equal(lv, rv) for lv, rv in zip(left, right))
    return bool(left == right)


class RawRolloutRecorder:
    def __init__(
        self,
        *,
        output_root: Path,
        logger: Any,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self._logger = logger
        self._manifest_path = self.output_root / RAW_ROLLOUT_MANIFEST_FILENAME
        self._episode_buffers: dict[int, _EpisodeBuffer] = {}
        self._episode_files: list[str] = []
        self._episode_file_set: set[str] = set()
        self._episodes_written = 0
        self._steps_written = 0
        self._append_errors = 0
        self._write_errors = 0

        self.output_root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def append_chunk(self, *, payload: dict[str, Any]) -> bool:
        episode_id = int(payload["episode_id"])
        chunk_seq = int(payload["chunk_seq"])
        task_prompt = str(payload["task_prompt"])
        buffer = self._episode_buffers.get(episode_id)
        if buffer is None:
            buffer = _EpisodeBuffer(
                episode_id=episode_id,
                task_prompt=task_prompt,
                observations=[],
                states=[],
                actions=[],
                rewards=[],
                dones=[],
                truncations=[],
                infos=[],
                chunk_seqs=[],
                seen_chunk_seqs=set(),
                episode_return=0.0,
                success=False,
                tainted=False,
            )
            self._episode_buffers[episode_id] = buffer

        if buffer.tainted:
            buffer.seen_chunk_seqs.add(chunk_seq)
            return False

        if chunk_seq in buffer.seen_chunk_seqs:
            return False

        try:
            normalized_chunk = normalize_chunk_result(dict(payload["chunk_result"]))
            steps = list(normalized_chunk.steps)
            if not steps:
                raise ValueError(
                    "raw rollout recorder received an empty normalized chunk"
                )

            episode_step_start = int(payload["episode_step_start"])
            expected_step_start = int(len(buffer.actions))
            if episode_step_start != expected_step_start:
                raise ValueError(
                    "raw rollout recorder expected contiguous episode steps: "
                    f"episode_id={episode_id} chunk_seq={chunk_seq} "
                    f"episode_step_start={episode_step_start} expected={expected_step_start}"
                )

            chunk_start_obs = _deepcopy_obs(dict(steps[0]["obs"]))
            if not buffer.observations:
                buffer.observations.append(chunk_start_obs)
                buffer.states.append(
                    np.asarray(build_libero_state(chunk_start_obs), dtype=np.float32)
                )
            elif not _structures_equal(buffer.observations[-1], chunk_start_obs):
                raise ValueError(
                    "raw rollout recorder observed a chunk boundary mismatch: "
                    f"episode_id={episode_id} chunk_seq={chunk_seq}"
                )

            for step in steps:
                step_dict = dict(step)
                next_obs = _deepcopy_obs(dict(step_dict["next_obs"]))
                step_info = copy.deepcopy(dict(step_dict["info"]))

                buffer.actions.append(
                    np.asarray(step_dict["action"], dtype=np.float32).reshape(-1).copy()
                )
                buffer.rewards.append(float(step_dict["reward"]))
                buffer.dones.append(bool(step_dict["done"]))
                buffer.truncations.append(bool(step_dict.get("truncated", False)))
                buffer.infos.append(step_info)
                buffer.chunk_seqs.append(chunk_seq)
                buffer.observations.append(next_obs)
                buffer.states.append(
                    np.asarray(build_libero_state(next_obs), dtype=np.float32)
                )
                buffer.episode_return += float(step_dict["reward"])
                buffer.success = bool(
                    buffer.success or bool(step_info.get("env_done", False))
                )

            buffer.success = bool(
                buffer.success
                or bool(normalized_chunk.chunk_info.get("env_done", False))
            )
            buffer.seen_chunk_seqs.add(chunk_seq)
            return True
        except Exception:
            buffer.tainted = True
            raise

    def finalize_episode(self, *, marker: dict[str, Any]) -> Path | None:
        episode_id = int(marker.get("episode_id", -1))
        buffer = self._episode_buffers.get(episode_id, None)
        if buffer is None:
            return None

        if buffer.tainted:
            self._logger.warning(
                "skip writing tainted raw rollout episode: episode_id=%s",
                int(episode_id),
            )
            self._episode_buffers.pop(episode_id, None)
            self._write_manifest()
            return None

        if not buffer.actions:
            self._logger.warning(
                "skip writing empty raw rollout episode: episode_id=%s",
                int(episode_id),
            )
            self._episode_buffers.pop(episode_id, None)
            self._write_manifest()
            return None

        episode_payload = {
            "format_version": RAW_ROLLOUT_FORMAT_VERSION,
            "episode_id": int(buffer.episode_id),
            "task_prompt": str(buffer.task_prompt),
            "num_steps": int(len(buffer.actions)),
            "observations": list(buffer.observations),
            "states": [
                np.asarray(state, dtype=np.float32).copy() for state in buffer.states
            ],
            "actions": [
                np.asarray(action, dtype=np.float32).copy() for action in buffer.actions
            ],
            "rewards": [float(value) for value in buffer.rewards],
            "dones": [bool(value) for value in buffer.dones],
            "truncations": [bool(value) for value in buffer.truncations],
            "infos": [copy.deepcopy(info) for info in buffer.infos],
            "chunk_seqs": [int(value) for value in buffer.chunk_seqs],
            "episode_return": float(buffer.episode_return),
            "success": bool(buffer.success),
        }

        episode_path = self.output_root / f"episode_{int(buffer.episode_id):06d}.pkl"
        episode_name = str(episode_path.name)
        next_episode_files = list(self._episode_files)
        if episode_name not in self._episode_file_set:
            next_episode_files.append(episode_name)
        next_episodes_written = int(self._episodes_written) + 1
        next_steps_written = int(self._steps_written) + int(len(buffer.actions))
        try:
            with open(episode_path, "wb") as fp:
                pickle.dump(episode_payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
            self._write_manifest(
                episode_files=next_episode_files,
                episodes_written=next_episodes_written,
                steps_written=next_steps_written,
                append_errors=self._append_errors,
                write_errors=self._write_errors,
            )
            if episode_name not in self._episode_file_set:
                self._episode_file_set.add(episode_name)
                self._episode_files.append(episode_name)
            self._episodes_written = int(next_episodes_written)
            self._steps_written = int(next_steps_written)
            self._episode_buffers.pop(episode_id, None)
            return episode_path
        except Exception:
            self.record_write_error()
            try:
                self._write_manifest()
            except Exception:  # noqa: BLE001
                pass
            raise

    def discard_pending(self) -> int:
        pending = int(len(self._episode_buffers))
        self._episode_buffers.clear()
        return pending

    def record_append_error(self) -> None:
        self._append_errors += 1
        try:
            self._write_manifest()
        except Exception:  # noqa: BLE001
            pass

    def record_write_error(self) -> None:
        self._write_errors += 1

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "output_root": str(self.output_root),
            "manifest_path": str(self._manifest_path),
            "episodes_written": int(self._episodes_written),
            "steps_written": int(self._steps_written),
            "append_errors": int(self._append_errors),
            "write_errors": int(self._write_errors),
            "pending_episodes": int(len(self._episode_buffers)),
        }

    def _write_manifest(
        self,
        *,
        episode_files: list[str] | None = None,
        episodes_written: int | None = None,
        steps_written: int | None = None,
        append_errors: int | None = None,
        write_errors: int | None = None,
    ) -> None:
        manifest_payload = {
            "format_version": RAW_ROLLOUT_FORMAT_VERSION,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "episode_files": (
                list(self._episode_files)
                if episode_files is None
                else [str(value) for value in episode_files]
            ),
            "recycle_stats": {
                "episodes_written": (
                    int(self._episodes_written)
                    if episodes_written is None
                    else int(episodes_written)
                ),
                "steps_written": (
                    int(self._steps_written)
                    if steps_written is None
                    else int(steps_written)
                ),
                "append_errors": (
                    int(self._append_errors)
                    if append_errors is None
                    else int(append_errors)
                ),
                "write_errors": (
                    int(self._write_errors)
                    if write_errors is None
                    else int(write_errors)
                ),
            },
        }
        with open(self._manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest_payload, fp, indent=2, ensure_ascii=False)


__all__ = [
    "RAW_ROLLOUT_FORMAT_VERSION",
    "RAW_ROLLOUT_MANIFEST_FILENAME",
    "RawRolloutRecorder",
]
