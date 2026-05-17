from __future__ import annotations

"""AgiBot raw rollout recorder for recovery and processor replay."""

import copy
from dataclasses import dataclass
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

RAW_ROLLOUT_FORMAT_VERSION = "agibot_raw_rollout_sequence_v1"
RAW_ROLLOUT_SCHEMA_VERSION = 1
RAW_ROLLOUT_MANIFEST_FILENAME = "manifest.json"


@dataclass
class _EpisodeBuffer:
    episode_id: int
    task_prompt: str
    chunks: list[dict[str, Any]]
    seen_chunk_seqs: set[int]
    expected_step_start: int
    episode_return: float
    success: bool
    tainted: bool
    started_at: float


def _copy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {str(key): _copy_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return copy.deepcopy(value)


def _executed_steps(chunk_result: dict[str, Any]) -> int:
    infos = [dict(info) for info in list(chunk_result.get("infos", ()))]
    if infos:
        executed = 0
        for info in infos:
            if not bool(info.get("controller_action_executed", True)):
                break
            executed += 1
        return int(executed)
    steps = list(chunk_result.get("steps", ()))
    if steps:
        return int(len(steps))
    return int(chunk_result.get("num_steps", 0))


def _chunk_success(chunk_result: dict[str, Any]) -> bool:
    chunk_info = dict(chunk_result.get("info", {}))
    if bool(chunk_info.get("success", False)):
        return True
    for info in list(chunk_result.get("infos", ())):
        if bool(dict(info).get("success", False)):
            return True
    for step in list(chunk_result.get("steps", ())):
        info = dict(dict(step).get("info", {}))
        if bool(info.get("success", False)):
            return True
    return False


class RawRolloutRecorder:
    """Write raw processor payloads as per-episode replayable records."""

    def __init__(
        self,
        *,
        output_root: Path,
        logger: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self._logger = logger
        self._manifest_path = self.output_root / RAW_ROLLOUT_MANIFEST_FILENAME
        self._metadata = {} if metadata is None else dict(metadata)
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
        task_prompt = str(payload.get("task_prompt", ""))
        buffer = self._episode_buffers.get(episode_id)
        if buffer is None:
            buffer = _EpisodeBuffer(
                episode_id=episode_id,
                task_prompt=task_prompt,
                chunks=[],
                seen_chunk_seqs=set(),
                expected_step_start=0,
                episode_return=0.0,
                success=False,
                tainted=False,
                started_at=time.time(),
            )
            self._episode_buffers[episode_id] = buffer

        if buffer.tainted:
            buffer.seen_chunk_seqs.add(chunk_seq)
            return False
        if chunk_seq in buffer.seen_chunk_seqs:
            return False

        try:
            episode_step_start = int(payload["episode_step_start"])
            if episode_step_start != int(buffer.expected_step_start):
                raise ValueError(
                    "raw rollout recorder expected contiguous episode steps: "
                    f"episode_id={episode_id} chunk_seq={chunk_seq} "
                    f"episode_step_start={episode_step_start} "
                    f"expected={int(buffer.expected_step_start)}"
                )
            chunk_result = dict(payload["chunk_result"])
            executed_steps = _executed_steps(chunk_result)
            if executed_steps <= 0 and not bool(
                chunk_result.get("done", False) or chunk_result.get("truncated", False)
            ):
                raise ValueError(
                    "raw rollout recorder received no executed steps for a "
                    "non-terminal chunk"
                )

            chunk_payload = _copy_tree(dict(payload))
            chunk_payload["recorded_at"] = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(),
            )
            chunk_payload["executed_steps"] = int(executed_steps)
            buffer.chunks.append(chunk_payload)
            buffer.seen_chunk_seqs.add(chunk_seq)
            buffer.expected_step_start += int(executed_steps)
            buffer.episode_return += float(chunk_result.get("reward_sum", 0.0))
            buffer.success = bool(buffer.success or _chunk_success(chunk_result))
            return True
        except Exception:
            buffer.tainted = True
            raise

    def finalize_episode(self, *, marker: dict[str, Any]) -> Path | None:
        episode_id = int(marker.get("episode_id", -1))
        buffer = self._episode_buffers.get(episode_id)
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
        if not buffer.chunks:
            self._logger.warning(
                "skip writing empty raw rollout episode: episode_id=%s",
                int(episode_id),
            )
            self._episode_buffers.pop(episode_id, None)
            self._write_manifest()
            return None

        episode_payload = {
            "format_version": RAW_ROLLOUT_FORMAT_VERSION,
            "schema_version": RAW_ROLLOUT_SCHEMA_VERSION,
            "episode_id": int(buffer.episode_id),
            "task_prompt": str(buffer.task_prompt),
            "num_chunks": int(len(buffer.chunks)),
            "num_steps": int(buffer.expected_step_start),
            "episode_return": float(buffer.episode_return),
            "success": bool(buffer.success),
            "started_at": float(buffer.started_at),
            "finished_at": time.time(),
            "final_marker": _copy_tree(dict(marker)),
            "metadata": _copy_tree(self._metadata),
            "chunks": list(buffer.chunks),
        }

        episode_path = self.output_root / f"episode_{int(buffer.episode_id):06d}.pkl"
        episode_name = str(episode_path.name)
        next_episode_files = list(self._episode_files)
        if episode_name not in self._episode_file_set:
            next_episode_files.append(episode_name)
        next_episodes_written = int(self._episodes_written) + 1
        next_steps_written = int(self._steps_written) + int(buffer.expected_step_start)
        try:
            self._atomic_pickle(episode_path, episode_payload)
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
            "schema_version": RAW_ROLLOUT_SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "episode_files": (
                list(self._episode_files)
                if episode_files is None
                else [str(value) for value in episode_files]
            ),
            "metadata": _copy_tree(self._metadata),
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
        self._atomic_json(self._manifest_path, manifest_payload)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    @staticmethod
    def _atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)


__all__ = [
    "RAW_ROLLOUT_FORMAT_VERSION",
    "RAW_ROLLOUT_MANIFEST_FILENAME",
    "RAW_ROLLOUT_SCHEMA_VERSION",
    "RawRolloutRecorder",
]
