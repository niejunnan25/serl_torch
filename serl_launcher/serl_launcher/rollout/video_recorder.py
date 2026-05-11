from __future__ import annotations

"""Asynchronous rollout video recording helpers."""

import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from queue import Full
from queue import Queue
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AsyncVideoRecorderConfig:
    camera_key: str
    fps: float
    output_dir: Path
    max_pending_frames: int
    drop_frames_when_busy: bool


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rollout video recording requires OpenCV; install opencv-python "
            "or set video.enabled=false."
        ) from exc
    return cv2


class AsyncImageVideoRecorder:
    """Write per-episode rollout videos on a background thread."""

    def __init__(
        self,
        *,
        config: AsyncVideoRecorderConfig,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._logger = logger
        self._cv2 = _load_cv2()
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue: Queue[tuple[str, Any]] = Queue(
            maxsize=int(config.max_pending_frames)
        )
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_main,
            name="rollout-video-writer",
            daemon=True,
        )
        self._worker.start()
        self._active_episode_id: int | None = None
        self._dropped_frames_by_episode: dict[int, int] = {}
        self._lock = threading.Lock()

    def start_episode(self, episode_id: int) -> None:
        self._put_control(("start", int(episode_id)))
        with self._lock:
            self._active_episode_id = int(episode_id)
            self._dropped_frames_by_episode.setdefault(int(episode_id), 0)

    def add_obs_frame(self, obs: dict[str, Any]) -> None:
        frame = obs.get(self._config.camera_key, None)
        if frame is None:
            return
        frame_array = np.asarray(frame, dtype=np.uint8)
        if frame_array.ndim != 3:
            return
        copied = np.ascontiguousarray(frame_array[..., :3]).copy()
        try:
            self._queue.put_nowait(("frame", copied))
        except Full:
            if not self._config.drop_frames_when_busy:
                self._put_control(("frame", copied))
                return
            with self._lock:
                if self._active_episode_id is not None:
                    self._dropped_frames_by_episode[self._active_episode_id] = (
                        self._dropped_frames_by_episode.get(
                            self._active_episode_id, 0
                        )
                        + 1
                    )

    def end_episode(
        self,
        *,
        episode_id: int,
        success: bool,
        episode_steps: int,
    ) -> None:
        with self._lock:
            dropped = int(self._dropped_frames_by_episode.pop(int(episode_id), 0))
            if self._active_episode_id == int(episode_id):
                self._active_episode_id = None
        self._put_control(
            (
                "end",
                {
                    "episode_id": int(episode_id),
                    "success": bool(success),
                    "episode_steps": int(episode_steps),
                    "dropped_frames": int(dropped),
                },
            )
        )

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._put_control(("stop", None))
        self._worker.join(timeout=10.0)

    def _put_control(self, item: tuple[str, Any]) -> None:
        while True:
            try:
                self._queue.put(item, timeout=0.5)
                return
            except Full:
                if self._stop_event.is_set():
                    return

    def _worker_main(self) -> None:
        cv2 = self._cv2
        writer = None
        writer_path: Path | None = None
        try:
            while True:
                try:
                    event, payload = self._queue.get(timeout=0.5)
                except Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                if event == "stop":
                    break

                if event == "start":
                    if writer is not None:
                        writer.release()
                        writer = None
                    current_episode_id = int(payload)
                    writer_path = (
                        self._config.output_dir
                        / f"episode_{int(current_episode_id):05d}_{self._config.camera_key.replace('/', '_')}.mp4"
                    )
                    writer = cv2.VideoWriter(
                        str(writer_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(self._config.fps),
                        (640, 480),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {writer_path}")
                    continue

                if event == "frame":
                    if writer is None:
                        continue
                    frame = np.asarray(payload, dtype=np.uint8)
                    if frame.ndim != 3 or frame.shape[-1] < 3:
                        continue
                    if frame.shape[2] > 3:
                        frame = frame[:, :, :3]
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(
                            frame,
                            (640, 480),
                            interpolation=cv2.INTER_AREA,
                        )
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    continue

                if event == "end":
                    metadata = dict(payload)
                    if writer is not None:
                        writer.release()
                        writer = None
                    if writer_path is not None:
                        self._logger.info(
                            "Saved rollout video: episode=%s success=%s steps=%s dropped_frames=%s path=%s",
                            int(metadata.get("episode_id", -1)),
                            bool(metadata.get("success", False)),
                            int(metadata.get("episode_steps", 0)),
                            int(metadata.get("dropped_frames", 0)),
                            writer_path,
                        )
                    writer_path = None
                    continue
        finally:
            if writer is not None:
                writer.release()
