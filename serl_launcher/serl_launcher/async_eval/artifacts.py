from __future__ import annotations

"""Generic async-eval checkpoint and artifact naming helpers."""

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from serl_launcher.utils.jsonl import append_jsonl

ASYNC_EVAL_CHECKPOINT_INDEX_FILE = "async_eval_checkpoint_index.jsonl"


def format_async_eval_checkpoint_filename(*, episode_id: int) -> str:
    """Format the checkpoint filename for an eval queued at an episode milestone."""

    return f"episode_{int(episode_id):06d}.pt"


def format_async_eval_run_dir_name(
    *,
    eval_index: int,
    checkpoint_step: int,
    train_episode_id: int | None = None,
) -> str:
    """Format the per-request eval output directory name."""

    if train_episode_id is None:
        return f"eval_{int(eval_index):06d}_step_{int(checkpoint_step):09d}"
    return (
        f"eval_{int(eval_index):06d}_episode_{int(train_episode_id):06d}"
        f"_step_{int(checkpoint_step):09d}"
    )


def save_async_eval_checkpoint_payload(
    checkpoint_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    episode_id: int,
) -> Path:
    """Save an async-eval checkpoint payload using the standard episode-based name."""

    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir_path / format_async_eval_checkpoint_filename(
        episode_id=int(episode_id)
    )
    torch.save(dict(payload), checkpoint_path)
    return checkpoint_path


def append_async_eval_checkpoint_index(
    checkpoint_dir: str | Path,
    *,
    episode_id: int,
    checkpoint_step: int,
    checkpoint_path: str | Path,
) -> None:
    """Append one checkpoint-index record for async-eval directory resolution."""

    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    index_path = checkpoint_dir_path / ASYNC_EVAL_CHECKPOINT_INDEX_FILE
    record = {
        "train_episode_id": int(episode_id),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_path": str(Path(checkpoint_path).name),
    }
    append_jsonl(index_path, record)


def resolve_async_eval_checkpoint_from_index(
    checkpoint_dir: str | Path,
    *,
    checkpoint_step: int | None,
) -> Path | None:
    """Resolve an async-eval checkpoint directory input through its index file."""

    checkpoint_dir_path = Path(checkpoint_dir)
    index_path = checkpoint_dir_path / ASYNC_EVAL_CHECKPOINT_INDEX_FILE
    if not index_path.exists():
        return None

    matches: list[tuple[int, int, Path]] = []
    with index_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            checkpoint_step_raw = payload.get("checkpoint_step", None)
            checkpoint_path_raw = payload.get("checkpoint_path", None)
            if checkpoint_step_raw is None or checkpoint_path_raw is None:
                continue
            try:
                record_checkpoint_step = int(checkpoint_step_raw)
            except Exception:
                continue

            checkpoint_file = Path(str(checkpoint_path_raw)).expanduser()
            if not checkpoint_file.is_absolute():
                checkpoint_file = checkpoint_dir_path / checkpoint_file
            checkpoint_file = checkpoint_file.resolve()
            if not checkpoint_file.exists():
                continue

            train_episode_id_raw = payload.get("train_episode_id", None)
            try:
                train_episode_id = (
                    -1 if train_episode_id_raw is None else int(train_episode_id_raw)
                )
            except Exception:
                train_episode_id = -1

            if checkpoint_step is None or record_checkpoint_step == int(checkpoint_step):
                matches.append(
                    (record_checkpoint_step, train_episode_id, checkpoint_file)
                )

    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1], item[2].name))[2]
