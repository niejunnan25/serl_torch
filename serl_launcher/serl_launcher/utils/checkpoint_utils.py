from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload


def _checkpoint_path(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"checkpoint_{int(step)}.pt"


def _extract_step(path: str | Path) -> int:
    stem = Path(path).stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return -1


def checkpoint_dir_size_bytes(checkpoint_dir: str | Path) -> int:
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        return 0
    total = 0
    for path in checkpoint_dir_path.glob("checkpoint_*.pt"):
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    return int(total)


def latest_checkpoint_step(checkpoint_dir: str | Path) -> Optional[int]:
    checkpoint_dir_path = Path(checkpoint_dir)
    paths = list(checkpoint_dir_path.glob("checkpoint_*.pt"))
    if not paths:
        return None
    return max(_extract_step(p) for p in paths)


def resolve_checkpoint_path(
    checkpoint_dir_or_path: str | Path,
    *,
    step: Optional[int] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_dir_or_path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if step is None:
        step = latest_checkpoint_step(checkpoint_path)
        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    return _checkpoint_path(checkpoint_path, int(step))


def load_checkpoint_payload(
    checkpoint_dir_or_path: str | Path,
    *,
    step: Optional[int] = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir_or_path, step=step)
    return torch.load(checkpoint_path, map_location=map_location)


def write_checkpoint_payload(
    checkpoint_dir: str | Path,
    payload: dict[str, Any],
    *,
    step: int,
    keep: int,
) -> Path:
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(checkpoint_dir_path, step)
    torch.save(payload, checkpoint_path)

    if keep is not None and keep > 0:
        paths = sorted(
            checkpoint_dir_path.glob("checkpoint_*.pt"),
            key=_extract_step,
        )
        for p in paths[:-keep]:
            try:
                p.unlink()
            except OSError:
                pass
    return checkpoint_path


def save_agent_checkpoint(
    checkpoint_dir: str | Path,
    agent: Any,
    step: int,
    keep: int = 20,
) -> Path:
    payload = snapshot_agent_checkpoint_payload(agent, step=int(step))
    return write_checkpoint_payload(
        checkpoint_dir,
        payload,
        step=int(step),
        keep=int(keep),
    )


def load_agent_checkpoint(
    checkpoint_dir_or_path: str | Path,
    agent: Any,
    step: Optional[int] = None,
):
    payload = load_checkpoint_payload(checkpoint_dir_or_path, step=step)
    apply_checkpoint_payload_to_agent(agent, payload, load_optimizers=True)
    return agent
