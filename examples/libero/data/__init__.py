"""Data helpers for LIBERO examples."""
from pathlib import Path

from serl_launcher.data.normalizer import (
    StateActionNormalizer,
    load_normalizer as _load_normalizer,
)


def load_normalizer(
    task_key: str,
    stats_dir: str | Path | None = None,
) -> StateActionNormalizer | None:
    if stats_dir is None:
        stats_dir = Path(__file__).resolve().parent / "stats"
    return _load_normalizer(task_key, stats_dir=stats_dir)

__all__ = ["StateActionNormalizer", "load_normalizer"]
