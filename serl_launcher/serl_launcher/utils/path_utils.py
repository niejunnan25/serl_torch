from __future__ import annotations

"""Small path helpers shared across launcher modules."""

from pathlib import Path


def resolve_original_cwd() -> Path:
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd()).resolve()
    except Exception:  # noqa: BLE001
        return Path.cwd().resolve()


def resolve_path(raw_path: str, *, base: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


__all__ = [
    "resolve_original_cwd",
    "resolve_path",
]
