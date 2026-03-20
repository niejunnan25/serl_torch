"""Compatibility namespace bridging to legacy ``serl_launcher`` package.

V1 keeps ``serl_launcher`` as the implementation source while allowing new code
paths to import from ``serl_torch.launcher``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def _ensure_legacy_serl_launcher_on_path() -> None:
    """Ensure local legacy ``serl_launcher`` repo path is importable."""
    try:
        importlib.import_module("serl_launcher")
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "serl_launcher"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)


_ensure_legacy_serl_launcher_on_path()
_legacy_launcher: ModuleType = importlib.import_module("serl_launcher")

# Reuse legacy package search paths so imports like
# ``import serl_torch.launcher.data.replay_buffer`` resolve to legacy modules.
__path__: Iterable[str] = tuple(getattr(_legacy_launcher, "__path__", ()))
__all__ = list(getattr(_legacy_launcher, "__all__", ()))


def __getattr__(name: str) -> Any:
    return getattr(_legacy_launcher, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy_launcher)))
