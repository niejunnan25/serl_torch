"""Path helpers for locating serl_torch, openpi, and serl_launcher."""
from __future__ import annotations

import sys
from pathlib import Path


def _find_serl_repo_root() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "serl_launcher").exists():
            return parent
    raise RuntimeError("Cannot locate serl_torch repo root from current file path")


def ensure_serl_launcher_importable() -> None:
    serl_launcher_root = _find_serl_repo_root() / "serl_launcher"
    if str(serl_launcher_root) not in sys.path:
        sys.path.append(str(serl_launcher_root))


def resolve_repo_candidate(repo_name: str) -> Path:
    repo_root = _find_serl_repo_root()
    sibling = (repo_root.parent / repo_name).resolve()
    local = (repo_root / repo_name).resolve()
    if sibling.exists():
        return sibling
    if local.exists():
        return local
    return sibling

