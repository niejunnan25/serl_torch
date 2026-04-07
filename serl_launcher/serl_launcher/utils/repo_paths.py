"""Helpers for locating the serl_torch repo and sibling checkouts."""
from __future__ import annotations

from pathlib import Path


def find_serl_repo_root() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "serl_launcher").exists() and (parent / "examples").exists():
            return parent
    for parent in this_file.parents:
        if (parent / "serl_launcher").exists():
            return parent
    raise RuntimeError("Cannot locate serl_torch repo root from current file path")


def resolve_repo_candidate(repo_name: str) -> Path:
    repo_root = find_serl_repo_root()
    sibling = (repo_root.parent / repo_name).resolve()
    local = (repo_root / repo_name).resolve()
    if sibling.exists():
        return sibling
    if local.exists():
        return local
    return sibling
