"""Core ``serl_launcher`` package."""

from __future__ import annotations

from pathlib import Path

_OUTER_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if _OUTER_PACKAGE_DIR.is_dir() and str(_OUTER_PACKAGE_DIR) not in __path__:
    __path__.append(str(_OUTER_PACKAGE_DIR))
