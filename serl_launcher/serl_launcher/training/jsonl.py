"""Minimal JSONL writer used by training and evaluation entrypoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlWriter:
    """Append-only JSONL writer with immediate flush."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
