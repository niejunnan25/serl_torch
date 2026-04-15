"""Minimal JSONL writer for run logs and evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from serl_launcher.utils.serialization import to_jsonable


def append_jsonl(path: Path | str, payload: Dict[str, Any]) -> None:
    """Append one JSONL record with immediate flush semantics."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
        fp.flush()


class JsonlWriter:
    """Simple JSONL writer with immediate flush."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
