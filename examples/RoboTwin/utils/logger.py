"""JSONL 日志工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlLogger:
    """
    按行追加 JSON 的日志器：每行一个 JSON 对象，用于记录 step 或 episode 级日志。
    写入后立即 flush，便于实时查看或断点后不丢数据。
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "w", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        """将 payload 序列化为一行 JSON 写入文件并 flush。"""
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        """关闭文件句柄。"""
        self._fp.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
