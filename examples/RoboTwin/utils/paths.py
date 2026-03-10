"""serl_torch / serl_launcher 路径查找与导入工具。"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_serl_repo_root() -> Path:
    """通过向上查找 serl_launcher 目录来定位 serl_torch 仓库根目录。"""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "serl_launcher").exists():
            return parent
    raise RuntimeError("Cannot locate serl_torch repo root from current file path")


def ensure_serl_launcher_importable() -> None:
    """确保 serl_launcher 可以被 import（无需安装为包）。"""
    serl_launcher_root = _find_serl_repo_root() / "serl_launcher"
    if str(serl_launcher_root) not in sys.path:
        sys.path.append(str(serl_launcher_root))
