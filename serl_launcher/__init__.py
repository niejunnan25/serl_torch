"""Repository-root import shim for the bundled ``serl_launcher`` package.

The actual Python package lives under ``serl_launcher/serl_launcher``.  This
shim keeps imports working when commands are launched from the repository root
without an editable install or an explicit ``PYTHONPATH=serl_launcher``.
"""

from __future__ import annotations

from pathlib import Path

_INNER_PACKAGE_DIR = Path(__file__).resolve().parent / "serl_launcher"
if _INNER_PACKAGE_DIR.is_dir():
    __path__.insert(0, str(_INNER_PACKAGE_DIR))
