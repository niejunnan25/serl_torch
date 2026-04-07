"""OpenPI path resolution and import bootstrap helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from serl_launcher.utils.repo_paths import resolve_repo_candidate


def resolve_openpi_root(openpi_root: Optional[str]) -> Path:
    if openpi_root:
        root = Path(openpi_root).expanduser().resolve()
    else:
        root = resolve_repo_candidate("openpi")
    if not root.exists():
        raise FileNotFoundError(f"openpi root not found: {root}")
    return root


def setup_openpi_client_pythonpath(openpi_root: Path) -> Path:
    client_src = openpi_root / "packages" / "openpi-client" / "src"
    if not client_src.exists():
        raise FileNotFoundError(f"openpi client src not found: {client_src}")
    if str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    return client_src
