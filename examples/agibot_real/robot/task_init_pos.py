"""Resolve init_node_pos(task_name) for episode reset.

Priority:
1. If env AGIBOT_CODE_ROOT is set to a directory containing init_node_pos.py (e.g. .../tangyili/code/agibot),
   import and use that module (same as inference_camera_position.py / AgiRobot.reset).
2. Otherwise use the bundled copy in init_node_pos_data (kept in sync with tangyili).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def init_node_pos(task_name: str):
    # Align with init_node_pos_data: default SERL task name uses office_setting pose.
    if str(task_name).strip() == "agibot_real_default":
        task_name = "office_setting"

    root = os.environ.get("AGIBOT_CODE_ROOT", "").strip()
    if root:
        p = Path(root).expanduser().resolve()
        init_py = p / "init_node_pos.py"
        if p.is_dir() and init_py.is_file():
            key = str(p)
            if key not in sys.path:
                sys.path.insert(0, key)
            try:
                from init_node_pos import init_node_pos as _ext  # type: ignore[import]

                return _ext(task_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AGIBOT_CODE_ROOT set but init_node_pos import failed (%s); using bundled table.",
                    exc,
                )
    from .init_node_pos_data import init_node_pos as _builtin

    return _builtin(task_name)
