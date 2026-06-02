#!/usr/bin/env python3
"""LIBERO env server for RLT training.

This is a thin re-export of the libero env server.
Usage:
    python scripts/serve_env.py --suite-name libero_10 --task-id 8 --port 30000
"""

import sys
from pathlib import Path

# Ensure the libero example is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for _path in (REPO_ROOT, SERL_LAUNCHER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_PARENT = REPO_ROOT.parent

from examples.libero.scripts.serve_env import main

if __name__ == "__main__":
    main()
