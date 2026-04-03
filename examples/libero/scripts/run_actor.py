from __future__ import annotations

"""Standalone actor entrypoint that forwards to train_residual_sac.py."""

import os
import sys
from pathlib import Path


TRAIN_SCRIPT = Path(__file__).resolve().with_name("train_residual_sac.py")


if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(TRAIN_SCRIPT), *sys.argv[1:]])
