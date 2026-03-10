#!/usr/bin/env python3
"""PyTorch DRQ entry for cable routing.

This task-specific script reuses the shared async DRQ loop.
"""

from absl import app

from scripts.async_drq import main


if __name__ == "__main__":
    app.run(main)
