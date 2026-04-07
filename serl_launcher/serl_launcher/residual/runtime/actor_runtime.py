from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from serl_launcher.residual.runtime.actor_loop import run_actor_loop
from serl_launcher.residual.runtime.actor_setup import build_actor_runtime_session


def run_residual_actor_loop(
    cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    bindings: Any,
    async_eval_watcher_path: Path,
) -> None:
    ctx, state = build_actor_runtime_session(
        cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
        async_eval_watcher_path=async_eval_watcher_path,
    )
    run_actor_loop(ctx, state)
