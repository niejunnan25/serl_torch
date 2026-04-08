from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from serl_launcher.residual.train.actor.loop import run_actor_loop
from serl_launcher.residual.train.actor.setup import build_actor_runtime_session
from serl_launcher.residual.train.bindings import ResidualRuntimeBindings


def run_residual_actor_loop(
    cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    bindings: ResidualRuntimeBindings,
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
