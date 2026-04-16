from __future__ import annotations

"""torch.compile helpers for AgiBot residual training/eval."""

import logging
from typing import Any

import torch


def maybe_enable_torch_compile(
    agent: Any,
    *,
    compile_cfg: Any,
    logger: logging.Logger,
) -> Any:
    if not bool(compile_cfg.enabled):
        return agent
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "training.torch_compile.enabled=true but torch.compile is unavailable "
            "in this PyTorch build"
        )

    target = str(compile_cfg.target)
    if target not in {"critic", "actor_critic"}:
        raise ValueError(
            "training.torch_compile.target must be one of {'critic', 'actor_critic'}, "
            f"got {target!r}"
        )

    compile_kwargs = {
        "backend": str(compile_cfg.backend),
        "mode": str(compile_cfg.mode),
        "fullgraph": bool(compile_cfg.fullgraph),
        "dynamic": bool(compile_cfg.dynamic),
    }
    logger.info(
        "enable torch.compile: target=%s backend=%s mode=%s fullgraph=%s dynamic=%s",
        target,
        compile_kwargs["backend"],
        compile_kwargs["mode"],
        compile_kwargs["fullgraph"],
        compile_kwargs["dynamic"],
    )

    agent.state.modules["critic"] = torch.compile(
        agent.state.modules["critic"],
        **compile_kwargs,
    )
    if "critic" in agent.state.target_modules:
        agent.state.target_modules["critic"] = torch.compile(
            agent.state.target_modules["critic"],
            **compile_kwargs,
        )
    if target == "actor_critic":
        agent.state.modules["actor"] = torch.compile(
            agent.state.modules["actor"],
            **compile_kwargs,
        )
    return agent


__all__ = ["maybe_enable_torch_compile"]
