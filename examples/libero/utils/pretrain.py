"""Pretraining helpers for residual SAC training."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from omegaconf import DictConfig

from .tb_metrics import _log_info_scalars

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer
    from torch.utils.tensorboard import SummaryWriter


def _pretrain_critic_with_calql(
    cfg: DictConfig,
    *,
    agent,
    offline_buffer: Optional["ReplayBuffer"],
    logger: logging.Logger,
    tb_writer: Optional["SummaryWriter"] = None,
) -> Dict[str, Any]:
    calql_cfg = cfg.training.get("calql_pretrain", None)
    if calql_cfg is None or (not bool(calql_cfg.get("enabled", False))):
        return {"enabled": 0, "steps": 0}

    warm_steps = int(calql_cfg.get("steps", 0))
    warm_batch_size = int(calql_cfg.get("batch_size", cfg.replay.batch_size))
    calql_alpha = float(calql_cfg.get("alpha", 0.0))
    calql_n_actions = int(calql_cfg.get("n_actions", cfg.sac.get("cql_n_actions", 10)))
    calql_temperature = float(calql_cfg.get("temperature", cfg.sac.get("cql_temperature", 1.0)))
    if warm_steps <= 0 or calql_alpha <= 0.0 or offline_buffer is None or len(offline_buffer) == 0:
        return {
            "enabled": 0,
            "steps": 0,
            "requested_steps": int(warm_steps),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
        }

    info_last: Dict[str, Any] = {}
    progress = range(warm_steps)
    if tqdm is not None:
        progress = tqdm(progress, desc="Cal-QL critic pretrain", unit="step", dynamic_ncols=True)

    for step in progress:
        batch = offline_buffer.sample(batch_size=warm_batch_size)
        agent, info_last = agent.update_critics_calql(
            batch,
            calql_alpha=calql_alpha,
            calql_n_actions=calql_n_actions,
            calql_temperature=calql_temperature,
        )
        if tqdm is not None and (step % 50 == 0 or step == warm_steps - 1):
            loss_str = f"loss={info_last.get('critic_loss', 0):.3f}"
            if "predicted_qs" in info_last:
                loss_str += f" Q={info_last['predicted_qs']:.2f}"
            progress.set_postfix_str(loss_str)
        if tb_writer is not None:
            _log_info_scalars(
                tb_writer,
                info_last,
                step,
                (
                    ("calql_pretrain/critic_loss", "critic_loss"),
                    ("calql_pretrain/critic_td_loss", "critic_td_loss"),
                    ("calql_pretrain/critic_cql_penalty", "critic_cql_penalty"),
                    ("calql_pretrain/predicted_qs", "predicted_qs"),
                    ("calql_pretrain/target_qs", "target_qs"),
                    ("calql_pretrain/predicted_q_min", "predicted_q_min"),
                    ("calql_pretrain/predicted_q_max", "predicted_q_max"),
                    ("calql_pretrain/predicted_q_std", "predicted_q_std"),
                    ("calql_pretrain/predicted_q_gap", "predicted_q_gap"),
                    ("calql_pretrain/critic_lr", "critic_lr"),
                ),
            )

    logger.info(
        (
            "Cal-QL critic pretrain done: steps=%s batch_size=%s offline_buffer=%s "
            "alpha=%.4f n_actions=%s temp=%.4f"
        ),
        warm_steps,
        warm_batch_size,
        len(offline_buffer),
        calql_alpha,
        calql_n_actions,
        calql_temperature,
    )
    return {
        "enabled": 1,
        "steps": int(warm_steps),
        "batch_size": int(warm_batch_size),
        "alpha": float(calql_alpha),
        "n_actions": int(calql_n_actions),
        "temperature": float(calql_temperature),
        "last_info": info_last,
    }
