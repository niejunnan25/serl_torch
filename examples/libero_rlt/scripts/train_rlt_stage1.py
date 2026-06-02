#!/usr/bin/env python3
"""Train the RLT Stage 1 encoder/decoder on frozen OpenPI embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for path in (SERL_LAUNCHER_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from serl_launcher.agents.rlt.modeling import RLTokenDecoder, RLTokenEncoder  # noqa: E402
from serl_launcher.policy.vla_backends import OpenPIBackend  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional for smoke runs.
    SummaryWriter = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="RLT Stage 1 encoder/decoder training")
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    resolved = str(value)
    return resolved if resolved else None


def _lr_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _select_reconstruction_target(
    prefix_embeddings: torch.Tensor,
    suffix_embeddings: torch.Tensor,
    *,
    source: str,
    max_tokens: int | None,
) -> tuple[torch.Tensor, str]:
    source = source.lower()
    if source not in {"auto", "prefix", "prefix_suffix"}:
        raise ValueError("rlt.reconstruction_source must be auto, prefix, or prefix_suffix")

    if source == "prefix":
        z_vla = prefix_embeddings
        source_name = "prefix"
    elif source == "prefix_suffix":
        if prefix_embeddings.shape[-1] != suffix_embeddings.shape[-1]:
            raise ValueError(
                "prefix_suffix reconstruction requires matching embedding dims, "
                f"got prefix={prefix_embeddings.shape[-1]} suffix={suffix_embeddings.shape[-1]}"
            )
        z_vla = torch.cat([prefix_embeddings, suffix_embeddings], dim=1)
        source_name = "prefix+suffix"
    elif prefix_embeddings.shape[-1] == suffix_embeddings.shape[-1]:
        z_vla = torch.cat([prefix_embeddings, suffix_embeddings], dim=1)
        source_name = "prefix+suffix"
    else:
        z_vla = prefix_embeddings
        source_name = "prefix"

    if max_tokens is not None and max_tokens > 0:
        z_vla = z_vla[:, :max_tokens, :]
        source_name = f"{source_name}[:{max_tokens}]"
    return z_vla.detach().to(torch.float32), source_name


def _build_modules(cfg: Any, input_dim: int, device: torch.device):
    encoder = RLTokenEncoder(
        input_dim=input_dim,
        rl_token_dim=int(cfg.rlt.rl_token_dim),
        num_layers=int(cfg.rlt.num_encoder_layers),
        num_heads=int(cfg.rlt.num_heads),
        ff_dim=int(cfg.rlt.ff_dim),
        dropout=float(cfg.rlt.dropout),
    ).to(device)
    decoder = RLTokenDecoder(
        rl_token_dim=int(cfg.rlt.rl_token_dim),
        output_dim=input_dim,
        num_layers=int(cfg.rlt.num_decoder_layers),
        num_heads=int(cfg.rlt.num_heads),
        ff_dim=int(cfg.rlt.ff_dim),
        dropout=float(cfg.rlt.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(
            step,
            total_steps=int(cfg.training.steps),
            warmup_steps=int(cfg.training.warmup_steps),
        ),
    )
    return encoder, decoder, optimizer, scheduler


def _checkpoint_payload(cfg: Any, input_dim: int, global_step: int, encoder, decoder, optimizer, scheduler) -> dict[str, Any]:
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_payload, dict):
        raise TypeError("Stage 1 config must serialize to a dict")
    rlt_cfg = dict(cfg_payload.get("rlt", {}))
    rlt_cfg["input_dim"] = int(input_dim)
    cfg_payload["rlt"] = rlt_cfg
    return {
        "global_step": int(global_step),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg_payload,
    }


def main() -> None:
    cfg = parse_args()
    np.random.seed(int(cfg.global_seed))
    torch.manual_seed(int(cfg.global_seed))

    output_dir = Path(str(cfg.training.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    writer = SummaryWriter(log_dir=str(output_dir / "logs")) if SummaryWriter is not None else None
    device = torch.device(str(cfg.vla.device) if torch.cuda.is_available() else "cpu")

    backend = OpenPIBackend(openpi_root=_optional_str(cfg.vla.openpi_root))
    base_policy = backend.load_base_policy(
        config_name=str(cfg.vla.config_name),
        checkpoint_path=str(cfg.vla.checkpoint_path),
        device=device,
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
    )
    dataloader = backend.create_dataloader(
        config_name=str(cfg.vla.config_name),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
        repo_id_override=_optional_str(cfg.vla.get("repo_id_override", None)),
        shuffle=True,
    )

    encoder = decoder = optimizer = scheduler = None
    input_dim = None
    data_iter = iter(dataloader)
    steps = int(cfg.training.steps)
    max_tokens = cfg.rlt.get("max_tokens", None)
    max_tokens = None if max_tokens is None else int(max_tokens)

    logger.info("Starting RLT Stage 1 training for %d steps", steps)
    for global_step in range(steps):
        try:
            observation, actions = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            observation, actions = next(data_iter)

        obs_torch = backend.observation_to_device_dict(observation, device)
        actions = actions.to(device)
        obs_obj = base_policy.to_observation(obs_torch)

        with torch.no_grad():
            prefix_embeddings, suffix_embeddings = base_policy.extract_embeddings(obs_obj, actions=actions)
            z_vla, source_name = _select_reconstruction_target(
                prefix_embeddings,
                suffix_embeddings,
                source=str(cfg.rlt.reconstruction_source),
                max_tokens=max_tokens,
            )

        if encoder is None:
            input_dim = int(z_vla.shape[-1])
            encoder, decoder, optimizer, scheduler = _build_modules(cfg, input_dim, device)
            logger.info(
                "Initialized RLT Stage 1 modules: input_dim=%d rl_token_dim=%d source=%s",
                input_dim,
                int(cfg.rlt.rl_token_dim),
                source_name,
            )

        z_rl = encoder(z_vla)
        z_recon = decoder(z_rl, z_vla)
        loss = F.mse_loss(z_recon, z_vla)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            float(cfg.training.clip_grad_norm),
        )
        optimizer.step()
        scheduler.step()

        if writer is not None:
            writer.add_scalar("loss/reconstruction", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), global_step)

        if global_step % int(cfg.training.log_every) == 0:
            logger.info(
                "step=%d/%d loss=%.6f lr=%.6g source=%s",
                global_step,
                steps,
                float(loss.item()),
                float(scheduler.get_last_lr()[0]),
                source_name,
            )

        step_number = global_step + 1
        if step_number % int(cfg.training.save_every) == 0:
            checkpoint_path = output_dir / f"checkpoint_{step_number}.pt"
            torch.save(
                _checkpoint_payload(cfg, input_dim, step_number, encoder, decoder, optimizer, scheduler),
                checkpoint_path,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)

    if encoder is None or decoder is None or optimizer is None or scheduler is None or input_dim is None:
        raise RuntimeError("Stage 1 did not run any optimization steps")

    final_path = output_dir / "final_model.pt"
    torch.save(_checkpoint_payload(cfg, input_dim, steps, encoder, decoder, optimizer, scheduler), final_path)
    torch.save(encoder.state_dict(), output_dir / "rlt_encoder.pt")
    if writer is not None:
        writer.close()
    logger.info("Stage 1 completed. Saved final checkpoint to %s", final_path)


if __name__ == "__main__":
    main()
