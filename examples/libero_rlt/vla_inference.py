"""Frozen OpenPI VLA + RLT encoder loading and feature extraction.

This module is shared by the actor and eval feature servers. OpenPI remains an
external provider; serl_torch only wraps it through a backend adapter.
"""

from __future__ import annotations

import datetime as _datetime
import logging
import re
from typing import Any

import numpy as np
import torch

from serl_launcher.agents.rlt.modeling import RLTokenEncoder
from serl_launcher.policy.vla_backends import OpenPIBackend, OpenPIBasePolicy

logger = logging.getLogger(__name__)

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc


def load_frozen_vla(
    vla_config_name: str,
    vla_checkpoint_path: str,
    device: str = "cuda",
    *,
    openpi_root: str | None = None,
) -> OpenPIBasePolicy:
    """Load a frozen OpenPI VLA through the common backend adapter."""
    backend = OpenPIBackend(openpi_root=openpi_root)
    return backend.load_base_policy(
        config_name=vla_config_name,
        checkpoint_path=vla_checkpoint_path,
        device=device,
    )


def load_frozen_pi0(
    pi0_config_name: str,
    pi0_checkpoint_path: str,
    device: str = "cuda",
    *,
    openpi_root: str | None = None,
) -> OpenPIBasePolicy:
    """Backward-compatible alias for existing Stage 2 scripts."""
    return load_frozen_vla(
        pi0_config_name,
        pi0_checkpoint_path,
        device,
        openpi_root=openpi_root,
    )


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _normalize_max_tokens(value: Any) -> int | None:
    if value is None:
        return None
    max_tokens = int(value)
    if max_tokens <= 0:
        return None
    return max_tokens


def _truncate_vla_tokens_for_encoder(
    z_vla: torch.Tensor,
    rl_token_encoder: RLTokenEncoder,
) -> torch.Tensor:
    max_tokens = getattr(rl_token_encoder, "max_tokens", None)
    if max_tokens is None:
        return z_vla
    return z_vla[:, :max_tokens, :]


def load_frozen_rlt_encoder(
    rlt_encoder_path: str,
    device: str = "cuda",
    *,
    input_dim: int = 2048,
    rl_token_dim: int = 2048,
    num_encoder_layers: int = 4,
    num_heads: int = 8,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    max_tokens: int | None = None,
) -> RLTokenEncoder:
    """Load the Stage 1 trained RLT encoder (frozen for Stage 2).

    Checkpoints produced by the Stage 1 script may include a ``config`` field.
    When present, it takes precedence over CLI defaults so the feature server
    matches the encoder architecture used during offline reconstruction.
    """
    checkpoint = torch.load(rlt_encoder_path, map_location="cpu")
    rlt_cfg: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get("config", {})
        rlt_cfg = cfg.get("rlt", {}) if isinstance(cfg, dict) else {}
        state_dict = checkpoint.get("encoder_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if "e_rl" in state_dict:
        input_dim = int(state_dict["e_rl"].shape[-1])

    in_proj_key = next(
        (key for key in state_dict if key.endswith("self_attn.in_proj_weight")),
        None,
    )
    if in_proj_key is not None:
        rl_token_dim = int(state_dict[in_proj_key].shape[1])

    ff_key = next((key for key in state_dict if key.endswith("linear1.weight")), None)
    if ff_key is not None:
        ff_dim = int(state_dict[ff_key].shape[0])

    layer_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [re.search(r"transformer\.layers\.(\d+)\.", key)]
            if match is not None
        }
    )
    if layer_ids:
        num_encoder_layers = max(layer_ids) + 1

    input_dim = int(rlt_cfg.get("input_dim", input_dim))
    rl_token_dim = int(rlt_cfg.get("rl_token_dim", rl_token_dim))
    num_encoder_layers = int(rlt_cfg.get("num_encoder_layers", num_encoder_layers))
    num_heads = int(rlt_cfg.get("num_heads", num_heads))
    ff_dim = int(rlt_cfg.get("ff_dim", ff_dim))
    dropout = float(rlt_cfg.get("dropout", dropout))
    if max_tokens is not None:
        max_tokens = _normalize_max_tokens(max_tokens)
    else:
        max_tokens = _normalize_max_tokens(rlt_cfg.get("max_tokens", None))

    encoder = RLTokenEncoder(
        input_dim=input_dim,
        rl_token_dim=rl_token_dim,
        num_layers=num_encoder_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.max_tokens = max_tokens
    encoder.to(device)
    logger.info(
        "Frozen RLT encoder loaded from %s (input_dim=%s rl_token_dim=%s layers=%s max_tokens=%s)",
        rlt_encoder_path, input_dim, rl_token_dim, num_encoder_layers, max_tokens,
    )
    return encoder


@torch.no_grad()
def extract_rlt_features(
    base_policy: OpenPIBasePolicy,
    rl_token_encoder: RLTokenEncoder,
    raw_obs: dict[str, Any],
    *,
    device: str = "cuda",
    chunk_size: int = 10,
    action_dim: int = 7,
    proprio_dim: int = 8,
) -> dict[str, np.ndarray]:
    """Run frozen OpenPI VLA + frozen RLT encoder on a raw observation.

    Returns:
        z_rl: (z_rl_dim,) float32 array
        reference_action: (chunk_size * action_dim,) unnormalized action chunk
        proprio: (proprio_dim,) float32 array
    """
    feature_batch = base_policy.infer_features(raw_obs)
    z_vla = feature_batch.z_vla.to(torch.device(device))
    z_vla = _truncate_vla_tokens_for_encoder(z_vla, rl_token_encoder)
    z_rl = rl_token_encoder(z_vla)
    rl_state = feature_batch.obs_torch["state"][:, :proprio_dim]
    ref_vla_unnorm = base_policy.unnormalize_actions(
        feature_batch.obs_torch,
        feature_batch.reference_actions,
    )

    ref_action_chunk = ref_vla_unnorm[:chunk_size, :action_dim]
    ref_action_flat = ref_action_chunk.reshape(-1).astype(np.float32)

    return {
        "z_rl": z_rl.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "reference_action": ref_action_flat,
        "proprio": rl_state.squeeze(0).detach().cpu().numpy().astype(np.float32),
    }
