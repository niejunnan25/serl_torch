"""Frozen Pi0 + RLT encoder loading and VLA feature extraction.

This module encapsulates all Pi0/encoder inference logic so it can be
shared between the actor process and the eval process.
"""

from __future__ import annotations

import datetime as _datetime
import logging
import re
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
logger = logging.getLogger(__name__)

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc




from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model
from openpi.training import config as _config
from serl_launcher.agents.rlt.modeling import RLTokenEncoder


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, value) for value in tree)
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return fn(tree)


@torch.no_grad()
def _run_vla(
    pi0_model: Any,
    obs_pt: _model.Observation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen VLA to get z_vla embeddings and reference action chunk.

    Returns:
        z_vla: (1, M, D) VLA token embeddings
        ref_actions: (1, 50, 32) normalized reference actions
    """
    device = obs_pt.state.device if obs_pt.state is not None else torch.device("cuda")

    ref_actions = pi0_model.sample_actions(
        device=device,
        observation=obs_pt,
        noise=None,
        num_steps=10,
    )

    images, img_masks, lang_tokens, lang_masks, state = pi0_model._preprocess_observation(
        obs_pt,
        train=False,
    )
    prefix_embs, _, _ = pi0_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    timestep = torch.zeros(ref_actions.shape[0], dtype=torch.float32, device=device)
    suffix_embs, _, _, _ = pi0_model.embed_suffix(state, ref_actions, timestep)

    if prefix_embs.shape[-1] == suffix_embs.shape[-1]:
        z_vla = torch.cat([prefix_embs, suffix_embs], dim=1)
    else:
        z_vla = prefix_embs

    z_vla = z_vla.to(torch.float32)
    return z_vla, ref_actions


def load_frozen_pi0(
    pi0_config_name: str,
    pi0_checkpoint_path: str,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load frozen Pi0 model and policy transforms.

    Returns:
        (pi0_model, pi0_policy) where pi0_model is the frozen PyTorch model
        and pi0_policy provides _input_transform / _output_transform.
    """
    logger.info("Loading Pi0 policy config %r from %s", pi0_config_name, pi0_checkpoint_path)
    pi0_policy = _policy_config.create_trained_policy(
        _config.get_config(pi0_config_name),
        pi0_checkpoint_path,
    )
    pi0_model = pi0_policy._model
    pi0_model.eval()
    for param in pi0_model.parameters():
        param.requires_grad = False
    logger.info("Frozen Pi0 model loaded on %s", device)
    return pi0_model, pi0_policy


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


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
    encoder.to(device)
    logger.info(
        "Frozen RLT encoder loaded from %s (input_dim=%s rl_token_dim=%s layers=%s)",
        rlt_encoder_path, input_dim, rl_token_dim, num_encoder_layers,
    )
    return encoder


@torch.no_grad()
def extract_rlt_features(
    pi0_model: Any,
    pi0_policy: Any,
    rl_token_encoder: RLTokenEncoder,
    raw_obs: dict[str, Any],
    *,
    device: str = "cuda",
    chunk_size: int = 10,
    action_dim: int = 7,
    proprio_dim: int = 8,
) -> dict[str, np.ndarray]:
    """Run frozen Pi0 + encoder on a raw observation.

    Returns dict with:
        z_rl: (z_rl_dim,) numpy array
        reference_action: (chunk_size * action_dim,) numpy array (unnormalized, flattened)
        proprio: (proprio_dim,) numpy array
    """
    torch_device = torch.device(device)

    # 1. Apply Pi0 input transforms (normalization, padding, prompts)
    processed_obs = pi0_policy._input_transform(raw_obs)

    # 2. Convert to PyTorch tensors with batch dim
    obs_torch = _tree_map(
        lambda x: torch.from_numpy(np.array(x, copy=True)).to(torch_device)[None, ...],
        processed_obs,
    )

    # 3. Cast to official observation format
    obs_pt = _model.Observation.from_dict(obs_torch)

    # 4. Run VLA inference
    z_vla, ref_vla_norm = _run_vla(pi0_model, obs_pt)

    # 5. Compute z_rl via frozen encoder
    z_rl = rl_token_encoder(z_vla)  # (1, z_rl_dim)

    # 6. Extract proprio
    rl_state = obs_torch["state"][:, :proprio_dim]  # (1, proprio_dim)

    # 7. Unnormalize reference actions
    out_dict = {
        "state": obs_torch["state"].detach().cpu()[0],
        "actions": ref_vla_norm.detach().cpu()[0],
    }
    out_dict_unnorm = pi0_policy._output_transform(out_dict)
    ref_vla_unnorm = out_dict_unnorm["actions"]  # (50, action_dim) or similar

    # Take first chunk_size steps, flatten
    ref_action_chunk = ref_vla_unnorm[:chunk_size, :action_dim]  # (chunk_size, action_dim)
    ref_action_flat = ref_action_chunk.reshape(-1).astype(np.float32)  # (chunk_size * action_dim,)

    return {
        "z_rl": z_rl.squeeze(0).cpu().numpy().astype(np.float32),
        "reference_action": ref_action_flat,
        "proprio": rl_state.squeeze(0).cpu().numpy().astype(np.float32),
    }
