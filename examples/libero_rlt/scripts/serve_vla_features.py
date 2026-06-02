#!/usr/bin/env python3
"""Serve frozen Pi0 + frozen RL-token encoder features for LIBERO RLT."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.libero_rlt.vla_inference import (
    extract_rlt_features,
    load_frozen_pi0,
    load_frozen_rlt_encoder,
)
from serl_launcher.policy.vla_features.server import VLAFeatureServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA feature server for LIBERO RLT")
    parser.add_argument("--pi0-config", type=str, default="pi0_libero")
    parser.add_argument("--pi0-path", type=str, required=True)
    parser.add_argument("--rlt-encoder-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--rlt-input-dim", type=int, default=2048)
    parser.add_argument("--z-rl-dim", type=int, default=2048)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading frozen Pi0 + RLT encoder")
    pi0_model, pi0_policy = load_frozen_pi0(args.pi0_config, args.pi0_path, args.device)
    rl_token_encoder = load_frozen_rlt_encoder(
        args.rlt_encoder_path,
        args.device,
        input_dim=args.rlt_input_dim,
        rl_token_dim=args.z_rl_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    def feature_fn(raw_obs):
        return extract_rlt_features(
            pi0_model,
            pi0_policy,
            rl_token_encoder,
            raw_obs,
            device=args.device,
            chunk_size=args.chunk_size,
            action_dim=args.action_dim,
            proprio_dim=args.proprio_dim,
        )

    server = VLAFeatureServer(
        feature_fn,
        host=args.host,
        port=args.port,
        metadata={
            "type": "vla_feature_server",
            "chunk_size": args.chunk_size,
            "action_dim": args.action_dim,
            "proprio_dim": args.proprio_dim,
            "z_rl_dim": args.z_rl_dim,
        },
        logger=logger,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
