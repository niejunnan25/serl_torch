#!/usr/bin/env python3
from __future__ import annotations

"""Run real LIBERO replay sampling/update microbenchmarks without actor processes."""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
REPO_PARENT = REPO_ROOT.parent
for path in (REPO_PARENT, REPO_ROOT, SERL_LAUNCHER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from serl_launcher.agents.continuous.drq_typed_config import (  # noqa: E402
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.residual.chunk_window_replay import (  # noqa: E402
    create_chunk_replay_buffer,
)
from serl_launcher.residual.chunk_window_replay import (  # noqa: E402
    sample_mixed_training_batch,
)
from serl_launcher.residual.observation import (  # noqa: E402
    build_chunk_residual_observation_space,
)
from serl_launcher.residual.observation import build_chunk_residual_sample_obs  # noqa: E402
from serl_launcher.residual.typed_action import ResidualActionSpec  # noqa: E402
from serl_torch.examples.libero.config import parse_train_cfg  # noqa: E402
from serl_torch.examples.libero.env.observation import LIBERO_STATE_DIM  # noqa: E402
from serl_torch.examples.libero.env.observation import (  # noqa: E402
    RESIDUAL_IMAGE_HEIGHT,
)
from serl_torch.examples.libero.env.observation import (  # noqa: E402
    RESIDUAL_IMAGE_WIDTH,
)
from serl_torch.examples.libero.env.offline_data import (  # noqa: E402
    load_prepared_offline_replay,
)
from serl_torch.examples.libero.env.offline_data import (  # noqa: E402
    resolve_and_validate_prepared_paths,
)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def summarize_profiles(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for record in records for key in record})
    return {
        key: summarize([record.get(key, 0.0) for record in records])
        for key in keys
    }


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="spatial_4_0514/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std5p0",
    )
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=10000)
    parser.add_argument("--offline-capacity", type=int, default=10000)
    parser.add_argument(
        "--mode",
        choices=("all", "stage1", "stage2"),
        default="all",
        help="Limit benchmark to one sampling/update mode.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("libero_real_microbench")
    config_dir = str(REPO_ROOT / "examples/libero/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=str(args.config_name),
            overrides=[
                "runtime.role=learner",
                "wandb.mode=disabled",
                "training.torch_compile.enabled=false",
                "training.async_eval.enabled=false",
                f"replay.capacity={int(args.capacity)}",
                f"offline.capacity={int(args.offline_capacity)}",
                "offline.load_max_transitions=null",
                "offline.load_max_episodes=null",
            ],
        )

    typed_cfg = parse_train_cfg(cfg)
    image_keys = tuple(typed_cfg.obs.image_keys)
    sample_obs = build_chunk_residual_sample_obs(
        state_dim=LIBERO_STATE_DIM,
        action_dim=int(typed_cfg.env.action_dim),
        chunk_horizon=int(typed_cfg.residual.chunk_horizon),
        image_keys=image_keys,
        image_height=RESIDUAL_IMAGE_HEIGHT,
        image_width=RESIDUAL_IMAGE_WIDTH,
    )
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
    resolution = resolve_and_validate_prepared_paths(typed_cfg, logger=logger)
    prepared_paths = tuple(resolution.prepared_paths[:1])
    if not prepared_paths:
        raise RuntimeError("no prepared offline replay found")

    def make_replay(capacity: int):
        replay = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=int(typed_cfg.env.action_dim),
            chunk_horizon=int(typed_cfg.residual.chunk_horizon),
            discount=float(typed_cfg.sac.discount),
            image_keys=image_keys,
            capacity=int(capacity),
        )
        load_stats = load_prepared_offline_replay(
            replay_buffer=replay,
            prepared_paths=prepared_paths,
            logger=logger,
            max_episodes=None,
            max_transitions=None,
        )
        return replay, load_stats

    load_start = time.perf_counter()
    online_replay, online_load = make_replay(int(args.capacity))
    offline_replay, offline_load = make_replay(int(args.offline_capacity))
    load_sec = time.perf_counter() - load_start

    residual_action_spec = ResidualActionSpec.from_cfg(
        typed_cfg,
        action_dim=int(typed_cfg.env.action_dim),
    )

    def make_agent():
        return create_drq_agent_from_typed_cfg(
            typed_cfg,
            sample_obs=sample_obs,
            action_dim=residual_action_spec.chunk_policy_action_dim,
            image_keys=image_keys,
            critic_action_dim=residual_action_spec.chunk_critic_action_dim,
            action_transform=residual_action_spec.build_chunk_action_transform(),
        )

    def run_mode(
        name: str,
        *,
        pack_obs_and_next_obs: bool,
        prefer_device_concat: bool,
    ) -> dict[str, Any]:
        agent = make_agent()
        metrics: dict[str, list[float]] = defaultdict(list)
        profile_records: list[dict[str, float]] = []
        start_all = time.perf_counter()
        total_updates = int(args.updates) + int(args.warmup)

        for update_index in range(total_updates):
            measured = update_index >= int(args.warmup)
            iter_start = time.perf_counter()
            merged_profile: dict[str, float] = defaultdict(float)

            profile: dict[str, float] = {}
            sync_cuda()
            section_start = time.perf_counter()
            batch, _ = sample_mixed_training_batch(
                online_replay_buffer=online_replay,
                offline_replay_buffer=offline_replay,
                batch_size=int(typed_cfg.replay.batch_size),
                offline_ratio=float(typed_cfg.offline.ratio),
                profile=profile,
                pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
                device=agent.device,
                prefer_device_concat=bool(prefer_device_concat),
            )
            sync_cuda()
            sample_replay_buffer_sec = time.perf_counter() - section_start
            for key, value in profile.items():
                if isinstance(value, (int, float)):
                    merged_profile[key] += float(value)

            sync_cuda()
            section_start = time.perf_counter()
            agent, _ = agent.update_critics(batch)
            sync_cuda()
            train_critics_sec = time.perf_counter() - section_start

            profile = {}
            sync_cuda()
            section_start = time.perf_counter()
            batch, _ = sample_mixed_training_batch(
                online_replay_buffer=online_replay,
                offline_replay_buffer=offline_replay,
                batch_size=int(typed_cfg.replay.batch_size),
                offline_ratio=float(typed_cfg.offline.ratio),
                profile=profile,
                pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
                device=agent.device,
                prefer_device_concat=bool(prefer_device_concat),
            )
            agent, _ = agent.update_high_utd(
                batch,
                utd_ratio=int(typed_cfg.sac.utd_ratio),
            )
            sync_cuda()
            train_sec = time.perf_counter() - section_start
            for key, value in profile.items():
                if isinstance(value, (int, float)):
                    merged_profile[key] += float(value)

            iter_sec = time.perf_counter() - iter_start
            if measured:
                metrics["sample_replay_buffer"].append(sample_replay_buffer_sec)
                metrics["train_critics"].append(train_critics_sec)
                metrics["train"].append(train_sec)
                metrics["update_total"].append(iter_sec)
                profile_records.append(dict(merged_profile))

        measured_total = max(sum(metrics["update_total"]), 1e-9)
        return {
            "name": name,
            "updates": int(args.updates),
            "warmup": int(args.warmup),
            "pack_obs_and_next_obs": bool(pack_obs_and_next_obs),
            "prefer_device_concat": bool(prefer_device_concat),
            "updates_per_sec": float(int(args.updates) / measured_total),
            "wall_elapsed_sec_including_warmup": float(time.perf_counter() - start_all),
            "timers": {key: summarize(value) for key, value in metrics.items()},
            "sample_profile": summarize_profiles(profile_records),
        }

    modes = []
    if args.mode in ("all", "stage1"):
        modes.append(
            run_mode(
                "stage1_compatible_numpy_concat",
                pack_obs_and_next_obs=False,
                prefer_device_concat=False,
            )
        )
    if args.mode in ("all", "stage2"):
        modes.append(
            run_mode(
                "stage2_packed_device_concat",
                pack_obs_and_next_obs=True,
                prefer_device_concat=True,
            )
        )

    result = {
        "config": {
            "updates": int(args.updates),
            "warmup": int(args.warmup),
            "mode": str(args.mode),
            "batch_size": int(typed_cfg.replay.batch_size),
            "offline_ratio": float(typed_cfg.offline.ratio),
            "critic_actor_ratio": int(typed_cfg.training.critic_actor_ratio),
            "utd_ratio": int(typed_cfg.sac.utd_ratio),
            "image_keys": list(image_keys),
            "stack_horizon": int(typed_cfg.obs.stack_horizon),
            "online_size": int(len(online_replay)),
            "offline_size": int(len(offline_replay)),
            "online_windows": int(online_replay.num_windows),
            "offline_windows": int(offline_replay.num_windows),
            "load_sec": float(load_sec),
            "online_load": online_load,
            "offline_load": offline_load,
        },
        "modes": modes,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
