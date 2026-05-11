from __future__ import annotations

"""Prepare temporary AgiBot residual offline data artifacts."""

from dataclasses import replace
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.data.offline_prepared import EPISODE_FILE_GLOB
from serl_launcher.data.offline_prepared import MANIFEST_FILENAME
from serl_launcher.data.offline_prepared import OfflinePreparedInputs
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.seeding import set_global_seeds
from serl_torch.examples.agibot_real.config import cfg_to_log_payload
from serl_torch.examples.agibot_real.config import parse_train_cfg
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.env.offline_data import (
    load_reference_raw_episode_steps,
)
from serl_torch.examples.agibot_real.env.offline_data import prepare_fingerprint
from serl_torch.examples.agibot_real.env.offline_data import (
    prepare_reference_episode_transitions,
)
from serl_torch.examples.agibot_real.env.offline_data import prepared_dir_for_cfg
from serl_torch.examples.agibot_real.env.offline_data import REFERENCE_NOTE
from serl_torch.examples.agibot_real.env.offline_data import (
    resolve_configured_prepared_paths,
)
from serl_torch.examples.agibot_real.env.offline_data import (
    resolve_reference_raw_episode_files,
)
from serl_torch.examples.agibot_real.env.offline_data import (
    resolve_reference_source_format,
)
from serl_torch.examples.agibot_real.env.offline_data import resolve_task_spec
from serl_torch.examples.agibot_real.env.offline_data import write_manifest


def prepare_offline_data(
    cfg,
    *,
    logger: logging.Logger,
) -> OfflinePreparedInputs:
    prepared_paths = resolve_configured_prepared_paths(cfg)
    if prepared_paths:
        return OfflinePreparedInputs(
            prepared_paths=prepared_paths,
            prepare_stats=None,
            manifest_paths=tuple(
                path / MANIFEST_FILENAME if path.is_dir() else path
                for path in prepared_paths
                if path.name == MANIFEST_FILENAME or path.is_dir()
            ),
        )

    task_spec = resolve_task_spec(cfg)
    prepared_dir = prepared_dir_for_cfg(cfg, task_spec=task_spec)
    manifest_path = prepared_dir / MANIFEST_FILENAME
    fingerprint = prepare_fingerprint(cfg, task_spec=task_spec)
    raw_episode_files = resolve_reference_raw_episode_files(task_spec.dataset_path)
    max_prepare_episodes = cfg.offline.prepare.max_episodes
    if max_prepare_episodes is not None:
        raw_episode_files = raw_episode_files[: int(max_prepare_episodes)]

    logger.warning(
        "Preparing AgiBot offline data with a temporary reference-only pipeline. "
        "This mirrors the LIBERO workflow shape but not the final AgiBot data source."
    )

    prepared_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in prepared_dir.glob(EPISODE_FILE_GLOB):
        stale_path.unlink()

    action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=cfg.env.action_dim)
    image_keys = cfg.obs.image_keys
    base_policy = build_agibot_base_policy(cfg, logger=logger)
    reference_source_format = resolve_reference_source_format(task_spec.dataset_path)
    prepare_stats: dict[str, object] = {
        "reference_only": True,
        "reference_source_format": reference_source_format,
        "reference_note": REFERENCE_NOTE,
        "raw_dataset_path": str(task_spec.dataset_path),
        "prepared_dir": str(prepared_dir),
        "max_episodes": max_prepare_episodes,
        "episodes_total": 0,
        "steps_total": 0,
        "steps_unrepresentable": 0,
        "steps_filtered": 0,
        "episodes_written": 0,
        "steps_written": 0,
        "elapsed_sec": 0.0,
    }
    start_time = time.time()
    episode_files: list[Path] = []

    logger.info(
        "Preparing AgiBot offline dataset (temporary reference mode): task=%s raw=%s output=%s",
        task_spec.task_key,
        task_spec.dataset_path,
        prepared_dir,
    )

    try:
        episode_iter = tqdm(
            enumerate(raw_episode_files),
            total=len(raw_episode_files),
            desc=f"prepare {task_spec.task_key}",
            unit="episode",
            dynamic_ncols=True,
            leave=True,
        )
        for episode_index, episode_source_path in episode_iter:
            raw_steps = load_reference_raw_episode_steps(episode_source_path)
            transitions, episode_stats = prepare_reference_episode_transitions(
                raw_steps=raw_steps,
                episode_id=int(episode_index),
                task_prompt=task_spec.task_description,
                action_spec=action_spec,
                image_keys=image_keys,
                base_policy=base_policy,
                expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
                clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
                filter_unrepresentable_steps=bool(
                    cfg.offline.prepare.filter_unrepresentable_steps
                ),
                source_path=episode_source_path,
            )
            prepare_stats["episodes_total"] = int(prepare_stats["episodes_total"]) + 1
            prepare_stats["steps_total"] = int(prepare_stats["steps_total"]) + int(
                episode_stats["steps_total"]
            )
            prepare_stats["steps_unrepresentable"] = int(
                prepare_stats["steps_unrepresentable"]
            ) + int(episode_stats["steps_unrepresentable"])
            prepare_stats["steps_filtered"] = int(
                prepare_stats["steps_filtered"]
            ) + int(episode_stats["steps_filtered"])
            prepare_stats["steps_written"] = int(prepare_stats["steps_written"]) + int(
                episode_stats["steps_written"]
            )
            if transitions:
                episode_path = prepared_dir / f"episode_{int(episode_index):06d}.pkl"
                with open(episode_path, "wb") as fp:
                    pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)
                episode_files.append(episode_path)
                prepare_stats["episodes_written"] = int(
                    prepare_stats["episodes_written"]
                ) + 1
    finally:
        base_policy_close = getattr(base_policy, "close", None)
        if callable(base_policy_close):
            try:
                base_policy_close()
            except Exception:  # noqa: BLE001
                pass

    prepare_stats["elapsed_sec"] = float(time.time() - start_time)
    write_manifest(
        manifest_path=manifest_path,
        task_spec=task_spec,
        cfg=cfg,
        fingerprint=fingerprint,
        episode_files=episode_files,
        prepare_stats=prepare_stats,
    )
    logger.info(
        "AgiBot offline prepare complete: episodes_total=%s steps_total=%s "
        "steps_unrepresentable=%s steps_filtered=%s episodes_written=%s "
        "steps_written=%s manifest=%s",
        int(prepare_stats["episodes_total"]),
        int(prepare_stats["steps_total"]),
        int(prepare_stats["steps_unrepresentable"]),
        int(prepare_stats["steps_filtered"]),
        int(prepare_stats["episodes_written"]),
        int(prepare_stats["steps_written"]),
        manifest_path,
    )
    return OfflinePreparedInputs(
        prepared_paths=(prepared_dir,),
        prepare_stats=prepare_stats,
        manifest_paths=(manifest_path,),
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_residual",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_prepare_offline")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))
    logger.warning(
        "AgiBot offline prepare is a temporary LIBERO-inspired reference implementation. "
        "Replace examples/agibot_real/env/offline_data.py after the real AgiBot data source is defined."
    )

    if not typed_cfg.offline.enabled:
        raise ValueError("run_residual_offline_prepare.py requires offline.enabled=true")
    if typed_cfg.offline.prepared_path is not None:
        logger.info(
            "ignoring offline.prepared_path during prepare: %s",
            typed_cfg.offline.prepared_path,
        )
        typed_cfg = replace(
            typed_cfg,
            offline=replace(typed_cfg.offline, prepared_path=None),
        )

    set_global_seeds(typed_cfg.global_seed)
    offline_inputs = prepare_offline_data(typed_cfg, logger=logger)
    summary = {
        "role": "run_residual_offline_prepare",
        "mode": "residual",
        "reference_only": True,
        "prepared_path": (
            None
            if not offline_inputs.prepared_paths
            else str(offline_inputs.prepared_paths[0])
        ),
        "manifest_path": (
            None
            if not offline_inputs.manifest_paths
            else str(offline_inputs.manifest_paths[0])
        ),
        "prepare_stats": (
            None if offline_inputs.prepare_stats is None else offline_inputs.prepare_stats
        ),
    }
    with open(run_dir / typed_cfg.logging.summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    logger.info("offline prepare done: %s", json.dumps(summary, ensure_ascii=False))
    if offline_inputs.prepared_paths:
        prepared_path = str(offline_inputs.prepared_paths[0])
        logger.info(
            "next learner command: python examples/agibot_real/scripts/run_residual_training.py "
            "runtime.role=learner offline.enabled=true "
            "offline.pretrain_steps=1000 offline.ratio=0.5 "
            "offline.prepared_path=%s",
            prepared_path,
        )


if __name__ == "__main__":
    main()
