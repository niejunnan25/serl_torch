from __future__ import annotations

"""Prepare LIBERO residual offline data artifacts."""

from dataclasses import replace
import json
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.utils.seeding import set_global_seeds
from serl_torch.examples.libero.config import cfg_to_log_payload
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.offline_data import prepare_current_task_offline_data


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
    logger = logging.getLogger("libero_prepare_offline")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    if not typed_cfg.offline.enabled:
        raise ValueError("prepare_offline_data.py requires offline.enabled=true")
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
    offline_inputs = prepare_current_task_offline_data(typed_cfg, logger=logger)
    summary = {
        "role": "prepare_offline_data",
        "mode": "residual",
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
            None
            if offline_inputs.prepare_stats is None
            else offline_inputs.prepare_stats
        ),
    }
    with open(run_dir / typed_cfg.logging.summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    logger.info("offline prepare done: %s", json.dumps(summary, ensure_ascii=False))
    if offline_inputs.prepared_paths:
        prepared_path = str(offline_inputs.prepared_paths[0])
        logger.info(
            "next learner command: python examples/libero/scripts/run_residual_training.py "
            "runtime.role=learner offline.enabled=true "
            "offline.pretrain_steps=1000 offline.ratio=0.5 "
            "offline.prepared_path=%s",
            prepared_path,
        )


if __name__ == "__main__":
    main()
