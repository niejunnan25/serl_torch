from __future__ import annotations

"""Thin standalone agentlace learner entrypoint for AgiBot residual SAC."""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from serl_launcher.residual.train.learner.service import run_residual_learner_service
from serl_launcher.training.seeding import set_global_seeds

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.runtime.data_bindings import (
    build_agibot_data_bindings,
)


def _validate_removed_data_injection(cfg: DictConfig) -> None:
    if bool(cfg.get("offline", {}).get("enabled", False)):
        raise ValueError(
            "AgiBot example-local offline data injection has been removed; "
            "set offline.enabled=false"
        )
    if bool(cfg.get("training", {}).get("online_prefill", {}).get("enabled", False)):
        raise ValueError(
            "AgiBot example-local online prefill injection has been removed; "
            "set training.online_prefill.enabled=false"
        )


@hydra.main(
    version_base=None, config_path="../../conf", config_name="train_residual_sac"
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("agibot_real_agentlace_learner")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _validate_removed_data_injection(cfg)
    set_global_seeds(int(cfg.seed))
    bindings = build_agibot_data_bindings(cfg, logger=logger)
    run_residual_learner_service(
        cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
    )


if __name__ == "__main__":
    main()
