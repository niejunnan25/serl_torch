from __future__ import annotations

"""Thin standalone agentlace learner entrypoint for LIBERO residual SAC."""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from serl_launcher.residual.runtime.config_utils import set_global_seeds
from serl_launcher.residual.runtime.learner_service import run_residual_learner_service

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.config import resolve_libero_cfg_image_keys
from serl_torch.examples.libero.training_config import LIBERO_RESIDUAL_BASE_CONFIG


@hydra.main(version_base=None, config_path="../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_agentlace_learner")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seeds(int(cfg.seed))
    run_residual_learner_service(
        cfg,
        run_dir=run_dir,
        logger=logger,
        data_config=LIBERO_RESIDUAL_BASE_CONFIG,
        resolve_cfg_image_keys=resolve_libero_cfg_image_keys,
    )


if __name__ == "__main__":
    main()
