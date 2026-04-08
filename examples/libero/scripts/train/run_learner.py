from __future__ import annotations

"""Thin standalone agentlace learner entrypoint for LIBERO residual SAC."""

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

from serl_torch.examples.libero.runtime.data_bindings import build_libero_data_bindings


@hydra.main(version_base=None, config_path="../../conf", config_name="train_residual_sac")
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
    bindings = build_libero_data_bindings(cfg, logger=logger)
    run_residual_learner_service(
        cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
    )


if __name__ == "__main__":
    main()
