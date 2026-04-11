from __future__ import annotations

"""Thin AgiBot residual actor entrypoint."""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from serl_launcher.residual.train.actor.runtime import run_residual_actor_loop
from serl_launcher.training.seeding import set_global_seeds

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.runtime.runtime_bindings import build_agibot_runtime_bindings
from serl_torch.examples.agibot_real.runtime.controller_actor import (
    run_agibot_controller_actor_loop,
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


@hydra.main(version_base=None, config_path="../../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("agibot_real_actor")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _validate_removed_data_injection(cfg)
    set_global_seeds(int(cfg.seed))
    bindings = build_agibot_runtime_bindings(cfg, logger=logger)
    async_eval_watcher_path = Path(__file__).resolve().parents[1] / "eval" / "process_eval_queue.py"
    if bool(cfg.get("controller", {}).get("enabled", False)):
        run_agibot_controller_actor_loop(
            cfg,
            run_dir=run_dir,
            logger=logger,
            bindings=bindings,
            async_eval_watcher_path=async_eval_watcher_path,
        )
    else:
        run_residual_actor_loop(
            cfg,
            run_dir=run_dir,
            logger=logger,
            bindings=bindings,
            async_eval_watcher_path=async_eval_watcher_path,
        )


if __name__ == "__main__":
    main()
