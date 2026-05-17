from __future__ import annotations

"""Run the standalone AgiBot rollout processor role."""

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

from serl_torch.examples.agibot_real.config import cfg_to_log_payload
from serl_torch.examples.agibot_real.config import parse_train_cfg
from serl_torch.examples.agibot_real.runtime.processor_runtime import run_processor


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_residual",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    if typed_cfg.runtime.role != "processor":
        raise ValueError(
            "run_residual_processor.py requires runtime.role=processor "
            "and processor.mode=standalone"
        )
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_processor")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))
    run_processor(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
