from __future__ import annotations

"""Evaluate a LIBERO residual checkpoint."""

import logging
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.config import parse_eval_cfg
from serl_torch.examples.libero.eval_runner import run_eval


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="eval_residual",
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    typed_cfg = parse_eval_cfg(cfg)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_eval")
    summary = run_eval(
        typed_cfg,
        run_dir=run_dir,
        logger=logger,
        original_cwd=Path(get_original_cwd()).resolve(),
    )
    logger.info("evaluation done: %s", summary)


if __name__ == "__main__":
    main()
