"""AgiBot residual-training data bindings and runtime defaults."""
from __future__ import annotations

import os

from omegaconf import DictConfig
from omegaconf import open_dict
from serl_launcher.residual.data.config import ResidualDataConfig
from serl_launcher.residual.data.config import register_residual_data_config
from serl_launcher.residual.data.schema import LIBERO_RESIDUAL_TRAINING_SCHEMA

from .schema import AGIBOT_IMAGE_SLOT_KEYS

AGIBOT_TRAINING_IMAGE_VIEWS = {
    slot_key: slot_key for slot_key in AGIBOT_IMAGE_SLOT_KEYS
}


AGIBOT_RESIDUAL_BASE_CONFIG = register_residual_data_config(
    ResidualDataConfig(
        name="agibot_residual_training",
        schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
        image_views=AGIBOT_TRAINING_IMAGE_VIEWS,
    )
)


def coerce_agibot_agentlace_async_cfg(cfg: DictConfig) -> None:
    """Force the example-local training runtime onto external Agentlace."""

    default_bootstrap = os.environ.get(
        "AGIBOT_AGENTLACE_BOOTSTRAP", "/tmp/agibot_agentlace_bootstrap.pkl"
    )
    with open_dict(cfg):
        if cfg.get("training", None) is None:
            cfg["training"] = {}
        training_cfg = cfg["training"]
        if training_cfg.get("async", None) is None:
            training_cfg["async"] = {}
        async_cfg = training_cfg["async"]
        async_cfg["enabled"] = True
        async_cfg["backend"] = "agentlace"
        if async_cfg.get("agentlace", None) is None:
            async_cfg["agentlace"] = {}
        agentlace_cfg = async_cfg["agentlace"]
        agentlace_cfg["spawn_local_worker"] = False
        bootstrap_file = str(agentlace_cfg.get("bootstrap_file", "")).strip()
        if not bootstrap_file:
            agentlace_cfg["bootstrap_file"] = default_bootstrap
