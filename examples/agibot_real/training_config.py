"""AgiBot residual-training data bindings."""
from __future__ import annotations

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
