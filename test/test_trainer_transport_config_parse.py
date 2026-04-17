from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from examples.agibot_real.config import parse_train_cfg as parse_agibot_train_cfg
from examples.libero.config import parse_train_cfg as parse_libero_train_cfg


def test_libero_copy_copy_config_parses_split_queue_defaults() -> None:
    cfg = OmegaConf.load(
        Path("examples/libero/configs/train_residual_copy_copy.yaml")
    )
    typed_cfg = parse_libero_train_cfg(cfg)
    assert typed_cfg.runtime.trainer_transport.mode == "split_queue"
    assert int(typed_cfg.runtime.trainer_transport.data_port) == 5690
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_episode_end is False
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_shutdown is True


def test_agibot_copy_config_parses_split_queue_defaults() -> None:
    cfg = OmegaConf.load(
        Path("examples/agibot_real/configs/train_residual_copy.yaml")
    )
    typed_cfg = parse_agibot_train_cfg(cfg)
    assert typed_cfg.runtime.trainer_transport.mode == "split_queue"
    assert int(typed_cfg.runtime.trainer_transport.data_port) == 5490
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_episode_end is False
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_shutdown is True
