from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.agibot_real.config import parse_train_cfg as parse_agibot_train_cfg
from examples.libero.config import parse_train_cfg as parse_libero_train_cfg


def test_libero_optimized_config_parses_async_commit_defaults() -> None:
    cfg = OmegaConf.load(
        Path("examples/libero/configs/train_residual_optimized.yaml")
    )
    typed_cfg = parse_libero_train_cfg(cfg)
    assert typed_cfg.runtime.trainer_transport.mode == "async_commit"
    assert int(typed_cfg.runtime.trainer_transport.data_port) == 5690
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_episode_end is False
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_shutdown is True


def test_agibot_copy_config_parses_async_commit_defaults() -> None:
    cfg = OmegaConf.load(
        Path("examples/agibot_real/configs/train_residual_copy.yaml")
    )
    typed_cfg = parse_agibot_train_cfg(cfg)
    assert typed_cfg.runtime.trainer_transport.mode == "async_commit"
    assert int(typed_cfg.runtime.trainer_transport.data_port) == 5490
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_episode_end is False
    assert typed_cfg.runtime.trainer_transport.wait_committed_on_shutdown is True


def test_legacy_transport_modes_are_rejected() -> None:
    libero_cfg = OmegaConf.load(Path("examples/libero/configs/train_residual.yaml"))
    libero_cfg.runtime.trainer_transport.mode = "legacy_reqrep"
    with pytest.raises(ValueError, match="runtime.trainer_transport.mode"):
        parse_libero_train_cfg(libero_cfg)

    agibot_cfg = OmegaConf.load(
        Path("examples/agibot_real/configs/train_residual_copy.yaml")
    )
    agibot_cfg.runtime.trainer_transport.mode = "split_queue"
    with pytest.raises(ValueError, match="runtime.trainer_transport.mode"):
        parse_agibot_train_cfg(agibot_cfg)
