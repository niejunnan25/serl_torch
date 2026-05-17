from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock
from unittest.mock import patch


class _ConfigDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value


class _FakeFlags(dict):
    def is_parsed(self) -> bool:
        return True


def test_wandb_logger_falls_back_to_native_wandb_when_swanlab_missing() -> None:
    fake_wandb = types.SimpleNamespace(
        init=MagicMock(return_value=object()),
        config=types.SimpleNamespace(update=MagicMock()),
    )
    fake_ml_collections = types.SimpleNamespace(
        ConfigDict=_ConfigDict,
        config_dict=types.SimpleNamespace(
            FieldReference=lambda value, field_type=None: value
        ),
    )
    fake_absl_flags = types.SimpleNamespace(FLAGS=_FakeFlags())
    fake_absl = types.SimpleNamespace(flags=fake_absl_flags)

    module_name = "serl_launcher.common.wandb"
    previous_module = sys.modules.pop(module_name, None)

    try:
        with patch.dict(
            sys.modules,
            {
                "absl": fake_absl,
                "absl.flags": fake_absl_flags,
                "ml_collections": fake_ml_collections,
                "wandb": fake_wandb,
            },
            clear=False,
        ):
            wandb_module = importlib.import_module(module_name)
            cfg = wandb_module.WandBLogger.get_default_config()
            cfg.project = "libero"
            cfg.entity = None
            cfg.exp_descriptor = "native-wandb"
            cfg.group = "tests"

            wandb_module.WandBLogger(
                wandb_config=cfg,
                variant={},
                wandb_output_dir="/tmp/wandb",
                mode="online",
            )

        fake_wandb.init.assert_called_once()
        assert fake_wandb.init.call_args.kwargs["mode"] == "online"
        fake_wandb.config.update.assert_called_once()
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
