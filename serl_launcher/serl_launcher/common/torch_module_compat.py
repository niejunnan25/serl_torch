"""Helpers for state_dict interoperability between compiled and plain modules."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch.nn as nn

_TORCH_COMPILE_PREFIX = "_orig_mod."


def unwrap_torch_compile_module(module: nn.Module) -> nn.Module:
    """Return the original nn.Module when ``torch.compile`` wraps it."""

    unwrapped = module
    while isinstance(getattr(unwrapped, "_orig_mod", None), nn.Module):
        unwrapped = getattr(unwrapped, "_orig_mod")
    return unwrapped


def _strip_torch_compile_prefix(name: str) -> str:
    stripped = name
    while stripped.startswith(_TORCH_COMPILE_PREFIX):
        stripped = stripped[len(_TORCH_COMPILE_PREFIX) :]
    return stripped


def normalize_module_state_dict_keys(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Normalize legacy compiled-module prefixes back to plain module keys."""

    needs_normalization = any(
        isinstance(key, str) and key.startswith(_TORCH_COMPILE_PREFIX)
        for key in state_dict.keys()
    )
    if not needs_normalization:
        return state_dict

    normalized = OrderedDict(
        (
            _strip_torch_compile_prefix(str(key)) if isinstance(key, str) else key,
            value,
        )
        for key, value in state_dict.items()
    )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        normalized._metadata = OrderedDict(
            (
                _strip_torch_compile_prefix(str(key)) if isinstance(key, str) else key,
                value,
            )
            for key, value in metadata.items()
        )
    return normalized


def module_state_dict(module: nn.Module) -> Mapping[str, object]:
    """Return a plain, checkpoint-stable state_dict for a module."""

    return unwrap_torch_compile_module(module).state_dict()


def load_module_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, object],
    *,
    strict: bool = True,
) -> None:
    """Load a state_dict into a plain or compiled module."""

    normalized_state_dict = normalize_module_state_dict_keys(state_dict)
    unwrap_torch_compile_module(module).load_state_dict(
        normalized_state_dict,
        strict=bool(strict),
    )
