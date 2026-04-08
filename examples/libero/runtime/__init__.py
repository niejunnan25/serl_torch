"""LIBERO runtime adapters for residual training and evaluation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from serl_launcher.residual.runtime.profiling import _RuntimeProfiler
from serl_launcher.residual.runtime.profiling import _profile_call

from .obs_adapter import LiberoObservationCache
from .obs_adapter import build_libero_state
from .obs_adapter import build_residual_step_core
from .obs_adapter import build_residual_step_obs
from .obs_adapter import extract_residual_images
from .policy_adapter import build_libero_policy_input

if TYPE_CHECKING:
    from .runtime_bindings import LiberoDataBindings
    from .runtime_bindings import LiberoRuntimeBindings


def build_residual_step_obs_profiled(
    profiler: _RuntimeProfiler | None,
    *args,
    **kwargs,
):
    return _profile_call(
        profiler,
        "build_residual_step_obs",
        build_residual_step_obs,
        *args,
        **kwargs,
    )


def __getattr__(name: str):
    if name in {
        "LiberoDataBindings",
        "LiberoRuntimeBindings",
        "build_libero_data_bindings",
        "build_libero_runtime_bindings",
    }:
        from . import runtime_bindings as _runtime_bindings

        return getattr(_runtime_bindings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LiberoDataBindings",
    "LiberoObservationCache",
    "LiberoRuntimeBindings",
    "build_libero_state",
    "build_libero_data_bindings",
    "build_libero_runtime_bindings",
    "build_libero_policy_input",
    "build_residual_step_core",
    "build_residual_step_obs",
    "build_residual_step_obs_profiled",
    "extract_residual_images",
]
