"""LIBERO runtime adapters for residual training and evaluation."""

from serl_launcher.residual.runtime.profiling import _RuntimeProfiler
from serl_launcher.residual.runtime.profiling import _profile_call

from .obs_adapter import LiberoObservationCache
from .obs_adapter import build_libero_state
from .obs_adapter import build_residual_step_core
from .obs_adapter import build_residual_step_obs
from .obs_adapter import extract_residual_images
from .policy_adapter import build_libero_policy_input
from .runtime_bindings import LiberoRuntimeBindings
from .runtime_bindings import build_libero_runtime_bindings


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


__all__ = [
    "LiberoObservationCache",
    "LiberoRuntimeBindings",
    "build_libero_state",
    "build_libero_runtime_bindings",
    "build_libero_policy_input",
    "build_residual_step_core",
    "build_residual_step_obs",
    "build_residual_step_obs_profiled",
    "extract_residual_images",
]
