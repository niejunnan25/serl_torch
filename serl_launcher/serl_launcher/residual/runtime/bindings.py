"""Shared binding protocols for residual runtime entrypoints."""
from __future__ import annotations

from typing import Any
from typing import Hashable
from typing import Optional
from typing import Protocol

from serl_launcher.data.normalizer import StateActionNormalizer
from serl_launcher.policy.base import PolicyInput
from serl_launcher.residual.runtime.profiling import _RuntimeProfiler


class ResidualDataBindings(Protocol):
    """Dataset- and task-facing binding contract used by runtime services."""

    image_keys: tuple[str, ...]
    normalizer: StateActionNormalizer | None
    task_key: str
    data_config: Any


class ResidualRuntimeBindings(ResidualDataBindings, Protocol):
    """Environment-specific runtime contract consumed by residual actor/learner code."""

    env: Any
    obs_cache: Any

    def build_policy_input(
        self,
        obs_raw: dict[str, Any],
        prompt: str,
        *,
        cache_key: Optional[Hashable] = None,
    ) -> PolicyInput:
        """Build a canonical chunk-policy input from an environment observation."""

    def build_step_core(
        self,
        obs_raw: dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the canonical residual observation core for the current step."""

    def build_step_obs(
        self,
        obs_raw: dict[str, Any],
        base_action: Any,
        *,
        stack_horizon: int = 1,
        cache_key: Optional[Hashable] = None,
        action_dim: Optional[int] = None,
        base_action_chunk: Any = None,
        alpha: Optional[float] = None,
        state_mode: str = "fused",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the actor-side residual observation for the current decision step."""

    def build_step_obs_profiled(
        self,
        profiler: _RuntimeProfiler | None,
        obs_raw: dict[str, Any],
        base_action: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Profiled variant of `build_step_obs(...)` used by runtime hot paths."""
