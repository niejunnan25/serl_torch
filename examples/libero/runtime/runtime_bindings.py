"""LIBERO-specific runtime bindings for residual actor/learner entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import Hashable
from typing import Optional

from omegaconf import DictConfig
from serl_launcher.residual.runtime.bindings import ResidualRuntimeBindings
from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.training.profiling import _profile_call

from ..env_wrappers.factory import _create_env
from .data_bindings import LiberoDataBindings
from .data_bindings import build_libero_data_bindings
from .obs_adapter import LiberoObservationCache
from .obs_adapter import build_residual_step_core
from .obs_adapter import build_residual_step_obs
from .policy_adapter import build_libero_policy_input


@dataclass
class LiberoRuntimeBindings(LiberoDataBindings, ResidualRuntimeBindings):
    env: Any
    obs_cache: LiberoObservationCache

    def build_policy_input(
        self,
        obs_raw: dict[str, Any],
        prompt: str,
        *,
        cache_key: Optional[Hashable] = None,
    ) -> Any:
        return build_libero_policy_input(
            obs_raw,
            prompt,
            obs_cache=self.obs_cache,
            cache_key=cache_key,
        )

    def build_step_core(
        self,
        obs_raw: dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return build_residual_step_core(
            obs_raw,
            image_keys=self.image_keys,
            normalizer=self.normalizer,
            obs_cache=self.obs_cache,
            cache_key=cache_key,
        )

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
        **_: Any,
    ) -> dict[str, Any]:
        return build_residual_step_obs(
            obs_raw,
            base_action,
            image_keys=self.image_keys,
            stack_horizon=stack_horizon,
            normalizer=self.normalizer,
            obs_cache=self.obs_cache,
            cache_key=cache_key,
            action_dim=action_dim,
            base_action_chunk=base_action_chunk,
            alpha=alpha,
            state_mode=state_mode,
        )

    def build_step_obs_profiled(
        self,
        profiler: _RuntimeProfiler | None,
        obs_raw: dict[str, Any],
        base_action: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _profile_call(
            profiler,
            "build_residual_step_obs",
            self.build_step_obs,
            obs_raw,
            base_action,
            **kwargs,
        )


def build_libero_runtime_bindings(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> LiberoRuntimeBindings:
    env = _create_env(cfg, logger)
    logger.info(
        "LIBERO task: suite=%s task_id=%s prompt=%s",
        cfg.task.suite_name,
        cfg.task.task_id,
        env.current_instruction,
    )

    data_bindings = build_libero_data_bindings(cfg, logger=logger)
    return LiberoRuntimeBindings(
        env=env,
        image_keys=data_bindings.image_keys,
        normalizer=data_bindings.normalizer,
        obs_cache=LiberoObservationCache(),
        task_key=data_bindings.task_key,
        data_config=data_bindings.data_config,
    )
