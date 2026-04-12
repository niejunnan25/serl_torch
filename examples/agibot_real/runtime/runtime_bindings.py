"""AgiBot-specific runtime bindings for residual actor/learner entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import Hashable
from typing import Optional

from omegaconf import DictConfig
from serl_launcher.residual.train.bindings import ResidualRuntimeBindings
from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.training.profiling import _profile_call

from ..env.factory import _create_env
from .data_bindings import AgiBotDataBindings
from .data_bindings import build_agibot_data_bindings
from .obs_adapter import AgiBotObservationCache
from .obs_adapter import build_residual_step_core
from .obs_adapter import build_residual_step_obs
from .policy_adapter import build_agibot_policy_input


@dataclass
class AgiBotRuntimeBindings(AgiBotDataBindings, ResidualRuntimeBindings):
    env: Any
    obs_cache: AgiBotObservationCache

    def build_policy_input(
        self,
        obs_raw: dict[str, Any],
        prompt: str,
        *,
        cache_key: Optional[Hashable] = None,
    ) -> Any:
        return build_agibot_policy_input(
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
        **_: Any,
    ) -> dict[str, Any]:
        return build_residual_step_obs(
            obs_raw,
            base_action,
            image_keys=self.image_keys,
            stack_horizon=stack_horizon,
            obs_cache=self.obs_cache,
            cache_key=cache_key,
            action_dim=action_dim,
            base_action_chunk=base_action_chunk,
            alpha=alpha,
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


def build_agibot_runtime_bindings(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> AgiBotRuntimeBindings:
    env = _create_env(cfg, logger)
    logger.info(
        "AgiBot task: name=%s prompt=%s",
        cfg.task.name,
        env.current_instruction,
    )

    data_bindings = build_agibot_data_bindings(cfg, logger=logger)
    return AgiBotRuntimeBindings(
        env=env,
        image_keys=data_bindings.image_keys,
        obs_cache=AgiBotObservationCache(),
        task_key=data_bindings.task_key,
        data_config=data_bindings.data_config,
    )
