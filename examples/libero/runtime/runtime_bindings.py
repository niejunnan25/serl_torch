"""LIBERO-specific runtime bindings for residual actor/learner entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Hashable
from typing import Optional

from omegaconf import DictConfig
from serl_launcher.data.normalizer import StateActionNormalizer
from serl_launcher.data.normalizer import load_normalizer
from serl_launcher.residual.runtime.bindings import ResidualRuntimeBindings
from serl_launcher.residual.runtime.profiling import _RuntimeProfiler
from serl_launcher.residual.runtime.profiling import _profile_call

from ..config import resolve_libero_cfg_image_keys
from ..env_wrappers.factory import _create_env
from ..training_config import LIBERO_RESIDUAL_BASE_CONFIG
from .obs_adapter import LiberoObservationCache
from .obs_adapter import build_residual_step_core
from .obs_adapter import build_residual_step_obs
from .policy_adapter import build_libero_policy_input


@dataclass
class LiberoRuntimeBindings(ResidualRuntimeBindings):
    env: Any
    image_keys: tuple[str, ...]
    normalizer: StateActionNormalizer | None
    obs_cache: LiberoObservationCache
    task_key: str
    data_config: Any

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

    normalizer: StateActionNormalizer | None = None
    norm_cfg = cfg.get("normalization", None)
    task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        stats_dir = norm_cfg.get(
            "stats_dir",
            str(Path(__file__).resolve().parents[1] / "data" / "stats"),
        )
        normalizer = load_normalizer(task_key, stats_dir=stats_dir)
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    return LiberoRuntimeBindings(
        env=env,
        image_keys=tuple(resolve_libero_cfg_image_keys(cfg)),
        normalizer=normalizer,
        obs_cache=LiberoObservationCache(),
        task_key=task_key,
        data_config=LIBERO_RESIDUAL_BASE_CONFIG,
    )
