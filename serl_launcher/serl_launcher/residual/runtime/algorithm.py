"""Residual algorithm interface and factory for runtime entrypoints."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class ResidualAlgorithm(Protocol):
    """Minimal algorithm surface consumed by residual actor/learner runtime."""

    name: str

    def build_actor_agent(
        self,
        cfg: Any,
        *,
        sample_obs: Dict[str, Any],
        action_dim: int,
        image_keys: tuple[str, ...],
        critic_action_dim: Optional[int] = None,
        action_transform: Optional[Dict[str, Any]] = None,
        device: Any = None,
    ) -> Any:
        ...

    def build_learner_agent(
        self,
        cfg: Any,
        *,
        sample_obs: Dict[str, Any],
        action_dim: int,
        image_keys: tuple[str, ...],
        critic_action_dim: Optional[int] = None,
        action_transform: Optional[Dict[str, Any]] = None,
        device: Any = None,
    ) -> Any:
        ...

    def sample_actions(
        self,
        agent: Any,
        obs_input: Dict[str, Any],
        *,
        deterministic: bool = False,
    ) -> Any:
        ...

    def update_high_utd(
        self,
        agent: Any,
        batch: Dict[str, Any],
        *,
        utd_ratio: int,
    ) -> tuple[Any, Dict[str, Any]]:
        ...

    def sync_modules(self, target_agent: Any, source_agent: Any) -> None:
        ...

    def apply_snapshot_payload(
        self,
        target_agent: Any,
        payload: Dict[str, Any],
        *,
        load_optimizers: bool = False,
    ) -> None:
        ...

    def snapshot_checkpoint_payload(
        self,
        agent: Any,
        *,
        step: int,
    ) -> Dict[str, Any]:
        ...


def _resolve_algorithm_type(cfg: Any | None) -> str:
    if cfg is None:
        return "sac"
    residual_cfg = cfg.get("residual", None) if hasattr(cfg, "get") else None
    algorithm_cfg = (
        residual_cfg.get("algorithm", None)
        if residual_cfg is not None and hasattr(residual_cfg, "get")
        else None
    )
    if algorithm_cfg is None:
        return "sac"
    return str(algorithm_cfg.get("type", "sac")).strip().lower()


def build_residual_algorithm(cfg: Any | None = None) -> ResidualAlgorithm:
    algorithm_type = _resolve_algorithm_type(cfg)
    if algorithm_type == "sac":
        from serl_launcher.residual.runtime.sac_algorithm import ResidualSACAlgorithm

        return ResidualSACAlgorithm()
    raise ValueError(f"Unsupported residual.algorithm.type: {algorithm_type}")
