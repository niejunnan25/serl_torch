"""Adapter interfaces for environment-specific integrations."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class EnvAdapter(Protocol):
    """Unified environment interaction protocol for training/evaluation loops."""

    current_instruction: str
    step_limit: int
    last_seed: Optional[int]

    def reset(self, *, seed: int, episode_id: int) -> Dict[str, Any]:
        """Reset environment and return raw observation."""

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Apply action and return (obs, reward, done, truncated, info)."""

    def close(self, *, clear_cache: bool = False) -> None:
        """Release environment resources."""


@runtime_checkable
class OpenPIAdapter(Protocol):
    """Protocol for converting observations into OpenPI request payloads."""

    def encode(self, obs: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Encode environment observation into OpenPI server input payload."""


@runtime_checkable
class OfflineAdapter(Protocol):
    """Protocol for mapping offline files into unified transition payloads."""

    def iter_transitions(self, payload: Dict[str, Any]):
        """Yield replay-ready transition dicts from raw payload."""
