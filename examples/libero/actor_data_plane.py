from __future__ import annotations

"""Compatibility shim for the renamed LIBERO rollout data processor."""

from .rollout_data_processor import RolloutDataProcessor

LiberoActorDataPlane = RolloutDataProcessor

__all__ = ["LiberoActorDataPlane", "RolloutDataProcessor"]
