from __future__ import annotations

"""Residual-learning helpers shared across environments."""

from .typed_action import ResidualActionSpec

__all__ = [
    "ResidualActionSpec",
    "ResidualActionStatsAccumulator",
    "ResidualDeltaActionFilter",
    "PreparedStepWindowReplayBufferSampler",
    "build_chunk_residual_obs",
    "build_chunk_residual_observation_space",
    "build_chunk_residual_sample_obs",
    "project_expert_action",
    "prepare_base_actions_chunk",
]


def __getattr__(name: str):
    if name == "ResidualActionStatsAccumulator":
        from .action_metrics import ResidualActionStatsAccumulator

        return ResidualActionStatsAccumulator
    if name == "ResidualDeltaActionFilter":
        from .action_filter import ResidualDeltaActionFilter

        return ResidualDeltaActionFilter
    if name == "PreparedStepWindowReplayBufferSampler":
        from .chunk_window_replay import PreparedStepWindowReplayBufferSampler

        return PreparedStepWindowReplayBufferSampler
    if name == "project_expert_action":
        from .expert_projection import project_expert_action

        return project_expert_action
    if name in {
        "build_chunk_residual_obs",
        "build_chunk_residual_observation_space",
        "build_chunk_residual_sample_obs",
        "prepare_base_actions_chunk",
    }:
        from .observation import build_chunk_residual_obs
        from .observation import build_chunk_residual_observation_space
        from .observation import build_chunk_residual_sample_obs
        from .observation import prepare_base_actions_chunk

        mapping = {
            "build_chunk_residual_obs": build_chunk_residual_obs,
            "build_chunk_residual_observation_space": build_chunk_residual_observation_space,
            "build_chunk_residual_sample_obs": build_chunk_residual_sample_obs,
            "prepare_base_actions_chunk": prepare_base_actions_chunk,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
