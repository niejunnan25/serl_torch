from __future__ import annotations

from .offline_prepared import load_prepared_offline_replay
from .offline_prepared import OfflinePreparedInputs
from .offline_prepared import OfflinePreparedResolution
from .offline_prepared import resolve_prepared_episode_files
from .offline_prepared import resolve_prepared_path_value
from .offline_prepared import validate_prepared_paths

__all__ = [
    "OfflinePreparedInputs",
    "OfflinePreparedResolution",
    "MemoryEfficientStepWindowReplayBuffer",
    "load_prepared_offline_replay",
    "resolve_prepared_episode_files",
    "resolve_prepared_path_value",
    "StepWindowReplayBuffer",
    "validate_prepared_paths",
]


def __getattr__(name: str):
    if name in {"MemoryEfficientStepWindowReplayBuffer", "StepWindowReplayBuffer"}:
        from .step_window_replay_buffer import MemoryEfficientStepWindowReplayBuffer
        from .step_window_replay_buffer import StepWindowReplayBuffer

        mapping = {
            "MemoryEfficientStepWindowReplayBuffer": MemoryEfficientStepWindowReplayBuffer,
            "StepWindowReplayBuffer": StepWindowReplayBuffer,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
