from __future__ import annotations

from .offline_prepared import build_residual_prepared_fingerprint
from .offline_prepared import build_residual_training_signature
from .offline_prepared import extract_residual_manifest_signature
from .offline_prepared import format_residual_alpha_token
from .offline_prepared import load_prepared_offline_replay
from .offline_prepared import OfflinePreparedInputs
from .offline_prepared import OfflinePreparedResolution
from .offline_prepared import RESIDUAL_PREPARED_SIGNATURE_KEYS
from .offline_prepared import resolve_prepared_episode_files
from .offline_prepared import resolve_prepared_path_value
from .offline_prepared import resolve_residual_prepared_dir
from .offline_prepared import validate_prepared_paths

__all__ = [
    "OfflinePreparedInputs",
    "OfflinePreparedResolution",
    "RESIDUAL_PREPARED_SIGNATURE_KEYS",
    "build_residual_prepared_fingerprint",
    "build_residual_training_signature",
    "extract_residual_manifest_signature",
    "format_residual_alpha_token",
    "MemoryEfficientStepWindowReplayBuffer",
    "load_prepared_offline_replay",
    "resolve_prepared_episode_files",
    "resolve_prepared_path_value",
    "resolve_residual_prepared_dir",
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
