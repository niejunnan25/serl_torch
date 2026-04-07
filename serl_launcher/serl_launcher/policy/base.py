"""Lightweight policy I/O contracts shared across backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Tuple

import numpy as np

PolicyInferInfo = Dict[str, Any]


@dataclass(frozen=True)
class PolicyInput:
    prompt: str
    state: np.ndarray
    images: Mapping[str, np.ndarray]
    image_mask: Mapping[str, bool]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutput:
    action_chunk: np.ndarray
    info: PolicyInferInfo = field(default_factory=dict)


class ChunkPolicyClient(Protocol):
    def infer_chunk(self, policy_input: PolicyInput) -> Tuple[np.ndarray, PolicyInferInfo]:
        """Run a chunked policy and return `(action_chunk, info)`."""
