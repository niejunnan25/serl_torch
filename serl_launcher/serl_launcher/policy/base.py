"""Lightweight policy I/O contracts shared across backends."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

PolicyInferInfo = Dict[str, Any]
PolicyInferResult = Tuple[np.ndarray, PolicyInferInfo]
PolicyBatchInferResult = Tuple[list[np.ndarray], PolicyInferInfo]


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


class PolicyClient(Protocol):
    def infer(self, policy_input: PolicyInput) -> PolicyInferResult:
        """Run the policy and return `(action_chunk, info)`."""


class BatchPolicyClient(Protocol):
    def infer_many(
        self,
        policy_inputs: Sequence[PolicyInput],
    ) -> PolicyBatchInferResult:
        """Run a batched policy request and return per-sample action chunks."""


class PolicyPrefetcher(Protocol):
    def submit(self, policy_input: PolicyInput) -> Future[PolicyInferResult]:
        """Submit a chunked inference request."""

    def close(self) -> None:
        """Release prefetch resources."""


def coerce_action_chunk(
    actions: Any,
    *,
    action_dim: Optional[int] = None,
) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    elif chunk.ndim == 1:
        if action_dim is None:
            chunk = chunk.reshape(1, -1)
        else:
            if chunk.size % int(action_dim) != 0:
                raise ValueError(
                    "Flat action payload must be divisible by action_dim, got "
                    f"shape={chunk.shape}, action_dim={int(action_dim)}"
                )
            chunk = chunk.reshape(-1, int(action_dim))
    if chunk.ndim != 2:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    if chunk.shape[0] < 1:
        raise ValueError("Policy returned empty action chunk")
    if action_dim is not None:
        if chunk.shape[1] < int(action_dim):
            raise ValueError(
                f"Policy action dim {chunk.shape[1]} is smaller than required env action dim "
                f"{int(action_dim)}"
            )
        if chunk.shape[1] > int(action_dim):
            chunk = chunk[:, : int(action_dim)]
    return np.asarray(chunk, dtype=np.float32)
