"""OpenPI base-policy client for LIBERO observations."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

from .observation import LiberoObservationCache, build_libero_state, extract_residual_images


def encode_obs_for_openpi(
    obs: Dict[str, Any],
    prompt: str,
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, Any]:
    images = extract_residual_images(obs, obs_cache=obs_cache, cache_key=cache_key)
    state = build_libero_state(obs, obs_cache=obs_cache, cache_key=cache_key)
    return {
        "observation/image": images["image"],
        "observation/wrist_image": images["wrist_image"],
        "observation/state": state,
        "prompt": prompt,
    }


def maybe_get_policy_infer_ms(pred: Dict[str, Any]) -> Optional[float]:
    if "policy_timing" in pred and isinstance(pred["policy_timing"], dict):
        ms = pred["policy_timing"].get("infer_ms")
        if ms is not None:
            return float(ms)
    return None


def maybe_get_server_infer_ms(pred: Dict[str, Any]) -> Optional[float]:
    if "server_timing" in pred and isinstance(pred["server_timing"], dict):
        ms = pred["server_timing"].get("infer_ms")
        if ms is not None:
            return float(ms)
    return None


class OpenPIChunkClient:
    def __init__(self, host: str, port: int, logger: Optional[logging.Logger] = None):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port)
        self._logger = logger or logging.getLogger(__name__)

    def infer_chunk(
        self,
        obs: Dict[str, Any],
        prompt: str,
        *,
        obs_cache: Optional[LiberoObservationCache] = None,
        cache_key: Optional[Hashable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        send_data = encode_obs_for_openpi(
            obs,
            prompt,
            obs_cache=obs_cache,
            cache_key=cache_key,
        )
        start = time.time()
        pred = self._client.infer(send_data)
        e2e_ms = (time.time() - start) * 1000.0
        chunk = np.asarray(pred["actions"], dtype=np.float32)
        info = {
            "e2e_ms": float(e2e_ms),
            "policy_ms": maybe_get_policy_infer_ms(pred),
            "server_ms": maybe_get_server_infer_ms(pred),
        }
        return chunk, info
