"""OpenPI base-policy client for LIBERO observations."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

from .observation import (
    LiberoObservationCache,
    build_libero_state,
    extract_residual_images,
)


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
    def __init__(
        self,
        host: str,
        port: int,
        logger: Optional[logging.Logger] = None,
        *,
        connect_timeout_sec: Optional[float] = 30.0,
        ping_interval_sec: Optional[float] = 20.0,
        ping_timeout_sec: Optional[float] = 120.0,
        reconnect_retry_count: int = 1,
        reconnect_retry_backoff_sec: float = 0.05,
    ):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from websockets import exceptions as websocket_exceptions

        self._logger = logger or logging.getLogger(__name__)
        self._host = str(host)
        self._port = int(port)
        self._connect_timeout_sec = connect_timeout_sec
        self._ping_interval_sec = ping_interval_sec
        self._ping_timeout_sec = ping_timeout_sec
        self._reconnect_retry_count = max(0, int(reconnect_retry_count))
        self._reconnect_retry_backoff_sec = max(0.0, float(reconnect_retry_backoff_sec))
        self._websocket_connection_closed_error = websocket_exceptions.ConnectionClosed
        self._client_lock = threading.Lock()
        self._client_ctor = WebsocketClientPolicy
        self._client = self._make_client()

    def _make_client(self):
        return self._client_ctor(
            host=self._host,
            port=self._port,
            open_timeout=self._connect_timeout_sec,
            ping_interval=self._ping_interval_sec,
            ping_timeout=self._ping_timeout_sec,
        )

    def _reconnect_locked(self) -> None:
        old_client = self._client
        close_fn = getattr(old_client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                self._logger.debug(
                    "Ignored OpenPI client close error during reconnect", exc_info=True
                )
        self._client = self._make_client()

    def _is_retryable_connection_error(self, err: BaseException) -> bool:
        return isinstance(err, (self._websocket_connection_closed_error, OSError))

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
        for attempt_idx in range(self._reconnect_retry_count + 1):
            try:
                start = time.time()
                with self._client_lock:
                    pred = self._client.infer(send_data)
                e2e_ms = (time.time() - start) * 1000.0
                chunk = np.asarray(pred["actions"], dtype=np.float32)
                info = {
                    "e2e_ms": float(e2e_ms),
                    "policy_ms": maybe_get_policy_infer_ms(pred),
                    "server_ms": maybe_get_server_infer_ms(pred),
                }
                return chunk, info
            except Exception as err:
                can_retry = (
                    self._is_retryable_connection_error(err)
                    and attempt_idx < self._reconnect_retry_count
                )
                if not can_retry:
                    raise
                self._logger.warning(
                    "OpenPI websocket disconnected (%s). Reconnecting and retrying (%s/%s).",
                    err,
                    attempt_idx + 1,
                    self._reconnect_retry_count,
                )
                with self._client_lock:
                    self._reconnect_locked()
                if self._reconnect_retry_backoff_sec > 0.0:
                    time.sleep(self._reconnect_retry_backoff_sec)

        raise RuntimeError("OpenPI infer retry loop exited unexpectedly.")
