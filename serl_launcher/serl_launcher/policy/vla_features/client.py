"""VLA feature client for RLT training.

Connects to a VLA feature server (serve_vla_features.py) via websocket,
sends raw observations, and receives pre-computed {z_rl, reference_action, proprio}.

This follows the same pattern as OpenPIPolicyClient in serl_launcher.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class VLAFeatureClient:
    """Client that queries a VLA feature server for z_rl, reference_action, proprio."""

    def __init__(
        self,
        host: str,
        port: int,
        logger: Optional[logging.Logger] = None,
        *,
        connect_timeout_sec: Optional[float] = 60.0,
        ping_interval_sec: Optional[float] = 20.0,
        ping_timeout_sec: Optional[float] = 600.0,
        reconnect_retry_count: int = 3,
        reconnect_retry_backoff_sec: float = 1.0,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._host = str(host)
        self._port = int(port)
        self._connect_timeout_sec = connect_timeout_sec
        self._ping_interval_sec = ping_interval_sec
        self._ping_timeout_sec = ping_timeout_sec
        self._reconnect_retry_count = max(0, int(reconnect_retry_count))
        self._reconnect_retry_backoff_sec = max(0.0, float(reconnect_retry_backoff_sec))
        self._client_lock = threading.Lock()
        self._client = self._make_client()
        self._logger.info("VLAFeatureClient connected to ws://%s:%s", self._host, self._port)

    def _make_client(self) -> WebsocketClientPolicy:
        return WebsocketClientPolicy(
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
                pass
        self._client = self._make_client()

    @staticmethod
    def _as_writable_float32(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).copy()

    def infer(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Send raw observation to VLA feature server, get back features.

        Args:
            raw_obs: Raw environment observation dict (images, state, etc.)

        Returns:
            dict with keys: "z_rl" (2048,), "reference_action" (70,), "proprio" (8,)
        """
        for attempt_idx in range(self._reconnect_retry_count + 1):
            try:
                with self._client_lock:
                    result = self._client.infer(raw_obs)
                return {
                    "z_rl": self._as_writable_float32(result["z_rl"]),
                    "reference_action": self._as_writable_float32(result["reference_action"]),
                    "proprio": self._as_writable_float32(result["proprio"]),
                }
            except Exception as err:
                can_retry = attempt_idx < self._reconnect_retry_count
                if not can_retry:
                    raise
                self._logger.warning(
                    "VLA feature server disconnected (%s). Reconnecting (%s/%s).",
                    err, attempt_idx + 1, self._reconnect_retry_count,
                )
                with self._client_lock:
                    self._reconnect_locked()
                if self._reconnect_retry_backoff_sec > 0.0:
                    time.sleep(self._reconnect_retry_backoff_sec)

        raise RuntimeError("VLA feature client retry loop exited unexpectedly.")

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
