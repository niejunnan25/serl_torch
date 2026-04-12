"""Generic OpenPI policy client."""
from __future__ import annotations

import inspect
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from serl_launcher.policy.base import coerce_action_chunk, PolicyInferInfo, PolicyInput
from serl_launcher.policy.openpi.request_builder import build_openpi_request


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


class OpenPIPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        action_dim: Optional[int] = None,
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
        self._action_dim = None if action_dim is None else int(action_dim)
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
        kwargs = {
            "host": self._host,
            "port": self._port,
            "open_timeout": self._connect_timeout_sec,
            "ping_interval": self._ping_interval_sec,
            "ping_timeout": self._ping_timeout_sec,
        }
        supported = set(inspect.signature(self._client_ctor.__init__).parameters)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in supported and value is not None
        }
        unsupported = sorted(key for key in kwargs if key not in supported)
        if unsupported:
            self._logger.debug(
                "OpenPI client %s does not support kwargs %s; constructing with %s",
                self._client_ctor.__name__,
                unsupported,
                sorted(filtered_kwargs),
            )
        return self._client_ctor(**filtered_kwargs)

    def _reconnect_locked(self) -> None:
        old_client = self._client
        close_fn = getattr(old_client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                self._logger.debug(
                    "Ignored OpenPI client close error during reconnect",
                    exc_info=True,
                )
        self._client = self._make_client()

    def _is_retryable_connection_error(self, err: BaseException) -> bool:
        return isinstance(err, (self._websocket_connection_closed_error, OSError))

    def infer_chunk(self, policy_input: PolicyInput) -> Tuple[np.ndarray, PolicyInferInfo]:
        send_data = build_openpi_request(policy_input)
        for attempt_idx in range(self._reconnect_retry_count + 1):
            try:
                start = time.time()
                with self._client_lock:
                    pred = self._client.infer(send_data)
                e2e_ms = (time.time() - start) * 1000.0
                chunk = coerce_action_chunk(
                    pred["actions"],
                    action_dim=self._action_dim,
                )
                info: PolicyInferInfo = {
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
