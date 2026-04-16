"""JoyRA websocket policy client."""
from __future__ import annotations

import inspect
import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import websockets.sync.client
from websockets import exceptions as websocket_exceptions

from serl_launcher.policy.base import (
    coerce_action_chunk,
    PolicyBatchInferResult,
    PolicyInferInfo,
    PolicyInput,
)
from serl_launcher.policy.joyra import msgpack_numpy
from serl_launcher.policy.joyra.request_builder import (
    build_joyra_batch_request,
    build_joyra_request,
)


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


class JoyRAPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        action_dim: int,
        logger: Optional[logging.Logger] = None,
        api_key: Optional[str] = None,
        connect_timeout_sec: Optional[float] = 30.0,
        ping_interval_sec: Optional[float] = 20.0,
        ping_timeout_sec: Optional[float] = 120.0,
        close_timeout_sec: Optional[float] = 10.0,
        reconnect_retry_count: int = 1,
        reconnect_retry_backoff_sec: float = 0.05,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        if int(action_dim) <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self._action_dim = int(action_dim)
        self._host = str(host)
        self._port = int(port)
        self._api_key = api_key
        self._connect_timeout_sec = connect_timeout_sec
        self._ping_interval_sec = ping_interval_sec
        self._ping_timeout_sec = ping_timeout_sec
        self._close_timeout_sec = close_timeout_sec
        self._reconnect_retry_count = max(0, int(reconnect_retry_count))
        self._reconnect_retry_backoff_sec = max(0.0, float(reconnect_retry_backoff_sec))
        self._packer = msgpack_numpy.Packer()
        self._client_lock = threading.Lock()
        self._server_metadata: Dict[str, Any] = {}
        self._ws = None
        self._connect()

    def _uri(self) -> str:
        if self._host.startswith("ws://") or self._host.startswith("wss://"):
            base = self._host.rstrip("/")
        else:
            base = f"ws://{self._host}"
        return f"{base}:{self._port}"

    def _connect_kwargs(self) -> Dict[str, Any]:
        connect_fn = websockets.sync.client.connect
        connect_sig = inspect.signature(connect_fn)
        supported_params = set(connect_sig.parameters.keys())
        headers_key = (
            "additional_headers"
            if "additional_headers" in supported_params
            else "extra_headers"
        )
        kwargs = {
            "compression": None,
            "max_size": None,
            headers_key: {"Authorization": f"Api-Key {self._api_key}"}
            if self._api_key
            else None,
            "open_timeout": self._connect_timeout_sec,
            "ping_interval": self._ping_interval_sec,
            "ping_timeout": self._ping_timeout_sec,
            "close_timeout": self._close_timeout_sec,
            "proxy": None,
        }
        return {
            key: value
            for key, value in kwargs.items()
            if (key in supported_params) and (value is not None)
        }

    def _wait_for_server(self):
        uri = self._uri()
        connect_kwargs = self._connect_kwargs()
        self._logger.info("Waiting for JoyRA server at %s", uri)
        while True:
            try:
                conn = websockets.sync.client.connect(uri, **connect_kwargs)
                metadata = msgpack_numpy.unpackb(conn.recv())
                if not isinstance(metadata, dict):
                    raise RuntimeError(
                        f"Expected JoyRA server metadata dict, got {type(metadata).__name__}"
                    )
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                self._logger.info("Still waiting for JoyRA server...")
                time.sleep(5.0)

    def _connect(self) -> None:
        self._ws, self._server_metadata = self._wait_for_server()

    def _reconnect_locked(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                self._logger.debug(
                    "Ignored JoyRA websocket close error during reconnect",
                    exc_info=True,
                )
        self._connect()

    def _is_retryable_connection_error(self, err: BaseException) -> bool:
        return isinstance(err, (websocket_exceptions.ConnectionClosed, OSError))

    def get_server_metadata(self) -> Dict[str, Any]:
        return dict(self._server_metadata)

    def close(self) -> None:
        with self._client_lock:
            if self._ws is not None:
                self._ws.close()
                self._ws = None

    def _send_request_with_retry(
        self,
        *,
        send_data: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float]:
        for attempt_idx in range(self._reconnect_retry_count + 1):
            try:
                start = time.time()
                payload = self._packer.pack(send_data)
                with self._client_lock:
                    if self._ws is None:
                        self._connect()
                    assert self._ws is not None
                    self._ws.send(payload)
                    response = self._ws.recv()
                e2e_ms = (time.time() - start) * 1000.0
                if isinstance(response, str):
                    raise RuntimeError(f"Error in JoyRA inference server:\n{response}")
                pred = msgpack_numpy.unpackb(response)
                if not isinstance(pred, dict):
                    raise RuntimeError(
                        f"Expected JoyRA inference response dict, got {type(pred).__name__}"
                    )
                return pred, float(e2e_ms)
            except Exception as err:
                can_retry = (
                    self._is_retryable_connection_error(err)
                    and attempt_idx < self._reconnect_retry_count
                )
                if not can_retry:
                    raise
                self._logger.warning(
                    "JoyRA websocket disconnected (%s). Reconnecting and retrying (%s/%s).",
                    err,
                    attempt_idx + 1,
                    self._reconnect_retry_count,
                )
                with self._client_lock:
                    self._reconnect_locked()
                if self._reconnect_retry_backoff_sec > 0.0:
                    time.sleep(self._reconnect_retry_backoff_sec)

        raise RuntimeError("JoyRA infer retry loop exited unexpectedly.")

    def infer(
        self, policy_input: PolicyInput
    ) -> Tuple[np.ndarray, PolicyInferInfo]:
        pred, e2e_ms = self._send_request_with_retry(
            send_data=build_joyra_request(policy_input)
        )
        chunk = coerce_action_chunk(
            pred["actions"],
            action_dim=self._action_dim,
        )
        info: PolicyInferInfo = {
            "e2e_ms": float(e2e_ms),
            "policy_ms": maybe_get_policy_infer_ms(pred),
            "server_ms": maybe_get_server_infer_ms(pred),
            "server_action_dim": int(np.asarray(pred["actions"]).shape[-1]),
            "used_action_dim": int(self._action_dim),
        }
        if "batch_size" in pred:
            info["batch_size"] = int(pred["batch_size"])
        return chunk, info

    def infer_many(
        self,
        policy_inputs: Sequence[PolicyInput],
    ) -> PolicyBatchInferResult:
        if not policy_inputs:
            raise ValueError("policy_inputs must be non-empty for JoyRA batch infer")
        pred, e2e_ms = self._send_request_with_retry(
            send_data=build_joyra_batch_request(policy_inputs)
        )
        raw_actions = np.asarray(pred["actions"], dtype=np.float32)
        if raw_actions.ndim == 2:
            raw_actions = np.expand_dims(raw_actions, axis=0)
        if raw_actions.ndim != 3:
            raise ValueError(
                f"Expected JoyRA batch actions to be rank-3, got {raw_actions.shape}"
            )
        expected_batch_size = len(policy_inputs)
        if int(raw_actions.shape[0]) != expected_batch_size:
            raise ValueError(
                "JoyRA batch response size does not match request size: "
                f"expected={expected_batch_size} got={int(raw_actions.shape[0])}"
            )
        chunks = [
            coerce_action_chunk(raw_actions[idx], action_dim=self._action_dim)
            for idx in range(expected_batch_size)
        ]
        info: PolicyInferInfo = {
            "e2e_ms": float(e2e_ms),
            "policy_ms": maybe_get_policy_infer_ms(pred),
            "server_ms": maybe_get_server_infer_ms(pred),
            "server_action_dim": int(raw_actions.shape[-1]),
            "used_action_dim": int(self._action_dim),
            "batch_size": expected_batch_size,
        }
        if "batch_size" in pred:
            info["server_batch_size"] = int(pred["batch_size"])
        return chunks, info

    def infer_chunk(
        self, policy_input: PolicyInput
    ) -> Tuple[np.ndarray, PolicyInferInfo]:
        """Backward-compatible alias for older call sites."""
        return self.infer(policy_input)
