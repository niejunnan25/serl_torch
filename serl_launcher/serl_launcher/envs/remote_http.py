"""Shared HTTP pickle RPC helpers for remote environment clients and servers."""
from __future__ import annotations

import http.client
import logging
import pickle
import traceback
from http.server import BaseHTTPRequestHandler
from typing import Any, Optional

import numpy as np


def sanitize_pickle_value(value: Any) -> Any:
    """Convert numpy values into plain Python containers before pickling."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: sanitize_pickle_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_pickle_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(sanitize_pickle_value(val) for val in value)
    return value


class RemoteHttpRpcClient:
    """Small HTTP RPC client for environment processes."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_sec: float = 120.0,
        keep_alive: bool = False,
        rpc_path: str = "/rpc",
        max_transport_attempts: int = 2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.keep_alive = bool(keep_alive)
        self.rpc_path = str(rpc_path)
        self.max_transport_attempts = max(1, int(max_transport_attempts))
        self.logger = logger or logging.getLogger(__name__)
        self._conn: Optional[http.client.HTTPConnection] = None

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _build_conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=self.timeout_sec,
        )

    def _ensure_conn(self) -> http.client.HTTPConnection:
        if not self.keep_alive:
            return self._build_conn()
        if self._conn is None:
            self._conn = self._build_conn()
        return self._conn

    def _reset_keep_alive_conn(self) -> None:
        if self.keep_alive:
            self.close()

    def _rpc_once(self, method: str, payload: bytes) -> Any:
        conn = self._ensure_conn()
        try:
            conn.request(
                "POST",
                self.rpc_path,
                body=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Connection": "keep-alive" if self.keep_alive else "close",
                },
            )
            resp = conn.getresponse()
            try:
                resp_bytes = resp.read()
            finally:
                if self.keep_alive:
                    if getattr(resp, "will_close", False):
                        self.close()
                else:
                    conn.close()
        except Exception:
            if not self.keep_alive:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

        if resp.status != 200:
            self._reset_keep_alive_conn()
            raise RuntimeError(
                f"remote env rpc failed method={method} status={resp.status} reason={resp.reason}"
            )

        data = pickle.loads(resp_bytes)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"remote env rpc invalid response type for method={method}"
            )
        if not bool(data.get("ok", False)):
            err = str(data.get("error", "unknown remote error"))
            raise RuntimeError(f"remote env rpc method={method} failed: {err}")
        return data.get("result", None)

    def call(self, method: str, **kwargs: Any) -> Any:
        payload = pickle.dumps(
            {"method": method, "kwargs": sanitize_pickle_value(kwargs)},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_transport_attempts):
            try:
                return self._rpc_once(method, payload)
            except (
                OSError,
                EOFError,
                http.client.HTTPException,
                pickle.PickleError,
            ) as exc:
                last_exc = exc
                self._reset_keep_alive_conn()
                if attempt + 1 < self.max_transport_attempts:
                    self.logger.debug(
                        "remote env rpc reconnect: method=%s error=%s",
                        method,
                        exc,
                    )
                    continue
                break

        assert last_exc is not None
        raise RuntimeError(
            f"remote env rpc method={method} transport error: {last_exc}"
        ) from last_exc


def make_pickle_rpc_handler(
    state: Any,
    logger: logging.Logger,
    *,
    keep_alive: bool,
    rpc_path: str = "/rpc",
) -> type[BaseHTTPRequestHandler]:
    """Build a small pickle-over-HTTP RPC request handler for an env server."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if keep_alive else "HTTP/1.0"

        def _should_keep_alive(self) -> bool:
            if not keep_alive:
                return False
            return str(self.headers.get("Connection", "")).lower() != "close"

        def _write(self, status: int, payload: dict[str, Any]) -> None:
            keep_conn = self._should_keep_alive()
            body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "keep-alive" if keep_conn else "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = not keep_conn

        def do_POST(self) -> None:  # noqa: N802
            if self.path != rpc_path:
                self._write(404, {"ok": False, "error": "not found"})
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len)
                req = pickle.loads(raw)
                method = str(req["method"])
                kwargs = req.get("kwargs", {})
                fn = getattr(state, method, None)
                if fn is None:
                    raise RuntimeError(f"unknown method: {method}")
                result = fn(**kwargs)
                self._write(200, {"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                logger.error("rpc error: %s\n%s", exc, traceback.format_exc())
                self._write(200, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler
