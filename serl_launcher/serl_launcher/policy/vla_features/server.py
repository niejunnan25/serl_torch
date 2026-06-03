"""Generic websocket server for VLA feature extraction."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import websockets.asyncio.server as _server
import websockets.frames

from openpi_client import msgpack_numpy

FeatureFn = Callable[[dict[str, Any]], Mapping[str, Any]]


class VLAFeatureServer:
    """Serve precomputed VLA/RL-token features over the OpenPI websocket format."""

    def __init__(
        self,
        feature_fn: FeatureFn,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        metadata: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._feature_fn = feature_fn
        self._host = str(host)
        self._port = int(port)
        self._metadata = dict(metadata or {})
        self._logger = logger or logging.getLogger(__name__)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            self._logger.info("VLA feature server listening on ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        self._logger.info("Client connected from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        if self._metadata:
            await websocket.send(packer.pack(dict(self._metadata)))

        while True:
            try:
                raw_msg = await websocket.recv()
                raw_obs = msgpack_numpy.unpackb(raw_msg)
                start_time = time.monotonic()
                features = dict(self._feature_fn(raw_obs))
                features.setdefault("server_timing", {})
                features["server_timing"].setdefault("infer_ms", (time.monotonic() - start_time) * 1000.0)
                await websocket.send(packer.pack(features))
            except websockets.ConnectionClosed:
                self._logger.info("Client %s disconnected", websocket.remote_address)
                break
            except Exception as exc:
                self._logger.error("Error processing feature request: %s", exc, exc_info=True)
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason=str(exc)[:120],
                    )
                except Exception:
                    pass
                break
