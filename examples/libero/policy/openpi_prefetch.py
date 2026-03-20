"""Asynchronous OpenPI chunk prefetch helpers."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .observation import LiberoObservationCache
from .openpi_client import OpenPIChunkClient


class _AsyncOpenPIChunkPrefetcher:
    def __init__(self, *, host: str, port: int, logger: logging.Logger) -> None:
        self.host = str(host)
        self.port = int(port)
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-openpi")
        self._client: Optional[OpenPIChunkClient] = None
        self._lock = threading.Lock()
        self._closed = False

    def _get_client(self) -> OpenPIChunkClient:
        with self._lock:
            if self._client is None:
                self._client = OpenPIChunkClient(host=self.host, port=self.port, logger=self.logger)
            return self._client

    def _infer_chunk(
        self,
        obs: Dict[str, Any],
        prompt: str,
        *,
        obs_cache: Optional[LiberoObservationCache] = None,
        cache_key: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        return self._get_client().infer_chunk(
            obs,
            prompt,
            obs_cache=obs_cache,
            cache_key=cache_key,
        )

    def submit(
        self,
        obs: Dict[str, Any],
        prompt: str,
        *,
        obs_cache: Optional[LiberoObservationCache] = None,
        cache_key: Optional[Any] = None,
    ) -> Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]:
        if self._closed:
            raise RuntimeError("OpenPI prefetcher is closed")
        return self._executor.submit(
            self._infer_chunk,
            obs,
            prompt,
            obs_cache=obs_cache,
            cache_key=cache_key,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
