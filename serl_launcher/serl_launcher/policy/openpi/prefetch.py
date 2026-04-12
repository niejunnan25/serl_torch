"""Asynchronous OpenPI chunk prefetch helpers."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

from serl_launcher.policy.base import PolicyInferInfo, PolicyInput
from serl_launcher.policy.openpi.client import OpenPIPolicyClient


class AsyncOpenPIPolicyPrefetcher:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        action_dim: int,
        logger: logging.Logger,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.action_dim = int(action_dim)
        self.logger = logger
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openpi-prefetch",
        )
        self._client: Optional[OpenPIPolicyClient] = None
        self._lock = threading.Lock()
        self._closed = False

    def _get_client(self) -> OpenPIPolicyClient:
        with self._lock:
            if self._client is None:
                self._client = OpenPIPolicyClient(
                    host=self.host,
                    port=self.port,
                    action_dim=self.action_dim,
                    logger=self.logger,
                )
            return self._client

    def _infer_chunk(
        self,
        policy_input: PolicyInput,
    ) -> tuple[np.ndarray, PolicyInferInfo]:
        return self._get_client().infer_chunk(policy_input)

    def submit(
        self,
        policy_input: PolicyInput,
    ) -> Future[tuple[np.ndarray, PolicyInferInfo]]:
        if self._closed:
            raise RuntimeError("OpenPI prefetcher is closed")
        return self._executor.submit(self._infer_chunk, policy_input)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
