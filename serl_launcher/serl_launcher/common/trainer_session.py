from __future__ import annotations

"""Shared runtime policy helpers for using trainer clients safely."""

from collections import defaultdict
from collections.abc import Callable
import time
from typing import Any
from typing import Optional
from typing import Protocol


class TrainerClientLike(Protocol):
    def update(self) -> bool:
        ...

    def request(self, type: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        ...

    def get_transport_status(
        self,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        ...

    def wait_until_committed(self) -> bool:
        ...


class TrainerClientSession:
    def __init__(
        self,
        *,
        client: TrainerClientLike,
        logger: Any,
        store_name: str | None = None,
        status_fallback: Callable[[], dict[str, Any]] | None = None,
        log_prefix: str = "trainer transport",
    ) -> None:
        self._client = client
        self._logger = logger
        self._store_name = None if store_name is None else str(store_name)
        self._status_fallback = status_fallback
        self._log_prefix = str(log_prefix)
        self._update_failures = 0
        self._best_effort_update_failures = 0
        self._request_failures: dict[str, int] = defaultdict(int)

    def status(self) -> dict[str, Any]:
        try:
            return dict(self._client.get_transport_status(self._store_name))
        except Exception:  # noqa: BLE001
            if self._status_fallback is None:
                return {}
            return dict(self._status_fallback())

    def update(
        self,
        *,
        context: str,
        log_prefix: str | None = None,
        max_failures: int | None = 5,
        failure_message: str | None = None,
    ) -> bool:
        active_log_prefix = (
            self._log_prefix if log_prefix is None else str(log_prefix)
        )
        ok = bool(self._client.update())
        if ok:
            self._update_failures = 0
            return True
        self._update_failures += 1
        self._logger.warning(
            "%s update failed: context=%s consecutive_failures=%s status=%s",
            active_log_prefix,
            str(context),
            int(self._update_failures),
            self.status(),
        )
        if max_failures is not None and int(self._update_failures) >= int(max_failures):
            raise RuntimeError(
                str(failure_message)
                if failure_message is not None
                else f"{active_log_prefix} update failed repeatedly"
            )
        return False

    def update_best_effort(
        self,
        *,
        context: str,
        log_prefix: str | None = None,
        failure_message: str | None = None,
    ) -> bool:
        del failure_message
        active_log_prefix = (
            self._log_prefix if log_prefix is None else str(log_prefix)
        )
        # Match the original SERL actor loop: TrainerClient.update() is a
        # datastore flush hint, not a liveness check.  When the learner misses
        # the ack window, local queued data remains available while it stays
        # inside the queue capacity, and the next update retries from the
        # learner's last accepted/committed id.
        ok = bool(self._client.update())
        if ok:
            if int(self._best_effort_update_failures) > 0:
                self._logger.info(
                    "%s best-effort update recovered: context=%s "
                    "consecutive_failures=%s status=%s",
                    active_log_prefix,
                    str(context),
                    int(self._best_effort_update_failures),
                    self.status(),
                )
            self._best_effort_update_failures = 0
            return True

        self._best_effort_update_failures += 1
        self._logger.warning(
            "%s best-effort update missed ack: context=%s "
            "consecutive_failures=%s status=%s",
            active_log_prefix,
            str(context),
            int(self._best_effort_update_failures),
            self.status(),
        )
        return False

    def update_until_success(
        self,
        *,
        context: str,
        log_prefix: str | None = None,
        retry_sleep_s: float = 0.5,
        max_retry_sleep_s: float = 5.0,
        log_every_s: float = 30.0,
    ) -> bool:
        active_log_prefix = (
            self._log_prefix if log_prefix is None else str(log_prefix)
        )
        next_log_time = 0.0
        sleep_s = max(0.0, float(retry_sleep_s))
        max_sleep_s = max(0.0, float(max_retry_sleep_s))
        log_interval_s = max(0.0, float(log_every_s))

        while True:
            ok = bool(self._client.update())
            if ok:
                if int(self._update_failures) > 0:
                    self._logger.info(
                        "%s update recovered: context=%s "
                        "consecutive_failures=%s status=%s",
                        active_log_prefix,
                        str(context),
                        int(self._update_failures),
                        self.status(),
                    )
                self._update_failures = 0
                return True

            self._update_failures += 1
            now = time.monotonic()
            if int(self._update_failures) == 1 or now >= next_log_time:
                self._logger.warning(
                    "%s update waiting: context=%s consecutive_failures=%s status=%s",
                    active_log_prefix,
                    str(context),
                    int(self._update_failures),
                    self.status(),
                )
                next_log_time = now + log_interval_s

            if sleep_s > 0.0:
                time.sleep(sleep_s)
                if max_sleep_s > sleep_s:
                    sleep_s = min(max_sleep_s, max(sleep_s * 1.5, 0.001))
            else:
                time.sleep(0.0)

    def request(
        self,
        request_type: str,
        payload: dict[str, Any],
        *,
        context: str | None = None,
        log_prefix: str | None = None,
        max_failures: int | None = 5,
        failure_message: str | None = None,
        raise_on_exhaustion: bool = True,
        retry_until_exhausted: bool = False,
        retry_sleep_s: float = 0.1,
    ) -> dict[str, Any] | None:
        active_log_prefix = (
            self._log_prefix if log_prefix is None else str(log_prefix)
        )
        if bool(raise_on_exhaustion) and max_failures is None:
            raise ValueError(
                "max_failures must be set when raise_on_exhaustion is enabled"
            )
        request_key = str(request_type)
        while True:
            response = self._client.request(str(request_type), dict(payload))
            if response is not None:
                self._request_failures[request_key] = 0
                return dict(response)
            self._request_failures[request_key] += 1
            failures = int(self._request_failures[request_key])
            self._logger.warning(
                "%s %s failed: context=%s consecutive_failures=%s status=%s",
                active_log_prefix,
                request_key,
                "" if context is None else str(context),
                failures,
                self.status(),
            )
            if (
                bool(raise_on_exhaustion)
                and max_failures is not None
                and failures >= int(max_failures)
            ):
                raise RuntimeError(
                    str(failure_message)
                    if failure_message is not None
                    else f"{active_log_prefix} {request_key} failed repeatedly"
                )
            if not bool(retry_until_exhausted):
                return None
            time.sleep(float(retry_sleep_s))

    def flush(
        self,
        *,
        context: str,
        wait_until_committed: bool,
        log_prefix: str | None = None,
        max_update_failures: int | None = None,
        update_failure_message: str | None = None,
        flush_update_failure_message: str | None = None,
        wait_timeout_message: str | None = None,
    ) -> None:
        active_log_prefix = (
            self._log_prefix if log_prefix is None else str(log_prefix)
        )
        if not bool(wait_until_committed) and max_update_failures is None:
            self.update_best_effort(
                context=context,
                log_prefix=active_log_prefix,
                failure_message=update_failure_message,
            )
            return
        if max_update_failures is None:
            updated = self.update_until_success(
                context=context,
                log_prefix=active_log_prefix,
            )
        else:
            updated = self.update(
                context=context,
                log_prefix=active_log_prefix,
                max_failures=max_update_failures,
                failure_message=update_failure_message,
            )
        if not bool(updated):
            if bool(wait_until_committed):
                raise RuntimeError(
                    str(flush_update_failure_message)
                    if flush_update_failure_message is not None
                    else (
                        f"{active_log_prefix} update did not succeed during flush: "
                        f"context={str(context)}"
                    )
                )
            return
        if not bool(wait_until_committed):
            return
        if not bool(self._client.wait_until_committed()):
            raise RuntimeError(
                str(wait_timeout_message)
                if wait_timeout_message is not None
                else (
                    f"{active_log_prefix} wait_until_committed timed out: "
                    f"context={str(context)}"
                )
            )
