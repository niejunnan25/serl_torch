from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))
existing_serl_launcher = sys.modules.get("serl_launcher")
if (
    existing_serl_launcher is not None
    and getattr(existing_serl_launcher, "__file__", None) is None
):
    sys.modules.pop("serl_launcher")

try:
    _WEBSOCKETS_CLIENT_SPEC = importlib.util.find_spec("websockets.sync.client")
except ModuleNotFoundError:
    _WEBSOCKETS_CLIENT_SPEC = None

if _WEBSOCKETS_CLIENT_SPEC is None:
    fake_websockets = types.ModuleType("websockets")
    fake_sync = types.ModuleType("websockets.sync")
    fake_client = types.ModuleType("websockets.sync.client")
    fake_exceptions = types.ModuleType("websockets.exceptions")

    class _ConnectionClosed(Exception):
        pass

    def _placeholder_connect(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("placeholder connect should be monkeypatched in tests")

    fake_client.connect = _placeholder_connect
    fake_sync.client = fake_client
    fake_exceptions.ConnectionClosed = _ConnectionClosed
    fake_websockets.sync = fake_sync
    fake_websockets.exceptions = fake_exceptions
    sys.modules["websockets"] = fake_websockets
    sys.modules["websockets.sync"] = fake_sync
    sys.modules["websockets.sync.client"] = fake_client
    sys.modules["websockets.exceptions"] = fake_exceptions

from serl_launcher.policy.joyra import client as joyra_client_module
from serl_launcher.policy.joyra.client import JoyRAPolicyClient


def _make_unconnected_client() -> JoyRAPolicyClient:
    assert JoyRAPolicyClient is not None
    client = object.__new__(JoyRAPolicyClient)
    client._api_key = None
    client._connect_timeout_sec = 30.0
    client._ping_interval_sec = 20.0
    client._ping_timeout_sec = 120.0
    client._close_timeout_sec = 10.0
    return client


class JoyRAPolicyClientTest(unittest.TestCase):
    def test_connect_kwargs_preserves_supported_proxy_none(self) -> None:
        original_connect = joyra_client_module.websockets.sync.client.connect

        def _connect_with_proxy(
            uri: str,
            *,
            open_timeout: float | None = None,
            proxy: str | None = None,
        ) -> None:
            del uri, open_timeout, proxy

        joyra_client_module.websockets.sync.client.connect = _connect_with_proxy
        try:
            kwargs = _make_unconnected_client()._connect_kwargs()
        finally:
            joyra_client_module.websockets.sync.client.connect = original_connect

        self.assertIn("proxy", kwargs)
        self.assertIsNone(kwargs["proxy"])
        self.assertEqual(kwargs["open_timeout"], 30.0)

    def test_connect_kwargs_omits_unsupported_proxy(self) -> None:
        original_connect = joyra_client_module.websockets.sync.client.connect

        def _connect_without_proxy(
            uri: str,
            *,
            open_timeout: float | None = None,
        ) -> None:
            del uri, open_timeout

        joyra_client_module.websockets.sync.client.connect = _connect_without_proxy
        try:
            kwargs = _make_unconnected_client()._connect_kwargs()
        finally:
            joyra_client_module.websockets.sync.client.connect = original_connect

        self.assertNotIn("proxy", kwargs)
        self.assertEqual(kwargs["open_timeout"], 30.0)


if __name__ == "__main__":
    unittest.main()
