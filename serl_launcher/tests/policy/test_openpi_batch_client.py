from __future__ import annotations

import logging
import sys
from pathlib import Path
import types
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

if "openpi_client.websocket_client_policy" not in sys.modules:
    fake_module = types.ModuleType("openpi_client.websocket_client_policy")

    class _PlaceholderWebsocketClientPolicy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    fake_module.WebsocketClientPolicy = _PlaceholderWebsocketClientPolicy
    sys.modules["openpi_client.websocket_client_policy"] = fake_module

if "websockets" not in sys.modules:
    fake_websockets = types.ModuleType("websockets")

    class _ConnectionClosed(OSError):
        pass

    fake_websockets.exceptions = types.SimpleNamespace(ConnectionClosed=_ConnectionClosed)
    sys.modules["websockets"] = fake_websockets

from serl_launcher.policy.base import PolicyInput
from serl_launcher.policy.openpi.client import OpenPIPolicyClient


def _make_policy_input(seed: int) -> PolicyInput:
    image = np.full((4, 4, 3), seed, dtype=np.uint8)
    return PolicyInput(
        prompt=f"prompt-{seed}",
        state=np.arange(8, dtype=np.float32) + float(seed),
        images={
            "image_rgb_0": image,
            "image_rgb_1": image + 1,
            "image_rgb_2": image + 2,
        },
        image_mask={
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": False,
        },
        metadata={},
    )


class _FakeWebsocketClientPolicy:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.sent: list[dict] = []

    def infer(self, payload: dict) -> dict:
        self.sent.append(payload)
        if not self._responses:
            raise RuntimeError("no fake responses left")
        return self._responses.pop(0)

    def close(self) -> None:
        pass


class _TestOpenPIPolicyClient(OpenPIPolicyClient):
    def __init__(self, responses: list[dict]) -> None:
        self._responses = responses
        super().__init__(
            host="127.0.0.1",
            port=8000,
            action_dim=7,
            logger=logging.getLogger(__name__),
        )

    def _make_client(self):
        return _FakeWebsocketClientPolicy(self._responses)


class OpenPIBatchClientTest(unittest.TestCase):
    def test_infer_many_packs_examples_and_decodes_batch(self) -> None:
        client = _TestOpenPIPolicyClient(
            [
                {
                    "actions": np.asarray(
                        [
                            np.full((5, 7), 1.0, dtype=np.float32),
                            np.full((5, 7), 2.0, dtype=np.float32),
                        ],
                        dtype=np.float32,
                    ),
                    "policy_timing": {"infer_ms": 12.5},
                    "server_timing": {"infer_ms": 13.5},
                    "batch_size": 2,
                }
            ]
        )

        chunks, info = client.infer_many(
            [_make_policy_input(0), _make_policy_input(1)]
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (5, 7))
        self.assertTrue(np.allclose(chunks[0], 1.0))
        self.assertTrue(np.allclose(chunks[1], 2.0))
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["server_batch_size"], 2)
        self.assertEqual(info["server_action_dim"], 7)
        sent_payload = client._client.sent[0]
        self.assertIn("examples", sent_payload)
        self.assertEqual(len(sent_payload["examples"]), 2)
        self.assertEqual(sent_payload["examples"][0]["prompt"], "prompt-0")

    def test_infer_many_rejects_batch_size_mismatch(self) -> None:
        client = _TestOpenPIPolicyClient(
            [
                {
                    "actions": np.asarray(
                        [np.full((5, 7), 1.0, dtype=np.float32)],
                        dtype=np.float32,
                    )
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "response size does not match request size"):
            client.infer_many([_make_policy_input(0), _make_policy_input(1)])


if __name__ == "__main__":
    unittest.main()
