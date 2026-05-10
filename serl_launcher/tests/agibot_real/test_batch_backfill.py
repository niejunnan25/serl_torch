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

if "gym" not in sys.modules and "gymnasium" not in sys.modules:
    fake_gym = types.ModuleType("gym")

    class _FakeBox:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    class _FakeDict:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    fake_gym.spaces = types.SimpleNamespace(Box=_FakeBox, Dict=_FakeDict)
    sys.modules["gym"] = fake_gym

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_launcher.policy.base import PolicyInput
    from serl_launcher.policy.joyra import msgpack_numpy
    from serl_launcher.policy.joyra.client import JoyRAPolicyClient
    from serl_torch.examples.agibot_real.runtime.transition_assembly import (
        backfill_post_step_residual_obs,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    PolicyInput = object  # type: ignore[assignment]
    JoyRAPolicyClient = object  # type: ignore[assignment]
    msgpack_numpy = None  # type: ignore[assignment]
    backfill_post_step_residual_obs = None  # type: ignore[assignment]


def _make_policy_input(seed: int) -> PolicyInput:
    image = np.full((4, 4, 3), seed, dtype=np.uint8)
    return PolicyInput(
        prompt=f"prompt-{seed}",
        state=np.arange(14, dtype=np.float32) + float(seed),
        images={
            "image_rgb_0": image,
            "image_rgb_1": image + 1,
            "image_rgb_2": image + 2,
        },
        image_mask={
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": True,
        },
        metadata={
            "joyra_state": np.arange(18, dtype=np.float32) + float(seed),
        },
    )


def _make_observation(seed: int) -> dict[str, object]:
    image = np.full((4, 4, 3), seed, dtype=np.uint8)
    return {
        "observation/state": np.arange(14, dtype=np.float32) + float(seed),
        "observation/image": image,
        "observation/wrist_left_image": image + 1,
        "observation/wrist_right_image": image + 2,
    }


class _FakeWebSocket:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        if not self.responses:
            raise RuntimeError("No fake websocket responses left")
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class _TestJoyRAPolicyClient(JoyRAPolicyClient):
    def __init__(self, responses: list[bytes]) -> None:
        self._test_ws = _FakeWebSocket(responses)
        super().__init__(
            host="127.0.0.1",
            port=8000,
            action_dim=18,
            logger=logging.getLogger(__name__),
        )

    def _connect(self) -> None:
        self._ws = self._test_ws
        self._server_metadata = {"status": "ok"}


class _FakeBatchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[PolicyInput, ...]] = []

    def infer_many(
        self,
        policy_inputs: tuple[PolicyInput, ...],
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        self.calls.append(tuple(policy_inputs))
        raw_actions = [
            np.full((3, 18), fill_value=float(idx + 1), dtype=np.float32)
            for idx, _ in enumerate(policy_inputs)
        ]
        return raw_actions, {"batch_size": len(policy_inputs)}


class _FakeBatchedBasePolicy:
    def __init__(self) -> None:
        self.backend_type = "joyra"
        self.action_dim = 14
        self.chunk_horizon = 3
        self.client = _FakeBatchClient()


class _FakeSerialBasePolicy:
    def __init__(self) -> None:
        self.backend_type = "openpi"
        self.calls: list[tuple[dict[str, object], str]] = []

    def infer(
        self,
        obs: dict[str, object],
        *,
        prompt: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        self.calls.append((dict(obs), str(prompt)))
        fill_value = float(len(self.calls))
        return np.full((3, 14), fill_value=fill_value, dtype=np.float32), {}


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing dependency: {_IMPORT_ERROR}")
class AgiBotBatchBackfillTest(unittest.TestCase):
    def test_joyra_client_infer_many_packs_examples_and_decodes_batch(self) -> None:
        packer = msgpack_numpy.Packer()
        response = packer.pack(
            {
                "actions": np.asarray(
                    [
                        np.full((2, 18), 1.0, dtype=np.float32),
                        np.full((2, 18), 2.0, dtype=np.float32),
                    ],
                    dtype=np.float32,
                ),
                "policy_timing": {"infer_ms": 12.5},
                "server_timing": {"infer_ms": 13.5},
                "batch_size": 2,
            }
        )
        client = _TestJoyRAPolicyClient([response])

        chunks, info = client.infer_many(
            [_make_policy_input(0), _make_policy_input(1)]
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (2, 18))
        self.assertTrue(np.allclose(chunks[0], 1.0))
        self.assertTrue(np.allclose(chunks[1], 2.0))
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["server_batch_size"], 2)
        sent_payload = msgpack_numpy.unpackb(client._test_ws.sent[0])
        self.assertIn("examples", sent_payload)
        self.assertEqual(len(sent_payload["examples"]), 2)
        self.assertEqual(sent_payload["examples"][0]["prompt"], "prompt-0")

    def test_backfill_post_step_residual_obs_uses_single_batched_joyra_call(
        self,
    ) -> None:
        base_policy = _FakeBatchedBasePolicy()
        observations = [_make_observation(3), _make_observation(7)]

        base_action_chunks, residual_observations = backfill_post_step_residual_obs(
            observations=observations,
            task_prompt="pick object",
            base_policy=base_policy,
            image_keys=("image_rgb_0", "image_rgb_1", "image_rgb_2"),
            residual_alpha=0.2,
        )

        self.assertEqual(len(base_policy.client.calls), 1)
        self.assertEqual(len(base_policy.client.calls[0]), 2)
        self.assertEqual(base_policy.client.calls[0][0].prompt, "pick object")
        self.assertEqual(len(base_action_chunks), 2)
        self.assertEqual(base_action_chunks[0].shape, (3, 14))
        self.assertTrue(np.allclose(base_action_chunks[0], 1.0))
        self.assertTrue(np.allclose(base_action_chunks[1], 2.0))
        self.assertEqual(len(residual_observations), 2)
        self.assertEqual(
            residual_observations[0]["base_action_chunk"].shape,
            (1, 3, 14),
        )
        self.assertTrue(
            np.allclose(
                residual_observations[1]["base_action_chunk"][0],
                2.0,
            )
        )

    def test_backfill_post_step_residual_obs_falls_back_to_serial_infer(self) -> None:
        base_policy = _FakeSerialBasePolicy()
        observations = [_make_observation(1), _make_observation(2)]

        base_action_chunks, residual_observations = backfill_post_step_residual_obs(
            observations=observations,
            task_prompt="serial path",
            base_policy=base_policy,
            image_keys=("image_rgb_0", "image_rgb_1", "image_rgb_2"),
            residual_alpha=0.3,
        )

        self.assertEqual(len(base_policy.calls), 2)
        self.assertEqual(len(base_action_chunks), 2)
        self.assertTrue(np.allclose(base_action_chunks[0], 1.0))
        self.assertTrue(np.allclose(base_action_chunks[1], 2.0))
        self.assertEqual(len(residual_observations), 2)


if __name__ == "__main__":
    unittest.main()
