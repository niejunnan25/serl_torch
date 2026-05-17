from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from serl_launcher.rollout import ProcessorClient
    from serl_launcher.rollout import ProcessorServer
    from serl_launcher.rollout import ProcessorTransportConfig
    from serl_torch.examples.libero.env.observation import build_libero_state
    from serl_torch.examples.libero.runtime.raw_rollout_recorder import (
        RAW_ROLLOUT_FORMAT_VERSION,
    )
    from serl_torch.examples.libero.runtime.raw_rollout_recorder import (
        RAW_ROLLOUT_MANIFEST_FILENAME,
    )
    from serl_torch.examples.libero.runtime.raw_rollout_recorder import (
        RawRolloutRecorder,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = exc
    ProcessorClient = object  # type: ignore[assignment]
    ProcessorServer = object  # type: ignore[assignment]
    ProcessorTransportConfig = object  # type: ignore[assignment]
    build_libero_state = None  # type: ignore[assignment]
    RAW_ROLLOUT_FORMAT_VERSION = ""
    RAW_ROLLOUT_MANIFEST_FILENAME = ""
    RawRolloutRecorder = object  # type: ignore[assignment]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_obs(seed: float) -> dict[str, object]:
    pixel = np.full((4, 4, 3), int(seed), dtype=np.uint8)
    return {
        "robot0_eef_pos": np.asarray([seed, seed + 0.1, seed + 0.2], dtype=np.float32),
        "robot0_eef_axis_angle": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
        "agentview_image": pixel,
        "robot0_eye_in_hand_image": pixel,
    }


def _build_chunk_payload(
    *,
    episode_id: int,
    chunk_seq: int,
    episode_step_start: int,
    obs_seeds: list[float],
    action_base: float,
    env_done_last: bool,
) -> dict[str, object]:
    if len(obs_seeds) < 2:
        raise ValueError("obs_seeds must contain at least 2 observations")

    observations = [_fake_obs(seed) for seed in obs_seeds]
    steps: list[dict[str, object]] = []
    for step_idx in range(len(observations) - 1):
        steps.append(
            {
                "obs": observations[step_idx],
                "action": np.asarray(
                    [action_base + step_idx, action_base + step_idx + 0.5],
                    dtype=np.float32,
                ),
                "reward": float(step_idx + 1),
                "done": bool(step_idx == len(observations) - 2),
                "truncated": False,
                "info": {
                    "env_done": bool(
                        env_done_last and step_idx == len(observations) - 2
                    )
                },
                "next_obs": observations[step_idx + 1],
            }
        )
    return {
        "chunk_seq": int(chunk_seq),
        "episode_id": int(episode_id),
        "episode_step_start": int(episode_step_start),
        "task_prompt": "stack blocks",
        "chunk_result": {
            "steps": steps,
            "num_steps": len(steps),
            "reward_sum": float(sum(float(step["reward"]) for step in steps)),
            "obs": observations[-1],
            "done": bool(steps[-1]["done"]),
            "truncated": False,
            "info": dict(steps[-1]["info"]),
        },
    }


@unittest.skipIf(_IMPORT_ERROR is not None, str(_IMPORT_ERROR))
class RawRolloutRecorderTest(unittest.TestCase):
    def test_recorder_finalizes_episode_sequence_with_expected_lengths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            payload0 = _build_chunk_payload(
                episode_id=7,
                chunk_seq=0,
                episode_step_start=0,
                obs_seeds=[1.0, 2.0, 3.0],
                action_base=0.1,
                env_done_last=False,
            )
            payload1 = _build_chunk_payload(
                episode_id=7,
                chunk_seq=1,
                episode_step_start=2,
                obs_seeds=[3.0, 4.0],
                action_base=10.0,
                env_done_last=True,
            )

            self.assertTrue(recorder.append_chunk(payload=payload0))
            self.assertTrue(recorder.append_chunk(payload=payload1))
            self.assertFalse(list(Path(tmpdir).glob("episode_*.pkl")))

            episode_path = recorder.finalize_episode(marker={"episode_id": 7})

            self.assertIsNotNone(episode_path)
            assert episode_path is not None
            with open(episode_path, "rb") as fp:
                episode_payload = pickle.load(fp)

            self.assertEqual(
                episode_payload["format_version"], RAW_ROLLOUT_FORMAT_VERSION
            )
            self.assertEqual(episode_payload["episode_id"], 7)
            self.assertEqual(episode_payload["num_steps"], 3)
            self.assertEqual(len(episode_payload["observations"]), 4)
            self.assertEqual(len(episode_payload["states"]), 4)
            self.assertEqual(len(episode_payload["actions"]), 3)
            self.assertEqual(len(episode_payload["rewards"]), 3)
            self.assertEqual(len(episode_payload["dones"]), 3)
            self.assertEqual(len(episode_payload["truncations"]), 3)
            self.assertEqual(len(episode_payload["infos"]), 3)
            self.assertEqual(episode_payload["chunk_seqs"], [0, 0, 1])
            self.assertAlmostEqual(episode_payload["episode_return"], 4.0)
            self.assertEqual(episode_payload["success"], True)
            self.assertTrue(
                np.array_equal(
                    episode_payload["states"][2],
                    build_libero_state(episode_payload["observations"][2]),
                )
            )

            with open(
                Path(tmpdir) / RAW_ROLLOUT_MANIFEST_FILENAME, "r", encoding="utf-8"
            ) as fp:
                manifest = json.load(fp)
            self.assertEqual(manifest["episode_files"], [episode_path.name])
            self.assertEqual(manifest["recycle_stats"]["episodes_written"], 1)
            self.assertEqual(manifest["recycle_stats"]["steps_written"], 3)
            self.assertEqual(manifest["recycle_stats"]["append_errors"], 0)
            self.assertEqual(manifest["recycle_stats"]["write_errors"], 0)

    def test_recorder_dedupes_duplicate_chunk_seq(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            payload = _build_chunk_payload(
                episode_id=3,
                chunk_seq=4,
                episode_step_start=0,
                obs_seeds=[1.0, 2.0],
                action_base=0.1,
                env_done_last=True,
            )

            self.assertTrue(recorder.append_chunk(payload=payload))
            self.assertFalse(recorder.append_chunk(payload=payload))

            episode_path = recorder.finalize_episode(marker={"episode_id": 3})
            assert episode_path is not None
            with open(episode_path, "rb") as fp:
                episode_payload = pickle.load(fp)
            self.assertEqual(episode_payload["num_steps"], 1)
            self.assertEqual(len(episode_payload["observations"]), 2)

    def test_recorder_writes_failure_episode_without_success_flag(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            payload = _build_chunk_payload(
                episode_id=5,
                chunk_seq=0,
                episode_step_start=0,
                obs_seeds=[1.0, 2.0],
                action_base=0.1,
                env_done_last=False,
            )

            recorder.append_chunk(payload=payload)
            episode_path = recorder.finalize_episode(marker={"episode_id": 5})

            assert episode_path is not None
            with open(episode_path, "rb") as fp:
                episode_payload = pickle.load(fp)
            self.assertEqual(episode_payload["success"], False)

    def test_recorder_success_uses_raw_env_signal_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            payload = _build_chunk_payload(
                episode_id=6,
                chunk_seq=0,
                episode_step_start=0,
                obs_seeds=[1.0, 2.0],
                action_base=0.1,
                env_done_last=False,
            )

            recorder.append_chunk(payload=payload)
            episode_path = recorder.finalize_episode(
                marker={
                    "episode_id": 6,
                    "rollout_stats": {"rollout": {"success": True}},
                }
            )

            assert episode_path is not None
            with open(episode_path, "rb") as fp:
                episode_payload = pickle.load(fp)
            self.assertEqual(episode_payload["success"], False)

    def test_recorder_retries_after_write_failure_without_losing_episode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            payload = _build_chunk_payload(
                episode_id=11,
                chunk_seq=0,
                episode_step_start=0,
                obs_seeds=[1.0, 2.0, 3.0],
                action_base=0.1,
                env_done_last=True,
            )

            recorder.append_chunk(payload=payload)
            with patch(
                "serl_torch.examples.libero.runtime.raw_rollout_recorder.pickle.dump",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    recorder.finalize_episode(marker={"episode_id": 11})

            failed_snapshot = recorder.status_snapshot()
            self.assertEqual(failed_snapshot["pending_episodes"], 1)
            self.assertEqual(failed_snapshot["append_errors"], 0)
            self.assertEqual(failed_snapshot["write_errors"], 1)

            episode_path = recorder.finalize_episode(marker={"episode_id": 11})
            assert episode_path is not None
            with open(episode_path, "rb") as fp:
                episode_payload = pickle.load(fp)
            self.assertEqual(episode_payload["num_steps"], 2)

            final_snapshot = recorder.status_snapshot()
            self.assertEqual(final_snapshot["pending_episodes"], 0)
            self.assertEqual(final_snapshot["write_errors"], 1)
            self.assertEqual(final_snapshot["episodes_written"], 1)

    def test_recorder_skips_followup_chunks_after_episode_is_tainted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )
            bad_payload = _build_chunk_payload(
                episode_id=13,
                chunk_seq=0,
                episode_step_start=1,
                obs_seeds=[1.0, 2.0],
                action_base=0.1,
                env_done_last=False,
            )
            followup_payload = _build_chunk_payload(
                episode_id=13,
                chunk_seq=1,
                episode_step_start=0,
                obs_seeds=[2.0, 3.0],
                action_base=1.0,
                env_done_last=True,
            )

            with self.assertRaisesRegex(
                ValueError,
                "raw rollout recorder expected contiguous episode steps",
            ):
                recorder.append_chunk(payload=bad_payload)

            self.assertFalse(recorder.append_chunk(payload=followup_payload))
            self.assertIsNone(recorder.finalize_episode(marker={"episode_id": 13}))
            self.assertEqual(list(Path(tmpdir).glob("episode_*.pkl")), [])

    def test_recorder_only_writes_after_processor_flush(self) -> None:
        port = _find_free_port()
        flush_calls: list[tuple[str, bool]] = []
        server = ProcessorServer(
            transport_config=ProcessorTransportConfig(
                host="127.0.0.1",
                port=port,
                timeout_ms=200,
                queue_capacity=2,
            ),
            transport_status_fn=lambda: {"ready": True},
            flush_transport_fn=lambda context, wait_until_committed: flush_calls.append(
                (str(context), bool(wait_until_committed))
            ),
            wait_committed_on_episode_end=False,
            wait_committed_on_shutdown=True,
        )
        server.start()
        client = ProcessorClient(
            transport_config=ProcessorTransportConfig(
                host="127.0.0.1",
                port=port,
                timeout_ms=200,
                queue_capacity=2,
            ),
            logger=logging.getLogger(__name__),
        )
        try:
            with TemporaryDirectory() as tmpdir:
                recorder = RawRolloutRecorder(
                    output_root=Path(tmpdir),
                    logger=logging.getLogger(__name__),
                )
                payload = _build_chunk_payload(
                    episode_id=9,
                    chunk_seq=2,
                    episode_step_start=0,
                    obs_seeds=[1.0, 2.0],
                    action_base=0.1,
                    env_done_last=True,
                )

                client.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)
                submit_response = client.submit(payload=payload, context="submit")
                self.assertEqual(submit_response["accepted_chunk_seq"], 2)

                mark_response = client.mark_episode_end(episode_id=9, last_chunk_seq=2)
                self.assertEqual(mark_response["pending_episode_flushes"], 1)

                queued_payload = server.get_chunk(timeout_s=1.0)
                assert queued_payload is not None
                self.assertEqual(
                    int(queued_payload["chunk_seq"]), int(payload["chunk_seq"])
                )
                self.assertEqual(
                    int(queued_payload["episode_id"]),
                    int(payload["episode_id"]),
                )
                self.assertEqual(
                    int(queued_payload["episode_step_start"]),
                    int(payload["episode_step_start"]),
                )
                recorder.append_chunk(payload=queued_payload)

                server.flush_ready_episode_markers()
                self.assertEqual(server.consume_flushed_episode_markers(), [])
                self.assertEqual(list(Path(tmpdir).glob("episode_*.pkl")), [])

                server.mark_chunk_committed(chunk_seq=2)
                server.task_done()
                server.flush_ready_episode_markers()
                flushed_markers = server.consume_flushed_episode_markers()
                self.assertEqual(len(flushed_markers), 1)
                self.assertEqual(flush_calls, [("episode_9_end", False)])

                episode_path = recorder.finalize_episode(marker=flushed_markers[0])
                self.assertIsNotNone(episode_path)
                self.assertEqual(len(list(Path(tmpdir).glob("episode_*.pkl"))), 1)
        finally:
            client.close()
            server.stop()


if __name__ == "__main__":
    unittest.main()
