from __future__ import annotations

import logging
import pickle
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RAW_ROLLOUT_FORMAT_VERSION,
)
from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RAW_ROLLOUT_MANIFEST_FILENAME,
)
from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RawRolloutRecorder,
)


def _payload(*, chunk_seq: int, episode_step_start: int) -> dict[str, object]:
    return {
        "chunk_seq": int(chunk_seq),
        "episode_id": 7,
        "episode_step_start": int(episode_step_start),
        "task_prompt": "pick",
        "residual_obs_before_chunk": {
            "state": np.asarray([float(episode_step_start)], dtype=np.float32)
        },
        "action_chunk": np.asarray([[0.1]], dtype=np.float32),
        "chunk_result": {
            "observations": [
                {"state": np.asarray([float(episode_step_start + 1)], dtype=np.float32)}
            ],
            "rewards": [1.0],
            "dones": [False],
            "infos": [{"controller_action_executed": True, "success": False}],
            "obs": {
                "state": np.asarray([float(episode_step_start + 1)], dtype=np.float32)
            },
            "done": False,
            "truncated": False,
            "reward_sum": 1.0,
            "info": {"success": False},
        },
    }


class RawRolloutRecorderTest(unittest.TestCase):
    def test_recorder_writes_manifest_and_episode_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
                metadata={"task_key": "test"},
            )

            self.assertTrue(
                recorder.append_chunk(payload=_payload(chunk_seq=0, episode_step_start=0))
            )
            self.assertFalse(
                recorder.append_chunk(payload=_payload(chunk_seq=0, episode_step_start=0))
            )
            path = recorder.finalize_episode(
                marker={"episode_id": 7, "last_chunk_seq": 0}
            )

            self.assertIsNotNone(path)
            assert path is not None
            with open(path, "rb") as fp:
                episode = pickle.load(fp)
            self.assertEqual(episode["format_version"], RAW_ROLLOUT_FORMAT_VERSION)
            self.assertEqual(episode["num_steps"], 1)
            self.assertEqual(len(episode["chunks"]), 1)
            self.assertTrue((Path(tmpdir) / RAW_ROLLOUT_MANIFEST_FILENAME).is_file())

    def test_non_contiguous_step_start_taints_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = RawRolloutRecorder(
                output_root=Path(tmpdir),
                logger=logging.getLogger(__name__),
            )

            recorder.append_chunk(payload=_payload(chunk_seq=0, episode_step_start=0))
            with self.assertRaisesRegex(ValueError, "expected contiguous episode steps"):
                recorder.append_chunk(payload=_payload(chunk_seq=1, episode_step_start=3))
            self.assertIsNone(
                recorder.finalize_episode(marker={"episode_id": 7, "last_chunk_seq": 1})
            )


if __name__ == "__main__":
    unittest.main()
