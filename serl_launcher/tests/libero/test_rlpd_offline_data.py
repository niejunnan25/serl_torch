from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

if OmegaConf is not None:
    from serl_torch.examples.libero.rlpd.config import parse_train_cfg
    from serl_torch.examples.libero.rlpd.offline_data import (
        _prepare_demo_transitions,
    )
    from serl_torch.examples.libero.rlpd.offline_data import (
        resolve_and_validate_prepared_paths,
    )


class _FakeLogger:
    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _train_cfg_with_prepared_path(prepared_path: str) -> object:
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "examples"
        / "libero"
        / "configs"
        / "train_rlpd.yaml"
    )
    cfg.runtime.role = "learner"
    cfg.offline.enabled = True
    cfg.offline.prepared_path = prepared_path
    return parse_train_cfg(cfg)


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class LiberoRLPDOfflineDataTest(unittest.TestCase):
    def test_validate_prepared_offline_rejects_manifestless_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_dir = Path(tmpdir) / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            with open(prepared_dir / "episode_000000.pkl", "wb") as fp:
                pickle.dump([], fp, protocol=pickle.HIGHEST_PROTOCOL)

            cfg = _train_cfg_with_prepared_path(str(prepared_dir))
            with self.assertRaisesRegex(
                ValueError,
                "must contain manifest.json",
            ):
                resolve_and_validate_prepared_paths(
                    cfg,
                    logger=_FakeLogger(),
                )

    def test_prepare_demo_transitions_preserves_raw_expert_actions(self) -> None:
        expert_actions = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0],
            ],
            dtype=np.float32,
        )
        demo = {
            "obs": {
                "agentview_rgb": np.zeros((2, 8, 8, 3), dtype=np.uint8),
                "eye_in_hand_rgb": np.ones((2, 8, 8, 3), dtype=np.uint8),
                "ee_pos": np.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32),
                "ee_ori": np.asarray([[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]], dtype=np.float32),
                "gripper_states": np.asarray([[0.7, 0.8], [0.8, 0.7]], dtype=np.float32),
            },
            "actions": expert_actions,
            "dones": np.asarray([False, True]),
            "rewards": np.asarray([0.0, 1.0], dtype=np.float32),
        }

        transitions, stats = _prepare_demo_transitions(
            demo=demo,
            episode_id=3,
            image_keys=("image", "wrist_image"),
        )

        self.assertEqual(len(transitions), 2)
        self.assertTrue(np.allclose(transitions[0]["actions"], expert_actions[0]))
        self.assertTrue(np.allclose(transitions[1]["actions"], expert_actions[1]))
        self.assertEqual(transitions[1]["masks"], 0.0)
        self.assertEqual(stats["steps_written"], 2)


if __name__ == "__main__":
    unittest.main()
