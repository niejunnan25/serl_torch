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

LIBERO_OFFLINE_IMPORTS_AVAILABLE = False
if OmegaConf is not None:
    try:
        from serl_torch.examples.libero.config import parse_train_cfg
        from serl_torch.examples.libero.env.offline_data import (
            _precompute_base_chunks_for_steps,
            resolve_and_validate_prepared_paths,
        )
    except ModuleNotFoundError:  # pragma: no cover - environment-dependent
        parse_train_cfg = None
        _precompute_base_chunks_for_steps = None
        resolve_and_validate_prepared_paths = None
    else:
        LIBERO_OFFLINE_IMPORTS_AVAILABLE = True


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
        / "train_residual_step.yaml"
    )
    cfg.runtime.role = "learner"
    cfg.offline.enabled = True
    cfg.offline.prepared_path = prepared_path
    return parse_train_cfg(cfg)


@unittest.skipIf(
    not LIBERO_OFFLINE_IMPORTS_AVAILABLE,
    "libero offline test dependencies are not installed",
)
class LiberoOfflineDataTest(unittest.TestCase):
    def test_precompute_base_chunks_uses_preparsed_policy_input_parts(self) -> None:
        class _FakePolicyClient:
            def infer(self, policy_input: object) -> tuple[np.ndarray, dict[str, object]]:
                del policy_input
                return np.full((2, 7), 0.5, dtype=np.float32), {}

        payload = {
            "actions": np.zeros((1, 7), dtype=np.float32),
            "agentview_rgb": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "eye_in_hand_rgb": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "ee_pos": np.zeros((1, 3), dtype=np.float32),
            "ee_ori": np.zeros((1, 3), dtype=np.float32),
            "gripper_states": np.zeros((1, 2), dtype=np.float32),
        }

        base_chunks = _precompute_base_chunks_for_steps(
            payload,
            task_prompt="pick up the block",
            policy_client=_FakePolicyClient(),
            chunk_horizon=2,
        )

        self.assertEqual(base_chunks.shape, (1, 2, 7))
        self.assertEqual(base_chunks.dtype, np.float32)

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

    def test_validate_prepared_offline_rejects_manifestless_episode_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_000000.pkl"
            with open(episode_path, "wb") as fp:
                pickle.dump([], fp, protocol=pickle.HIGHEST_PROTOCOL)

            cfg = _train_cfg_with_prepared_path(str(episode_path))
            with self.assertRaisesRegex(
                ValueError,
                "without manifest is no longer supported",
            ):
                resolve_and_validate_prepared_paths(
                    cfg,
                    logger=_FakeLogger(),
                )


if __name__ == "__main__":
    unittest.main()
