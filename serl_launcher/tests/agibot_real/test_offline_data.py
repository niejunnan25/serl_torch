from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

if OmegaConf is not None:
    from serl_torch.examples.agibot_real.config import parse_train_cfg
    from serl_torch.examples.agibot_real.offline_data import load_prepared_offline_replay
    from serl_torch.examples.agibot_real.offline_data import (
        resolve_and_validate_prepared_paths,
    )
    from serl_torch.examples.agibot_real.offline_data import (
        resolve_prepared_episode_files,
    )


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.inserted: list[dict[str, object]] = []

    def insert(self, transition: dict[str, object]) -> None:
        self.inserted.append(dict(transition))


class _FakeLogger:
    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _train_cfg_with_prepared_path(prepared_path: str) -> object:
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "examples"
        / "agibot_real"
        / "configs"
        / "train_residual.yaml"
    )
    cfg.runtime.role = "learner"
    cfg.offline.enabled = True
    cfg.offline.prepared_path = prepared_path
    return parse_train_cfg(cfg)


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class AgiBotOfflineDataTest(unittest.TestCase):
    def test_validate_and_load_prepared_offline_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_dir = Path(tmpdir) / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            episode_path = prepared_dir / "episode_000000.pkl"
            transitions = [
                {
                    "episode_id": 0,
                    "episode_step": 0,
                    "observations": {"robot_proprio": [[0.0]]},
                    "actions": [0.0] * 14,
                    "next_observations": {"robot_proprio": [[0.0]]},
                    "rewards": 1.0,
                    "masks": 0.0,
                    "dones": True,
                }
            ]
            with open(episode_path, "wb") as fp:
                pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)

            cfg = _train_cfg_with_prepared_path(str(prepared_dir))
            manifest = {
                "fingerprint": {
                    "task_key": cfg.task.task_key,
                    "policy_backend_type": cfg.policy.type,
                    "policy_backend_id": cfg.policy.id,
                    "chunk_horizon": cfg.residual.chunk_horizon,
                    "action_dim": cfg.env.action_dim,
                    "alpha": cfg.residual.alpha,
                    "action_mask": list(cfg.residual.action_mask),
                    "action_limits": list(cfg.residual.action_limits),
                    "clip_gripper": cfg.residual.clip_gripper,
                    "image_keys": list(cfg.obs.image_keys),
                    "vector_obs_keys": list(cfg.obs.vector_obs_keys),
                },
                "episode_files": [episode_path.name],
            }
            with open(prepared_dir / "manifest.json", "w", encoding="utf-8") as fp:
                json.dump(manifest, fp)

            resolution = resolve_and_validate_prepared_paths(
                cfg,
                logger=_FakeLogger(),
            )
            self.assertEqual(resolution.prepared_paths, (prepared_dir.resolve(),))
            self.assertEqual(
                resolve_prepared_episode_files(resolution.prepared_paths),
                [episode_path.resolve()],
            )

            replay_buffer = _FakeReplayBuffer()
            stats = load_prepared_offline_replay(
                replay_buffer=replay_buffer,
                prepared_paths=resolution.prepared_paths,
                logger=_FakeLogger(),
            )
            self.assertEqual(stats["episodes_loaded"], 1)
            self.assertEqual(stats["inserted"], 1)
            self.assertEqual(len(replay_buffer.inserted), 1)

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
