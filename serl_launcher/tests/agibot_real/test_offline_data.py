from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

try:
    from omegaconf import OmegaConf
    from serl_torch.examples.agibot_real.config import parse_train_cfg
    from serl_torch.examples.agibot_real.env.offline_data import (
        load_prepared_offline_replay,
    )
    from serl_torch.examples.agibot_real.env.offline_data import (
        prepare_reference_episode_transitions,
    )
    from serl_torch.examples.agibot_real.env.offline_data import (
        resolve_and_validate_prepared_paths,
    )
    from serl_torch.examples.agibot_real.env.offline_data import (
        resolve_prepared_episode_files,
    )
    from serl_launcher.residual.typed_action import ResidualActionSpec
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

from serl_torch.examples.agibot_real.env.offline_data import (
    _coerce_lerobot_state_action_vector,
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


def _make_raw_obs() -> dict[str, object]:
    return {
        "state/pose": [0.0] * 14,
        "image/head": [[[0, 0, 0]] * 8 for _ in range(8)],
        "image/left_wrist": [[[0, 0, 0]] * 8 for _ in range(8)],
        "image/right_wrist": [[[0, 0, 0]] * 8 for _ in range(8)],
    }


class AgiBotLeRobotRawDataTest(unittest.TestCase):
    def test_openpi_14d_vector_is_used_as_is(self) -> None:
        vector = list(range(14))
        result = _coerce_lerobot_state_action_vector(
            vector,
            column="actions",
            source_path=Path("/tmp/episode.parquet"),
            arm_layout="dual_arm",
        )
        self.assertEqual(result.tolist(), [float(value) for value in vector])

    def test_openpi_14d_vector_can_be_projected_to_right_arm(self) -> None:
        vector = list(range(14))
        result = _coerce_lerobot_state_action_vector(
            vector,
            column="actions",
            source_path=Path("/tmp/episode.parquet"),
            arm_layout="right_arm",
        )
        self.assertEqual(result.tolist(), [float(value) for value in range(7, 14)])

    def test_single_arm_7d_vector_is_kept_for_right_arm(self) -> None:
        vector = list(range(7))
        result = _coerce_lerobot_state_action_vector(
            vector,
            column="actions",
            source_path=Path("/tmp/episode.parquet"),
            arm_layout="right_arm",
        )
        self.assertEqual(result.tolist(), [float(value) for value in vector])

    def test_joyra_30d_vector_uses_last_14_then_projects(self) -> None:
        vector = list(range(30))
        result = _coerce_lerobot_state_action_vector(
            vector,
            column="action",
            source_path=Path("/tmp/episode.parquet"),
            arm_layout="right_arm",
        )
        self.assertEqual(result.tolist(), [float(value) for value in range(23, 30)])

    def test_rejects_7d_vector_for_dual_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual_arm"):
            _coerce_lerobot_state_action_vector(
                [0.0] * 7,
                column="action",
                source_path=Path("/tmp/episode.parquet"),
                arm_layout="dual_arm",
            )

    def test_rejects_unexpected_vector_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "7D single-arm data"):
            _coerce_lerobot_state_action_vector(
                [0.0] * 16,
                column="action",
                source_path=Path("/tmp/episode.parquet"),
                arm_layout="right_arm",
            )


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
                    "expert_reference_scale": (
                        cfg.offline.prepare.expert_reference_scale
                    ),
                    "clip_residual_to_unit": (
                        cfg.offline.prepare.clip_residual_to_unit
                    ),
                    "filter_unrepresentable_steps": (
                        cfg.offline.prepare.filter_unrepresentable_steps
                    ),
                    "image_keys": list(cfg.obs.image_keys),
                    "vector_obs_keys": list(cfg.obs.vector_obs_keys),
                    "arm_layout": cfg.env.arm_layout,
                    "robot_action_dim": cfg.env.robot_action_dim,
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
            self.assertEqual(stats["steps_loaded"], 1)
            self.assertEqual(stats["load_errors"], 0)
            self.assertEqual(len(replay_buffer.inserted), 1)

    def test_prepare_reference_episode_filters_unrepresentable_steps(self) -> None:
        cfg = _train_cfg_with_prepared_path("/tmp/prepared")
        cfg = dataclasses.replace(
            cfg,
            offline=dataclasses.replace(
                cfg.offline,
                prepare=dataclasses.replace(
                    cfg.offline.prepare,
                    filter_unrepresentable_steps=True,
                ),
            ),
        )
        action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=cfg.env.action_dim)
        base_chunk = [[0.0] * cfg.env.action_dim for _ in range(cfg.residual.chunk_horizon)]
        raw_steps = [
            {
                "observations": _make_raw_obs(),
                "next_observations": _make_raw_obs(),
                "expert_action": [2.0] + ([0.0] * (cfg.env.action_dim - 1)),
                "base_chunk": base_chunk,
                "reward": 0.0,
                "done": False,
            },
            {
                "observations": _make_raw_obs(),
                "expert_action": [0.0] * cfg.env.action_dim,
                "base_chunk": base_chunk,
                "reward": 1.0,
                "done": True,
            },
        ]

        transitions, episode_stats = prepare_reference_episode_transitions(
            raw_steps=raw_steps,
            episode_id=7,
            task_prompt=cfg.task.prompt,
            action_spec=action_spec,
            image_keys=cfg.obs.image_keys,
            base_policy=object(),
            expert_reference_scale=cfg.offline.prepare.expert_reference_scale,
            clip_residual_to_unit=cfg.offline.prepare.clip_residual_to_unit,
            filter_unrepresentable_steps=cfg.offline.prepare.filter_unrepresentable_steps,
            source_path=Path("/tmp/reference_episode.pkl"),
            arm_layout=cfg.env.arm_layout,
        )

        self.assertEqual(episode_stats["steps_total"], 2)
        self.assertEqual(episode_stats["steps_unrepresentable"], 1)
        self.assertEqual(episode_stats["steps_filtered"], 1)
        self.assertEqual(episode_stats["steps_written"], 1)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["episode_id"], 7)
        self.assertEqual(transitions[0]["episode_step"], 1)
        self.assertTrue(transitions[0]["dones"])

    def test_prepare_reference_episode_right_arm_writes_7d_replay(self) -> None:
        cfg = _train_cfg_with_prepared_path("/tmp/prepared")
        cfg = dataclasses.replace(
            cfg,
            env=dataclasses.replace(
                cfg.env,
                arm_layout="right_arm",
                action_dim=7,
                robot_action_dim=14,
            ),
            policy=dataclasses.replace(cfg.policy, action_layout="right_arm"),
            residual=dataclasses.replace(
                cfg.residual,
                action_mask=(True,) * 7,
                action_limits=(1.0,) * 7,
            ),
        )
        action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=cfg.env.action_dim)
        base_chunk = [[0.0] * cfg.env.action_dim for _ in range(cfg.residual.chunk_horizon)]
        raw_steps = [
            {
                "observations": _make_raw_obs(),
                "expert_action": [0.0] * 14,
                "base_chunk": base_chunk,
                "reward": 1.0,
                "done": True,
            },
        ]

        transitions, _episode_stats = prepare_reference_episode_transitions(
            raw_steps=raw_steps,
            episode_id=9,
            task_prompt=cfg.task.prompt,
            action_spec=action_spec,
            image_keys=cfg.obs.image_keys,
            base_policy=object(),
            expert_reference_scale=cfg.offline.prepare.expert_reference_scale,
            clip_residual_to_unit=cfg.offline.prepare.clip_residual_to_unit,
            filter_unrepresentable_steps=False,
            source_path=Path("/tmp/reference_episode.pkl"),
            arm_layout=cfg.env.arm_layout,
        )

        self.assertEqual(np.asarray(transitions[0]["actions"]).shape, (7,))
        self.assertEqual(
            transitions[0]["observations"]["robot_proprio"].shape,
            (1, 7),
        )
        self.assertEqual(
            transitions[0]["observations"]["base_action_chunk"].shape[-1],
            7,
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
