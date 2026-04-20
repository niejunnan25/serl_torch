from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

from serl_launcher.data.offline_prepared import load_prepared_offline_replay
from serl_launcher.data.offline_prepared import resolve_prepared_episode_files
from serl_launcher.data.offline_prepared import validate_prepared_paths
from serl_launcher.residual.expert_projection import project_expert_action


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


def test_validate_prepared_paths_rejects_manifestless_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_dir = Path(tmpdir) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        with open(prepared_dir / "episode_000000.pkl", "wb") as fp:
            pickle.dump([], fp, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            validate_prepared_paths(
                (prepared_dir,),
                expected_signature={"task_key": "task_a"},
                manifest_signature_fn=lambda manifest: (
                    None if manifest is None else manifest.get("fingerprint", None)
                ),
            )
        except ValueError as exc:
            assert "must contain manifest.json" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected validate_prepared_paths to reject missing manifest")


def test_validate_and_load_prepared_replay_from_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_dir = Path(tmpdir) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        episode_path = prepared_dir / "episode_000000.pkl"
        transitions = [
            {
                "episode_id": 0,
                "episode_step": 0,
                "observations": {"robot_proprio": [[0.0]]},
                "actions": [0.0, 0.0],
                "next_observations": {"robot_proprio": [[0.0]]},
                "rewards": 1.0,
                "masks": 0.0,
                "dones": True,
            }
        ]
        with open(episode_path, "wb") as fp:
            pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)

        fingerprint = {"task_key": "task_a", "alpha": 0.1}
        with open(prepared_dir / "manifest.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "episode_files": [episode_path.name],
                    "prepare_stats": {
                        "steps_total": 1,
                        "steps_unrepresentable": 0,
                        "steps_filtered": 0,
                        "steps_written": 1,
                    },
                },
                fp,
            )

        resolution = validate_prepared_paths(
            (prepared_dir,),
            expected_signature=fingerprint,
            manifest_signature_fn=lambda manifest: (
                None if manifest is None else manifest.get("fingerprint", None)
            ),
        )
        assert resolution.prepared_paths == (prepared_dir.resolve(),)
        assert resolution.manifest_paths == ((prepared_dir / "manifest.json").resolve(),)

        resolved_files = resolve_prepared_episode_files(
            (prepared_dir, prepared_dir / "manifest.json"),
        )
        assert resolved_files == [episode_path.resolve()]

        replay_buffer = _FakeReplayBuffer()
        stats = load_prepared_offline_replay(
            replay_buffer=replay_buffer,
            prepared_paths=resolution.prepared_paths,
            logger=_FakeLogger(),
        )
        assert stats["episodes_loaded"] == 1
        assert stats["steps_loaded"] == 1
        assert stats["load_errors"] == 0
        assert replay_buffer.inserted == transitions


def test_project_expert_action_clips_and_reports_unrepresentable() -> None:
    action_spec = SimpleNamespace(
        alpha=0.5,
        control_indices=(0,),
        residual_limits=(1.0,),
        clip_gripper=False,
    )
    projected, clipped_values, step_unrepresentable = project_expert_action(
        expert_action=[2.0, 0.0],
        base_action=[0.0, 0.0],
        action_spec=action_spec,
        expert_reference_scale=1.0,
        clip_residual_to_unit=True,
    )
    assert clipped_values == 1
    assert step_unrepresentable is True
    assert projected.shape == (2,)
    assert projected[0] == 0.5
