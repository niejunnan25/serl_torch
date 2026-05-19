from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

from serl_launcher.data.offline_prepared import build_residual_prepared_fingerprint
from serl_launcher.data.offline_prepared import build_residual_training_signature
from serl_launcher.data.offline_prepared import extract_residual_manifest_signature
from serl_launcher.data.offline_prepared import format_residual_alpha_token
from serl_launcher.data.offline_prepared import load_prepared_offline_replay
from serl_launcher.data.offline_prepared import resolve_prepared_episode_files
from serl_launcher.data.offline_prepared import resolve_residual_prepared_dir
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


def test_residual_prepared_metadata_helpers_build_shared_signature() -> None:
    fingerprint = build_residual_prepared_fingerprint(
        format_version="format_v1",
        task_key="suite_task_1",
        task_description="pick object",
        policy_backend_type="openpi",
        policy_backend_id="openpi:v1",
        chunk_horizon=30,
        action_dim=7,
        alpha=0.2,
        action_mask=(True, False),
        action_limits=(1, 2),
        clip_gripper=True,
        expert_reference_scale=1.5,
        clip_residual_to_unit=False,
        filter_unrepresentable_steps=True,
        image_keys=("agentview", "wrist"),
        vector_obs_keys=("robot",),
        raw_dataset_path="/tmp/raw.hdf5",
        extra_fields={"raw_source_format": "reference"},
    )

    assert fingerprint["format_version"] == "format_v1"
    assert fingerprint["task_key"] == "suite_task_1"
    assert fingerprint["task_description"] == "pick object"
    assert fingerprint["action_mask"] == [True, False]
    assert fingerprint["action_limits"] == [1.0, 2.0]
    assert fingerprint["image_keys"] == ["agentview", "wrist"]
    assert fingerprint["vector_obs_keys"] == ["robot"]
    assert fingerprint["raw_dataset_path"] == "/tmp/raw.hdf5"
    assert fingerprint["raw_source_format"] == "reference"

    expected_signature = build_residual_training_signature(
        task_key="suite_task_1",
        policy_backend_type="openpi",
        policy_backend_id="openpi:v1",
        chunk_horizon=30,
        action_dim=7,
        alpha=0.2,
        action_mask=(True, False),
        action_limits=(1, 2),
        clip_gripper=True,
        expert_reference_scale=1.5,
        clip_residual_to_unit=False,
        filter_unrepresentable_steps=True,
        image_keys=("agentview", "wrist"),
        vector_obs_keys=("robot",),
    )

    assert extract_residual_manifest_signature({"fingerprint": fingerprint}) == (
        expected_signature
    )
    assert extract_residual_manifest_signature({}) is None


def test_residual_prepared_dir_uses_shared_chunk_alpha_naming() -> None:
    prepared_dir = resolve_residual_prepared_dir(
        output_root="offline_data",
        task_key="suite_task_1",
        policy_backend="joyra:office",
        chunk_horizon=30,
        alpha=0.2,
    )

    assert format_residual_alpha_token(0.2) == "0p2"
    assert prepared_dir.name == "joyra_office_chunk30_alpha0p2"
    assert prepared_dir.parent.name == "suite_task_1"


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


def test_load_prepared_offline_replay_filters_min_episode_step() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_dir = Path(tmpdir) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        transitions = [
            {"episode_id": 0, "episode_step": 0, "value": "skip"},
            {"episode_id": 0, "episode_step": 30, "value": "keep"},
        ]
        episode_path = prepared_dir / "episode_000000.pkl"
        with open(episode_path, "wb") as fp:
            pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(prepared_dir / "manifest.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "fingerprint": {},
                    "episode_files": [episode_path.name],
                    "prepare_stats": {"steps_written": 2},
                },
                fp,
            )

        replay_buffer = _FakeReplayBuffer()
        stats = load_prepared_offline_replay(
            replay_buffer=replay_buffer,
            prepared_paths=(prepared_dir,),
            logger=_FakeLogger(),
            min_episode_step=30,
        )

        assert stats["steps_loaded"] == 1
        assert stats["steps_skipped_min_episode_step"] == 1
        assert replay_buffer.inserted == [transitions[1]]


def test_load_prepared_offline_replay_filters_active_step_ranges() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_dir = Path(tmpdir) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        transitions = [
            {
                "episode_id": 0,
                "episode_step": 0,
                "masks": 1.0,
                "value": "skip_early",
            },
            {
                "episode_id": 0,
                "episode_step": 30,
                "masks": 1.0,
                "value": "keep_stage1",
            },
            {
                "episode_id": 0,
                "episode_step": 74,
                "masks": 1.0,
                "value": "keep_stage1_boundary",
            },
            {
                "episode_id": 0,
                "episode_step": 90,
                "masks": 1.0,
                "value": "skip_middle",
            },
            {
                "episode_id": 0,
                "episode_step": 120,
                "masks": 1.0,
                "value": "keep_stage2",
            },
            {
                "episode_id": 0,
                "episode_step": 159,
                "masks": 1.0,
                "value": "keep_stage2_boundary",
            },
            {
                "episode_id": 0,
                "episode_step": 160,
                "masks": 1.0,
                "value": "skip_end",
            },
        ]
        episode_path = prepared_dir / "episode_000000.pkl"
        with open(episode_path, "wb") as fp:
            pickle.dump(transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(prepared_dir / "manifest.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "fingerprint": {},
                    "episode_files": [episode_path.name],
                    "prepare_stats": {"steps_written": len(transitions)},
                },
                fp,
            )

        replay_buffer = _FakeReplayBuffer()
        stats = load_prepared_offline_replay(
            replay_buffer=replay_buffer,
            prepared_paths=(prepared_dir,),
            logger=_FakeLogger(),
            active_step_ranges=((30, 75), (110, 160)),
        )

        assert stats["steps_loaded"] == 4
        assert stats["steps_skipped_active_step_ranges"] == 3
        assert stats["steps_terminalized_active_step_ranges"] == 2
        assert [item["value"] for item in replay_buffer.inserted] == [
            "keep_stage1",
            "keep_stage1_boundary",
            "keep_stage2",
            "keep_stage2_boundary",
        ]
        assert [item["masks"] for item in replay_buffer.inserted] == [
            1.0,
            0.0,
            1.0,
            0.0,
        ]
        assert transitions[2]["masks"] == 1.0
        assert transitions[5]["masks"] == 1.0


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
