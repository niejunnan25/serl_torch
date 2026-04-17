from __future__ import annotations

import copy

import gym
import numpy as np

from serl_launcher.data.batch_ops import pack_transition_batch
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.data.data_store import StepWindowReplayBufferDataStore


def _assert_nested_equal(lhs, rhs) -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict)
        assert set(lhs.keys()) == set(rhs.keys())
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
        return
    np.testing.assert_array_equal(lhs, rhs)


def _nested_active_prefix(data, size: int):
    if isinstance(data, dict):
        return {key: _nested_active_prefix(value, size) for key, value in data.items()}
    return np.array(data[: int(size)], copy=True)


def _simple_observation_space():
    return gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        }
    )


def _pixel_observation_space():
    return gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            ),
            "pixels": gym.spaces.Box(
                low=0,
                high=255,
                shape=(2, 4, 4, 3),
                dtype=np.uint8,
            ),
        }
    )


def _action_space():
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _make_transition(step: int, *, done: bool = False) -> dict:
    return {
        "observations": {
            "state": np.asarray([step, step + 0.1, step + 0.2], dtype=np.float32),
        },
        "next_observations": {
            "state": np.asarray(
                [step + 1, step + 1.1, step + 1.2],
                dtype=np.float32,
            ),
        },
        "actions": np.asarray([step, -step], dtype=np.float32),
        "rewards": np.float32(step + 0.5),
        "masks": np.float32(0.0 if done else 1.0),
        "dones": bool(done),
    }


def _make_step_window_transition(
    step: int,
    *,
    episode_id: int = 0,
    done: bool = False,
) -> dict:
    transition = _make_transition(step, done=done)
    transition["episode_id"] = np.int64(episode_id)
    transition["episode_step"] = np.int32(step)
    return transition


def _make_memory_transition(step: int, *, done: bool = False) -> dict:
    observations_pixels = np.stack(
        [
            np.full((4, 4, 3), fill_value=step + offset, dtype=np.uint8)
            for offset in range(2)
        ],
        axis=0,
    )
    next_observations_pixels = np.stack(
        [
            np.full((4, 4, 3), fill_value=step + offset + 1, dtype=np.uint8)
            for offset in range(2)
        ],
        axis=0,
    )
    transition = _make_transition(step, done=done)
    transition["observations"]["pixels"] = observations_pixels
    transition["next_observations"]["pixels"] = next_observations_pixels
    return transition


def _make_memory_step_window_transition(
    step: int,
    *,
    episode_id: int = 0,
    done: bool = False,
) -> dict:
    transition = _make_memory_transition(step, done=done)
    transition["episode_id"] = np.int64(episode_id)
    transition["episode_step"] = np.int32(step)
    return transition


def test_replay_buffer_batch_insert_matches_single_insert() -> None:
    observation_space = _simple_observation_space()
    action_space = _action_space()
    transitions = [_make_transition(step) for step in range(6)]

    single = ReplayBufferDataStore(observation_space, action_space, capacity=8)
    batched = ReplayBufferDataStore(observation_space, action_space, capacity=8)

    for transition in transitions:
        single.insert(copy.deepcopy(transition))
    batched.batch_insert(pack_transition_batch(copy.deepcopy(transitions)))

    _assert_nested_equal(
        _nested_active_prefix(single.dataset_dict, len(single)),
        _nested_active_prefix(batched.dataset_dict, len(batched)),
    )
    assert int(single._insert_index) == int(batched._insert_index)
    assert int(single._size) == int(batched._size)


def test_step_window_batch_insert_matches_single_insert() -> None:
    observation_space = _simple_observation_space()
    action_space = _action_space()
    transitions = [
        _make_step_window_transition(step, done=(step == 4))
        for step in range(5)
    ]

    single = StepWindowReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        window_size=3,
        discount=0.99,
    )
    batched = StepWindowReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        window_size=3,
        discount=0.99,
    )

    for transition in transitions:
        single.insert(copy.deepcopy(transition))
    batched.batch_insert(pack_transition_batch(copy.deepcopy(transitions)))

    _assert_nested_equal(
        _nested_active_prefix(single.dataset_dict, len(single)),
        _nested_active_prefix(batched.dataset_dict, len(batched)),
    )
    np.testing.assert_array_equal(single._episode_ids[: len(single)], batched._episode_ids[: len(batched)])
    np.testing.assert_array_equal(single._episode_steps[: len(single)], batched._episode_steps[: len(batched)])
    np.testing.assert_array_equal(single._step_ids[: len(single)], batched._step_ids[: len(batched)])
    assert list(single._candidate_start_step_ids) == list(batched._candidate_start_step_ids)
    window_single = single._build_transition(0)
    window_batched = batched._build_transition(0)
    _assert_nested_equal(window_single, window_batched)


def test_memory_efficient_replay_batch_insert_matches_single_insert() -> None:
    observation_space = _pixel_observation_space()
    action_space = _action_space()
    transitions = [_make_memory_transition(step, done=(step == 3)) for step in range(4)]

    single = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        image_keys=("pixels",),
    )
    batched = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        image_keys=("pixels",),
    )

    for transition in transitions:
        single.insert(copy.deepcopy(transition))
    batched.batch_insert(pack_transition_batch(copy.deepcopy(transitions)))

    _assert_nested_equal(
        _nested_active_prefix(single.dataset_dict, len(single)),
        _nested_active_prefix(batched.dataset_dict, len(batched)),
    )
    np.testing.assert_array_equal(single._is_correct_index, batched._is_correct_index)
    assert bool(single._first) == bool(batched._first)
    assert int(single._insert_index) == int(batched._insert_index)


def test_memory_efficient_step_window_batch_insert_matches_single_insert() -> None:
    observation_space = _pixel_observation_space()
    action_space = _action_space()
    transitions = [
        _make_memory_step_window_transition(step, done=(step == 4))
        for step in range(5)
    ]

    single = MemoryEfficientStepWindowReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        window_size=3,
        discount=0.99,
        image_keys=("pixels",),
    )
    batched = MemoryEfficientStepWindowReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=16,
        window_size=3,
        discount=0.99,
        image_keys=("pixels",),
    )

    for transition in transitions:
        single.insert(copy.deepcopy(transition))
    batched.batch_insert(pack_transition_batch(copy.deepcopy(transitions)))

    _assert_nested_equal(
        _nested_active_prefix(single.dataset_dict, len(single)),
        _nested_active_prefix(batched.dataset_dict, len(batched)),
    )
    np.testing.assert_array_equal(single._episode_ids[: len(single)], batched._episode_ids[: len(batched)])
    np.testing.assert_array_equal(single._episode_steps[: len(single)], batched._episode_steps[: len(batched)])
    np.testing.assert_array_equal(single._step_ids[: len(single)], batched._step_ids[: len(batched)])
    np.testing.assert_array_equal(
        single._has_explicit_next_pixels[: len(single)],
        batched._has_explicit_next_pixels[: len(batched)],
    )
    for pixel_key in single.pixel_keys:
        np.testing.assert_array_equal(
            single._explicit_next_pixels[pixel_key][: len(single)],
            batched._explicit_next_pixels[pixel_key][: len(batched)],
        )
    window_single = single._build_transition(0)
    window_batched = batched._build_transition(0)
    _assert_nested_equal(window_single, window_batched)
