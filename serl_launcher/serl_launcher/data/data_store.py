from threading import Lock
from typing import Iterable, Optional, TypeVar

import gym

from serl_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.step_window_replay_buffer import (
    MemoryEfficientStepWindowReplayBuffer,
)
from serl_launcher.data.step_window_replay_buffer import StepWindowReplayBuffer

from agentlace.data.data_store import DataStoreBase

try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print(
        "rlds logger is not installed, install it if required: "
        "https://github.com/rail-berkeley/oxe_envlogger "
    )
    RLDSLogger = TypeVar("RLDSLogger")


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)

            if self._logger:
                if self.step_type in {RLDSStepType.TERMINATION, RLDSStepType.TRUNCATION}:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["next_observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=image_keys,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)

            if self._logger:
                if self.step_type in {RLDSStepType.TERMINATION, RLDSStepType.TRUNCATION}:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["next_observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class StepWindowReplayBufferDataStore(StepWindowReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        window_size: int,
        discount: float,
        sample_stride: int = 1,
        require_full_window: bool = False,
        next_observation_space: Optional[gym.Space] = None,
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        StepWindowReplayBuffer.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            window_size=window_size,
            discount=discount,
            sample_stride=sample_stride,
            require_full_window=require_full_window,
            next_observation_space=next_observation_space,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)

            if self._logger:
                if self.step_type in {RLDSStepType.TERMINATION, RLDSStepType.TRUNCATION}:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["next_observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_count

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientStepWindowReplayBufferDataStore(
    MemoryEfficientStepWindowReplayBuffer,
    DataStoreBase,
):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        window_size: int,
        discount: float,
        sample_stride: int = 1,
        require_full_window: bool = False,
        next_observation_space: Optional[gym.Space] = None,
        image_keys: Iterable[str] = ("image",),
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        MemoryEfficientStepWindowReplayBuffer.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            window_size=window_size,
            discount=discount,
            sample_stride=sample_stride,
            require_full_window=require_full_window,
            next_observation_space=next_observation_space,
            pixel_keys=tuple(image_keys),
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)

            if self._logger:
                if self.step_type in {RLDSStepType.TERMINATION, RLDSStepType.TRUNCATION}:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["next_observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_count

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


def populate_data_store(data_store: DataStoreBase, demos_path: str):
    import pickle as pkl

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(data_store: DataStoreBase, demos_path: str):
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
