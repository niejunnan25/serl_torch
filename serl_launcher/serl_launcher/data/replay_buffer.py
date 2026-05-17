import collections
from typing import Optional, Union

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym
import numpy as np
import torch

from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    if isinstance(obs_space, gym.spaces.Dict):
        return {k: _init_replay_dict(v, capacity) for k, v in obs_space.spaces.items()}
    raise TypeError(f"Unsupported space type: {type(obs_space)}")


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        if dataset_dict.keys() != data_dict.keys():
            raise ValueError((dataset_dict.keys(), data_dict.keys()))
        for k in dataset_dict:
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError("Unsupported dataset type")


def _to_torch(batch, device=None):
    if isinstance(batch, dict):
        return {k: _to_torch(v, device=device) for k, v in batch.items()}
    tensor = torch.from_numpy(batch) if isinstance(batch, np.ndarray) else torch.as_tensor(batch)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self):
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = None, device=None):
        sample_args = sample_args or {}
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(_to_torch(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
