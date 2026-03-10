from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from gym.utils import seeding

DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            if dataset_len != item_len:
                raise ValueError("Inconsistent item lengths in the dataset")
        else:
            raise TypeError("Unsupported type")
    return int(dataset_len or 0)


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    out = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            out[k] = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            out[k] = v[index]
        else:
            raise TypeError("Unsupported type")
    return out


def _sample(dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray):
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    if isinstance(dataset_dict, dict):
        return {k: _sample(v, indx) for k, v in dataset_dict.items()}
    raise TypeError("Unsupported type")


class Dataset:
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self):
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> DatasetDict:
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        keys = list(keys or self.dataset_dict.keys())
        batch = {}
        for k in keys:
            v = self.dataset_dict[k]
            batch[k] = _sample(v, indx) if isinstance(v, dict) else v[indx]
        return batch

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        if not (0 < ratio < 1):
            raise ValueError("ratio must be in (0,1)")

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self):
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(self, take_top: Optional[float] = None, threshold: Optional[float] = None):
        if not ((take_top is None) ^ (threshold is None)):
            raise ValueError("Specify exactly one of take_top or threshold")

        episode_starts, episode_ends, episode_returns = self._trajectory_boundaries_and_returns()

        if take_top is not None:
            threshold = np.percentile(episode_returns, 100 - take_top)

        bool_indx = np.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)
        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(episode_returns)
        self.dataset_dict["rewards"] *= scaling
