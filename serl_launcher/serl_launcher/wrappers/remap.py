from typing import Any

import gym
import gym.spaces


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, v) for v in tree)
    return fn(tree)


class RemapWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_structure: Any):
        super().__init__(env)
        self.new_structure = new_structure

        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple([env.observation_space[v] for v in new_structure])
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict({k: env.observation_space[v] for k, v in new_structure.items()})
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported type {type(new_structure)}")

    def observation(self, observation):
        return _tree_map(lambda x: observation[x], self.new_structure)
