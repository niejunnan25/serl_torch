from collections import deque
from typing import Optional

import gym
import gym.spaces
import numpy as np


def _tree_stack(list_of_dicts):
    out = {}
    for key in list_of_dicts[0].keys():
        values = [item[key] for item in list_of_dicts]
        if isinstance(values[0], dict):
            out[key] = _tree_stack(values)
        else:
            out[key] = np.stack(values)
    return out


def stack_obs(obs):
    return _tree_stack(list(obs))


def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
    raise TypeError(f"Unsupported space type: {type(space)}")

class ChunkingWrapper(gym.Wrapper):
    """Enables observation histories and receding horizon control."""

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(self.env.action_space, self.act_exec_horizon)

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return stack_obs(self.current_obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info
