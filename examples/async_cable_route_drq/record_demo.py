#!/usr/bin/env python3

import pickle as pkl

import gym
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name")
flags.DEFINE_string("output", "demo.pkl", "Output demo path")
flags.DEFINE_integer("episodes", 5, "Number of episodes")
flags.DEFINE_integer("max_steps", 500, "Max steps per episode")


def main(_):
    env = gym.make(FLAGS.env)
    trajectories = []

    for _ in range(FLAGS.episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < FLAGS.max_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            trajectories.append(
                {
                    "observations": obs,
                    "actions": action,
                    "next_observations": next_obs,
                    "rewards": np.asarray(reward, dtype=np.float32),
                    "masks": np.asarray(1.0 - float(terminated), dtype=np.float32),
                    "dones": bool(terminated or truncated),
                }
            )
            obs = next_obs
            done = terminated or truncated
            steps += 1

    with open(FLAGS.output, "wb") as f:
        pkl.dump(trajectories, f)

    print(f"Saved {len(trajectories)} transitions to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
