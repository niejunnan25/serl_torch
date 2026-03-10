#!/usr/bin/env python3

import gym
import numpy as np
from absl import app, flags

from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name")
flags.DEFINE_string("checkpoint_path", None, "Classifier checkpoint path")
flags.DEFINE_integer("step", 0, "Checkpoint step, 0 means latest")


def main(_):
    if FLAGS.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required")

    env = gym.make(FLAGS.env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [k for k in env.observation_space.keys() if k != "state"]
    sample = env.observation_space.sample()

    classifier = load_classifier_func(
        key=0,
        sample=sample,
        image_keys=image_keys,
        checkpoint_path=FLAGS.checkpoint_path,
        step=None if FLAGS.step == 0 else FLAGS.step,
    )

    obs, _ = env.reset()
    logits = classifier(obs)
    print("Classifier logits shape:", np.asarray(logits).shape)
    print("Classifier logits:", logits)


if __name__ == "__main__":
    app.run(main)
