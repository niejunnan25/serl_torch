#!/usr/bin/env python3

import pickle as pkl
from typing import List

import gym
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags

from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name")
flags.DEFINE_string("positive_demo_path", None, "Pickle file with positive transitions")
flags.DEFINE_string("negative_demo_path", None, "Pickle file with negative transitions")
flags.DEFINE_string("checkpoint_path", "reward_classifier.pt", "Output checkpoint path")
flags.DEFINE_integer("steps", 5000, "Optimization steps")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_float("lr", 1e-4, "Learning rate")


def _stack_obs(items: List[dict]):
    if isinstance(items[0], dict):
        return {k: _stack_obs([x[k] for x in items]) for k in items[0].keys()}
    return np.stack(items)


def _sample_batch(pos_obs, neg_obs, batch_size):
    half = batch_size // 2
    pos_idx = np.random.randint(0, len(pos_obs), size=half)
    neg_idx = np.random.randint(0, len(neg_obs), size=half)

    batch_pos = [pos_obs[i] for i in pos_idx]
    batch_neg = [neg_obs[i] for i in neg_idx]

    obs = _stack_obs(batch_pos + batch_neg)
    labels = np.concatenate(
        [np.ones((half,), dtype=np.float32), np.zeros((half,), dtype=np.float32)],
        axis=0,
    )
    return obs, labels


def _to_torch(data, device):
    if isinstance(data, dict):
        return {k: _to_torch(v, device) for k, v in data.items()}
    return torch.as_tensor(data, device=device)


def main(_):
    if FLAGS.positive_demo_path is None or FLAGS.negative_demo_path is None:
        raise ValueError("Both --positive_demo_path and --negative_demo_path are required")

    with open(FLAGS.positive_demo_path, "rb") as f:
        pos_traj = pkl.load(f)
    with open(FLAGS.negative_demo_path, "rb") as f:
        neg_traj = pkl.load(f)

    pos_obs = [t["next_observations"] for t in pos_traj]
    neg_obs = [t["next_observations"] for t in neg_traj]

    env = gym.make(FLAGS.env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if k != "state"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = create_classifier(
        key=0,
        sample=env.observation_space.sample(),
        image_keys=image_keys,
        device=device,
    )
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=FLAGS.lr)

    for step in range(FLAGS.steps):
        obs_np, labels_np = _sample_batch(pos_obs, neg_obs, FLAGS.batch_size)
        obs = _to_torch(obs_np, device)
        labels = _to_torch(labels_np, device)

        logits = classifier(obs, train=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
            print(f"step={step} loss={loss.item():.4f} acc={acc.item():.4f}")

    torch.save({"model": classifier.state_dict()}, FLAGS.checkpoint_path)
    print(f"Saved classifier checkpoint to {FLAGS.checkpoint_path}")


if __name__ == "__main__":
    app.run(main)
