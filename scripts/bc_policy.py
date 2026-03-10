#!/usr/bin/env python3

import time
from copy import deepcopy

import gym
import numpy as np
from absl import app, flags
from tqdm import tqdm

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
    populate_data_store_with_z_axis_only,
)
from serl_launcher.utils.checkpoint_utils import load_agent_checkpoint, save_agent_checkpoint
from serl_launcher.utils.launcher import make_bc_agent, make_wandb_logger
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "FrankaEnv-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "bc", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("max_steps", 10000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_bool("remove_xy", False, "Use z-axis-only state preprocessing.")
flags.DEFINE_string("encoder_type", "resnet", "Encoder type.")
flags.DEFINE_multi_string("demo_paths", None, "Paths to demo pickles.")
flags.DEFINE_string("checkpoint_path", "./checkpoints_bc", "Checkpoint directory.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Evaluate checkpoint step. 0 means train mode.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of eval trajectories.")
flags.DEFINE_boolean("debug", False, "Disable wandb logging.")


def _build_env():
    env = gym.make(FLAGS.env)

    try:
        from franka_env.envs.relative_env import RelativeFrame
        from franka_env.envs.wrappers import GripperCloseEnv, Quat2EulerWrapper, ZOnlyWrapper

        env = GripperCloseEnv(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        if FLAGS.remove_xy:
            env = ZOnlyWrapper(env)
    except Exception:
        pass

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    return env


def _collect_image_keys(env):
    if isinstance(env.observation_space, gym.spaces.Dict):
        return [k for k in env.observation_space.keys() if k != "state"]
    return []


def train(agent: BCAgent, env, image_keys):
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    replay_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        FLAGS.replay_buffer_capacity,
        image_keys=image_keys,
    )

    if FLAGS.demo_paths:
        if FLAGS.remove_xy:
            replay_buffer = populate_data_store_with_z_axis_only(replay_buffer, FLAGS.demo_paths)
        else:
            replay_buffer = populate_data_store(replay_buffer, FLAGS.demo_paths)

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=None,
    )

    for step in tqdm(range(FLAGS.max_steps), desc="bc-train"):
        batch = next(replay_iterator)
        agent, info = agent.update(batch)

        if wandb_logger is not None:
            wandb_logger.log(info, step=step)

        if FLAGS.save_model and (step + 1) % 1000 == 0:
            save_agent_checkpoint(
                FLAGS.checkpoint_path,
                agent,
                step=step + 1,
                keep=100,
            )


def evaluate(agent: BCAgent, env):
    agent = load_agent_checkpoint(
        FLAGS.checkpoint_path,
        agent,
        step=None if FLAGS.eval_checkpoint_step == 0 else FLAGS.eval_checkpoint_step,
    )

    success = 0.0
    time_list = []

    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start = time.time()

        while not done:
            action = agent.sample_actions(observations=obs, argmax=True)
            next_obs, reward, terminated, truncated, _ = env.step(np.asarray(action))
            done = terminated or truncated
            obs = next_obs
            if done:
                success += float(reward)
                time_list.append(time.time() - start)

        print(f"episode={episode + 1}/{FLAGS.eval_n_trajs} reward={reward}")

    print(f"success_rate={success / FLAGS.eval_n_trajs:.4f}")
    if time_list:
        print(f"avg_time={np.mean(time_list):.4f}")


def main(_):
    np.random.seed(FLAGS.seed)
    env = _build_env()
    image_keys = _collect_image_keys(env)

    agent: BCAgent = make_bc_agent(
        FLAGS.seed,
        deepcopy(env.observation_space.sample()),
        env.action_space.sample(),
        encoder_type=FLAGS.encoder_type,
        image_keys=image_keys,
    )

    if FLAGS.eval_checkpoint_step == 0:
        train(agent, env, image_keys)
    else:
        evaluate(agent, env)


if __name__ == "__main__":
    app.run(main)
