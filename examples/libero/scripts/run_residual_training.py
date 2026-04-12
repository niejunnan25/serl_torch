from __future__ import annotations

"""Reference-style LIBERO residual DRQ training script."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient
from agentlace.trainer import TrainerConfig
from agentlace.trainer import TrainerServer
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.policy.typed_factory import build_policy_client
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.config import LiberoTrainConfig
from serl_torch.examples.libero.config import cfg_to_log_payload
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.env.factory import _create_env as create_env
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input
from serl_torch.examples.libero.residual_observation import (
    build_chunk_residual_obs,
)
from serl_torch.examples.libero.residual_observation import (
    build_chunk_residual_observation_space,
)
from serl_torch.examples.libero.residual_observation import (
    build_chunk_residual_sample_obs,
)
from serl_torch.examples.libero.residual_observation import (
    prepare_base_actions_chunk,
)

FILL_WAIT_SLEEP_SEC = 1.0
LEARNER_IDLE_SLEEP_SEC = 1.0


def actor(cfg: LiberoTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    env = create_env(cfg, logger)
    task_prompt = str(env.task_description)
    policy_client = build_policy_client(cfg, logger=logger)
    policy_backend = describe_policy_backend(cfg)
    logger.info("Chunk policy backend: %s", policy_backend)

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)

    residual_alpha = residual_action_spec.alpha

    sample_obs = build_chunk_residual_sample_obs(
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
    )

    agent = create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=image_keys,
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )

    data_store = QueuedDataStore(cfg.runtime.data_store_queue_size)
    client = TrainerClient(
        "actor_env",
        cfg.runtime.trainer_host,
        TrainerConfig(  # pyright: ignore[reportCallIssue]
            port_number=cfg.runtime.trainer_port,
            broadcast_port=cfg.runtime.broadcast_port,
            request_types=["send-stats"],
        ),
        data_store,
        wait_for_server=True,
    )

    def update_actor(payload: dict[str, Any]) -> None:
        apply_checkpoint_payload_to_agent(
            agent,
            dict(payload),
            load_optimizers=False,
        )

    client.recv_network_callback(update_actor)

    timer = Timer()
    steps_per_update = cfg.training.steps_per_update
    log_period = cfg.training.log_period
    max_env_steps = cfg.training.max_env_steps
    env_seed = cfg.env.seed

    env_steps = 0
    episode_id = 0
    success_count = 0
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "residual",
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
    }

    try:
        while env_steps < max_env_steps:
            episode_id += 1
            reset_seed = env_seed
            init_episode_idx = episode_id - 1
            obs = env.reset(seed=reset_seed, init_episode_idx=init_episode_idx)
            prefetched = None
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}

            while env_steps < max_env_steps:
                timer.tick("total")
                with timer.context("sample_actions"):
                    if prefetched is None:
                        base_policy_input = build_libero_policy_input(obs, task_prompt)

                        base_actions, _ = policy_client.infer(base_policy_input)
                        base_actions = prepare_base_actions_chunk(
                            base_actions=base_actions,
                            chunk_horizon=chunk_horizon,
                        )

                        residual_obs = build_chunk_residual_obs(
                            obs=obs,
                            base_actions=base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )
                    else:
                        base_actions = prefetched["base_actions"]
                        residual_obs = prefetched["residual_obs"]
                        prefetched = None

                    residual_actions = agent.sample_action(residual_obs, deterministic=False)

                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                episode_done = False

                for action in final_actions:

                    if env_steps >= max_env_steps:
                        break

                    with timer.context("step_env"):
                        next_obs, reward, done, truncated, info = env.step(action)

                    with timer.context("build_decision_obs"):
                        next_base_policy_input = build_libero_policy_input(
                            next_obs, task_prompt
                        )
                        next_base_actions, _ = policy_client.infer(next_base_policy_input)
                        next_base_actions = prepare_base_actions_chunk(
                            base_actions=next_base_actions,
                            chunk_horizon=chunk_horizon,
                        )

                        next_residual_obs = build_chunk_residual_obs(
                            obs=next_obs,
                            base_actions=next_base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )

                    env_done = bool(info.get("env_done", False))
                    episode_done = bool(done or truncated)
                    transition = {
                        "episode_id": int(episode_id),
                        "episode_step": int(episode_steps),
                        "observations": residual_obs,
                        "actions": np.asarray(action, dtype=np.float32).reshape(-1),
                        "next_observations": next_residual_obs,
                        "rewards": float(reward),
                        "masks": float(0.0 if env_done else 1.0),
                        "dones": episode_done,
                    }
                    data_store.insert(transition)

                    last_info = dict(info)
                    env_steps += 1
                    episode_steps += 1
                    episode_return += float(reward)
                    episode_success = bool(episode_success or env_done)
                    obs = dict(next_obs)

                    prefetched = {
                        "base_actions": next_base_actions,
                        "residual_obs": next_residual_obs,
                    }

                    residual_obs = next_residual_obs

                    if env_steps % steps_per_update == 0:
                        client.update()

                    if env_steps % log_period == 0:
                        client.request(
                            "send-stats",
                            {"actor_timer": to_jsonable(timer.get_average_times())},
                        )
                    if episode_done or env_steps >= max_env_steps:
                        break

                timer.tock("total")

                if episode_done:
                    break

            client.update()
            success_count += int(episode_success)
            episode_stats = {
                "train": to_jsonable(last_info),
                "env_steps": int(env_steps),
                "actor_episode": {
                    "episode_id": int(episode_id),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "seed": int(reset_seed),
                    "init_episode_idx": int(init_episode_idx),
                    "success": bool(episode_success),
                    "running_success_rate": float(success_count / max(1, episode_id)),
                },
            }

            client.request("send-stats", episode_stats)
            logger.info(
                "episode=%s success=%s steps=%s return=%.3f env_steps=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(env_steps),
            )

    finally:
        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        try:
            client.update()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:  # noqa: BLE001
                pass


def learner(cfg: LiberoTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(
        cfg,
        action_dim=action_dim,
    )
    sample_obs = build_chunk_residual_sample_obs(
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
    )
    agent = create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=image_keys,
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
    replay_buffer = MemoryEfficientStepWindowReplayBufferDataStore(
        observation_space=observation_space,
        action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(action_dim,),
            dtype=np.float32,
        ),
        capacity=cfg.replay.capacity,
        window_size=chunk_horizon,
        discount=cfg.sac.discount,
        sample_stride=1,
        require_full_window=False,
        image_keys=image_keys,
    )
    wandb_cfg = WandBLogger.get_default_config()
    run_name = cfg.wandb.exp_name
    wandb_cfg.update(
        {
            "project": cfg.wandb.project,
            "exp_descriptor": run_name,
            "tag": [run_name],
            "group": cfg.wandb.group,
        }
    )
    wandb_variant = cfg_to_log_payload(cfg)
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandBLogger(
        wandb_config=wandb_cfg,
        variant=wandb_variant,
        wandb_output_dir=str(wandb_dir),
        debug=cfg.wandb.debug,
    )

    update_steps = 0
    env_steps = 0
    summary: dict[str, Any] = {
        "role": "learner",
        "mode": "residual",
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
    }

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal env_steps
        if request_type != "send-stats":
            raise ValueError(f"Invalid request type: {request_type}")
        if "env_steps" in payload:
            env_steps = max(int(env_steps), int(payload["env_steps"]))
        wandb_logger.log(to_jsonable(payload), step=update_steps)
        return {}

    server = TrainerServer(
        TrainerConfig(  # pyright: ignore[reportCallIssue]
            port_number=cfg.runtime.trainer_port,
            broadcast_port=cfg.runtime.broadcast_port,
            request_types=["send-stats"],
        ),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    training_starts = cfg.training.training_starts
    while len(replay_buffer) < training_starts:
        logger.info(
            "filling replay buffer: %s / %s",
            int(len(replay_buffer)),
            int(training_starts),
        )
        time.sleep(FILL_WAIT_SLEEP_SEC)

    server.publish_network(
        snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
    )
    logger.info("Published initial learner network")

    checkpoint_every = cfg.training.checkpoint.every_steps
    checkpoint_keep = cfg.training.checkpoint.keep
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_period = cfg.training.log_period
    max_update_steps = cfg.training.max_update_steps
    critic_actor_ratio = max(1, cfg.training.critic_actor_ratio)
    steps_per_update = cfg.training.steps_per_update
    timer = Timer()

    try:
        while update_steps < max_update_steps:
            if not update_steps < env_steps:
                time.sleep(LEARNER_IDLE_SLEEP_SEC)
                continue

            for _ in range(max(0, critic_actor_ratio - 1)):
                with timer.context("sample_replay_buffer"):
                    batch = replay_buffer.sample(cfg.replay.batch_size)
                    batch["actions"] = batch["actions"].reshape(
                        int(batch["actions"].shape[0]),
                        -1,
                    )
                    batch["action_mask"] = batch["action_mask"].reshape(
                        int(batch["action_mask"].shape[0]),
                        -1,
                    )
                with timer.context("train_critics"):
                    agent, _critics_info = agent.update_critics(batch)

            with timer.context("train"):
                batch = replay_buffer.sample(cfg.replay.batch_size)
                batch["actions"] = batch["actions"].reshape(
                    int(batch["actions"].shape[0]),
                    -1,
                )
                batch["action_mask"] = batch["action_mask"].reshape(
                    int(batch["action_mask"].shape[0]),
                    -1,
                )
                agent, update_info = agent.update_high_utd(
                    batch,
                    utd_ratio=cfg.sac.utd_ratio,
                )

            if update_steps > 0 and update_steps % steps_per_update == 0:
                server.publish_network(
                    snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
                )

            if update_steps % log_period == 0:
                update_metrics = to_jsonable(update_info)
                wandb_logger.log(update_metrics, step=update_steps)
                wandb_logger.log(
                    {"timer": to_jsonable(timer.get_average_times())},
                    step=update_steps,
                )
                logger.info(
                    "update_steps=%s env_steps=%s replay=%s info=%s",
                    int(update_steps),
                    int(env_steps),
                    int(len(replay_buffer)),
                    update_metrics,
                )

            if checkpoint_every > 0 and update_steps % checkpoint_every == 0:
                save_agent_checkpoint(
                    checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(checkpoint_keep),
                )

            update_steps += 1

    finally:
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        try:
            if getattr(wandb_logger, "run", None) is not None:
                wandb_logger.run.finish()
        except Exception:  # noqa: BLE001
            pass
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_residual",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_residual")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    set_global_seeds(typed_cfg.global_seed)

    if typed_cfg.runtime.role == "actor":
        actor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    learner(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
