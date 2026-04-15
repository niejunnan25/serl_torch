from __future__ import annotations

"""Reference-style AgiBot direct DRQ training script."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.agents.continuous.drq_config import create_drq_agent_from_cfg
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.residual.train.obs_utils import _obs_space_from_sample
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.timer_utils import Timer

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.config import resolve_agibot_cfg_image_keys
from serl_torch.examples.agibot_real.env.task_env import AgiBotTaskEnv
from serl_torch.examples.agibot_real.runtime.obs_adapter import (
    RESIDUAL_IMAGE_HEIGHT,
)
from serl_torch.examples.agibot_real.runtime.obs_adapter import RESIDUAL_IMAGE_WIDTH
from serl_torch.examples.agibot_real.runtime.obs_adapter import build_agibot_state
from serl_torch.examples.agibot_real.runtime.obs_adapter import extract_residual_images

AGIBOT_STATE_DIM = 14
FILL_WAIT_SLEEP_SEC = 1.0
LEARNER_IDLE_SLEEP_SEC = 1.0


def _normalize_role(value: Any) -> str:
    role = str(value).strip().lower()
    if role not in {"actor", "learner"}:
        raise ValueError(
            f"reference_style.role must be 'actor' or 'learner', got {value!r}"
        )
    return role


def _make_trainer_config(port_number: int, broadcast_port: int):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port_number),
        broadcast_port=int(broadcast_port),
        request_types=["send-stats"],
    )


def _create_wandb_logger(cfg: DictConfig, *, run_dir: Path) -> WandBLogger:
    wandb_cfg = WandBLogger.get_default_config()
    description = str(cfg.wandb.get("exp_name", None) or cfg.task.name)
    wandb_cfg.update(
        {
            "project": str(cfg.wandb.get("project", "serl_dev")),
            "exp_descriptor": description,
            "tag": description,
            "group": cfg.wandb.get("group", None),
        }
    )
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return WandBLogger(
        wandb_config=wandb_cfg,
        variant=variant if isinstance(variant, dict) else {},
        wandb_output_dir=str(wandb_dir),
        debug=bool(cfg.wandb.get("debug", False)),
    )


def _validate_cfg(cfg: DictConfig) -> str:
    role = _normalize_role(cfg.reference_style.role)

    if int(cfg.env.action_dim) <= 0:
        raise ValueError(f"env.action_dim must be positive, got {cfg.env.action_dim}")
    if int(cfg.training.training_starts) < 0:
        raise ValueError(
            f"training.training_starts must be >= 0, got {cfg.training.training_starts}"
        )
    if int(cfg.training.steps_per_update) <= 0:
        raise ValueError(
            f"training.steps_per_update must be positive, got {cfg.training.steps_per_update}"
        )
    if int(cfg.training.critic_actor_ratio) <= 0:
        raise ValueError(
            "training.critic_actor_ratio must be positive, "
            f"got {cfg.training.critic_actor_ratio}"
        )
    if int(cfg.training.log_period) <= 0:
        raise ValueError(
            f"training.log_period must be positive, got {cfg.training.log_period}"
        )
    if int(cfg.training.max_env_steps) <= 0:
        raise ValueError(
            f"training.max_env_steps must be positive, got {cfg.training.max_env_steps}"
        )
    if int(cfg.training.max_update_steps) <= 0:
        raise ValueError(
            "training.max_update_steps must be positive, "
            f"got {cfg.training.max_update_steps}"
        )
    if int(cfg.training.max_episodes) <= 0:
        raise ValueError(
            f"training.max_episodes must be positive, got {cfg.training.max_episodes}"
        )
    if int(cfg.sac.obs_stack_horizon) != 1:
        raise ValueError(
            "reference-style AgiBot direct DRQ currently supports only "
            "sac.obs_stack_horizon=1"
        )
    return role


def _create_env(cfg: DictConfig, logger: logging.Logger) -> AgiBotTaskEnv:
    task_cfg = cfg.get("task", {})
    robot_cfg = cfg.get("robot", {})
    controller_cfg = OmegaConf.to_container(cfg.get("controller", {}), resolve=True)
    return AgiBotTaskEnv(
        task_name=str(task_cfg.get("name", "agibot_real_task")),
        prompt=str(task_cfg.get("prompt", task_cfg.get("name", "agibot_real_task"))),
        action_dim=int(cfg.get("env", {}).get("action_dim", 14)),
        control_mode=str(task_cfg.get("control_mode", "camera_position")),
        hz=float(task_cfg.get("hz", 20.0)),
        use_smooth_trajectory=bool(task_cfg.get("use_smooth_trajectory", False)),
        trajectory_time=task_cfg.get("trajectory_time", None),
        max_episode_steps=task_cfg.get("max_episode_steps", None),
        assets_root=robot_cfg.get("assets_root", None),
        retargeter_urdf_path=robot_cfg.get("retargeter_urdf_path", None),
        retargeter_camera_extrinsic_path=robot_cfg.get(
            "retargeter_camera_extrinsic_path", None
        ),
        controller=controller_cfg,
        reset_hook=task_cfg.get("reset_hook", None),
        success_hook=task_cfg.get("success_hook", None),
        expert_precheck_hook=task_cfg.get("expert_precheck_hook", None),
        logger=logger,
    )


def _build_step_obs(
    obs_raw: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    state = build_agibot_state(obs_raw)
    images_all = extract_residual_images(obs_raw)
    obs_out: dict[str, np.ndarray] = {
        "state": np.expand_dims(np.asarray(state, dtype=np.float32), axis=0),
    }
    for key in image_keys:
        obs_out[key] = np.expand_dims(np.asarray(images_all[key]).copy(), axis=0)
    return obs_out


def _build_sample_obs(image_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    obs_out: dict[str, np.ndarray] = {
        "state": np.zeros((1, AGIBOT_STATE_DIM), dtype=np.float32),
    }
    for key in image_keys:
        obs_out[key] = np.zeros(
            (1, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )
    return obs_out


def _create_replay_store(
    cfg: DictConfig,
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
):
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

    action_dim = int(cfg.env.action_dim)
    action_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(action_dim,),
        dtype=np.float32,
    )
    return MemoryEfficientReplayBufferDataStore(
        observation_space=_obs_space_from_sample(sample_obs),
        action_space=action_space,
        capacity=int(cfg.replay.capacity),
        image_keys=tuple(image_keys),
    )


def _create_agent(
    cfg: DictConfig,
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
):
    action_dim = int(cfg.env.action_dim)
    return create_drq_agent_from_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=action_dim,
        image_keys=tuple(image_keys),
        critic_action_dim=action_dim,
        action_transform=None,
    )


def actor(cfg: DictConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient

    env = _create_env(cfg, logger)
    image_keys = tuple(resolve_agibot_cfg_image_keys(cfg))
    sample_obs = _build_sample_obs(image_keys)
    agent = _create_agent(cfg, sample_obs=sample_obs, image_keys=image_keys)

    data_store = QueuedDataStore(int(cfg.reference_style.data_store_queue_size))
    client = TrainerClient(
        "actor_env",
        str(cfg.reference_style.trainer_host),
        _make_trainer_config(
            port_number=int(cfg.reference_style.trainer_port),
            broadcast_port=int(cfg.reference_style.broadcast_port),
        ),
        data_store,
        wait_for_server=True,
    )

    def _update_actor_agent(payload: dict[str, Any]) -> None:
        apply_checkpoint_payload_to_agent(
            agent,
            dict(payload),
            load_optimizers=False,
        )

    client.recv_network_callback(_update_actor_agent)

    timer = Timer()
    steps_per_update = int(cfg.training.steps_per_update)
    log_period = int(cfg.training.log_period)
    max_env_steps = int(cfg.training.max_env_steps)
    max_episodes = int(cfg.training.max_episodes)

    env_steps = 0
    episode_id = 0
    success_count = 0
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "direct",
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
    }

    try:
        while env_steps < max_env_steps and episode_id < max_episodes:
            episode_id += 1
            obs_raw = env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}

            while env_steps < max_env_steps:
                timer.tick("total")
                with timer.context("sample_actions"):
                    obs = _build_step_obs(obs_raw, image_keys=image_keys)
                    action = np.asarray(
                        agent.sample_actions(obs, deterministic=False),
                        dtype=np.float32,
                    ).reshape(-1)

                with timer.context("step_env"):
                    next_obs_raw, reward, done, truncated, info = env.step(action)
                    next_obs = _build_step_obs(next_obs_raw, image_keys=image_keys)
                    done_flag = bool(done or truncated)
                    transition = {
                        "observations": obs,
                        "actions": np.asarray(action, dtype=np.float32).reshape(-1),
                        "next_observations": next_obs,
                        "rewards": float(reward),
                        "masks": float(0.0 if done_flag else 1.0),
                        "dones": bool(done_flag),
                    }
                    data_store.insert(transition)

                timer.tock("total")

                obs_raw = next_obs_raw
                last_info = dict(info)
                env_steps += 1
                episode_steps += 1
                episode_return += float(reward)
                episode_success = bool(info.get("success", episode_success))

                if env_steps % steps_per_update == 0:
                    client.update()
                if env_steps % log_period == 0:
                    client.request(
                        "send-stats",
                        {"actor_timer": timer.get_average_times()},
                    )
                if done_flag:
                    break

            client.update()
            success_count += int(episode_success)
            stats = {
                "train": dict(last_info),
                "env_steps": int(env_steps),
                "actor_episode": {
                    "episode_id": int(episode_id),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "success": bool(episode_success),
                    "running_success_rate": float(
                        success_count / max(1, episode_id)
                    ),
                },
            }
            client.request("send-stats", stats)
            logger.info(
                "episode=%s success=%s steps=%s return=%.3f env_steps=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(env_steps),
            )

        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
            }
        )
    finally:
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        try:
            client.update()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass


def learner(cfg: DictConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    from agentlace.trainer import TrainerServer

    image_keys = tuple(resolve_agibot_cfg_image_keys(cfg))
    sample_obs = _build_sample_obs(image_keys)
    agent = _create_agent(cfg, sample_obs=sample_obs, image_keys=image_keys)
    replay_buffer = _create_replay_store(
        cfg,
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
    wandb_logger = _create_wandb_logger(cfg, run_dir=run_dir)

    update_steps = 0
    env_steps = 0
    summary: dict[str, Any] = {
        "role": "learner",
        "mode": "direct",
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
    }

    def _stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal env_steps
        if request_type != "send-stats":
            raise ValueError(f"Invalid request type: {request_type}")
        if "env_steps" in payload:
            env_steps = max(int(env_steps), int(payload["env_steps"]))
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(
        _make_trainer_config(
            port_number=int(cfg.reference_style.trainer_port),
            broadcast_port=int(cfg.reference_style.broadcast_port),
        ),
        request_callback=_stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    training_starts = int(cfg.training.training_starts)
    while len(replay_buffer) < training_starts:
        logger.info(
            "filling replay buffer: %s / %s",
            int(len(replay_buffer)),
            int(training_starts),
        )
        time.sleep(FILL_WAIT_SLEEP_SEC)

    server.publish_network(snapshot_agent_checkpoint_payload(agent, step=int(update_steps)))
    logger.info("Published initial learner network")

    checkpoint_every = int(cfg.training.checkpoint.every_steps)
    checkpoint_keep = int(cfg.training.checkpoint.keep)
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_period = int(cfg.training.log_period)
    max_update_steps = int(cfg.training.max_update_steps)
    critic_actor_ratio = max(1, int(cfg.training.critic_actor_ratio))
    steps_per_update = int(cfg.training.steps_per_update)
    timer = Timer()

    try:
        while update_steps < max_update_steps:
            if not update_steps < env_steps:
                time.sleep(LEARNER_IDLE_SLEEP_SEC)
                continue

            for _ in range(max(0, critic_actor_ratio - 1)):
                with timer.context("sample_replay_buffer"):
                    batch = replay_buffer.sample(
                        int(cfg.replay.batch_size),
                        pack_obs_and_next_obs=True,
                    )
                with timer.context("train_critics"):
                    agent, _critics_info = agent.update_critics(batch)

            with timer.context("train"):
                batch = replay_buffer.sample(
                    int(cfg.replay.batch_size),
                    pack_obs_and_next_obs=True,
                )
                agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

            if update_steps > 0 and update_steps % steps_per_update == 0:
                server.publish_network(
                    snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
                )

            if update_steps % log_period == 0:
                wandb_logger.log(update_info, step=update_steps)
                wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
                logger.info(
                    "update_steps=%s env_steps=%s replay=%s info=%s",
                    int(update_steps),
                    int(env_steps),
                    int(len(replay_buffer)),
                    update_info,
                )

            if checkpoint_every > 0 and update_steps % checkpoint_every == 0:
                save_agent_checkpoint(
                    checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(checkpoint_keep),
                )

            update_steps += 1

        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
            }
        )
    finally:
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        try:
            if wandb_logger is not None and getattr(wandb_logger, "run", None) is not None:
                wandb_logger.run.finish()
        except Exception:  # noqa: BLE001
            pass
        server.stop()


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="train_reference_style",
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_reference_style_direct")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    role = _validate_cfg(cfg)
    if role == "actor":
        actor(cfg, run_dir=run_dir, logger=logger)
        return
    learner(cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
