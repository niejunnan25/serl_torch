"""RLT Stage 2 async training script for LIBERO.

Actor/learner split via agentlace. This keeps the RLT Stage 2 semantics
while adapting rollout, replay, checkpointing, and async eval to serl_torch.

Usage:
    # Learner
    python scripts/run_rlt_training.py runtime.role=learner \
        rlt.pi0_checkpoint_path=/path/to/pi0 \
        rlt.rlt_encoder_path=/path/to/encoder

    # Actor
    python scripts/run_rlt_training.py runtime.role=actor \
        rlt.pi0_checkpoint_path=/path/to/pi0 \
        rlt.rlt_encoder_path=/path/to/encoder
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for _path in (SERL_LAUNCHER_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_PARENT = REPO_ROOT.parent

import gym
import wandb
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from serl_launcher.common.checkpoint_codec import (
    apply_checkpoint_payload_to_agent,
    snapshot_agent_checkpoint_payload,
)
from serl_launcher.async_eval import (
    append_async_eval_checkpoint_index,
    load_async_eval_queue,
    load_completed_async_eval_indices,
    prune_async_eval_checkpoints,
    save_async_eval_checkpoint_payload,
)
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

from examples.libero_rlt.config import LiberoRLTTrainConfig, cfg_to_log_payload, parse_train_cfg
from serl_launcher.agents.rlt.agent import create_rlt_agent_from_cfg
from serl_launcher.agents.rlt.observation import build_rlt_obs, build_rlt_observation_space
from serl_launcher.policy.vla_features.client import VLAFeatureClient
from examples.libero_rlt.async_eval import (
    append_async_eval_request,
    append_async_eval_stop,
    check_async_eval_worker,
    load_new_async_eval_results,
    start_async_eval_worker,
    wait_for_async_eval_worker,
)
from examples.libero.env.factory import create_env
from openpi_client import image_tools

FILL_WAIT_SLEEP_SEC = 1.0
LEARNER_IDLE_SLEEP_SEC = 1.0
ACTOR_TIMER_LOG_FILE = "actor_timers.jsonl"
LEARNER_TIMER_LOG_FILE = "learner_timers.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# Actor
# ═══════════════════════════════════════════════════════════════════════════

def prepare_vla_obs(obs: Dict[str, Any], task_description: str, resize_size: int = 224) -> Dict[str, Any]:
    """Convert a raw LIBERO observation into the OpenPI policy input format."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )

    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": task_description,
    }

def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def actor(cfg: LiberoRLTTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    rlt = cfg.rlt
    chunk_size = rlt.chunk_size
    action_dim = rlt.action_dim
    warmup_steps = rlt.warmup_steps
    execute_horizon = rlt.execute_horizon

    # Connect to VLA feature server (Pi0 + encoder run there)
    vla_client = VLAFeatureClient(
        host=rlt.vla_server_host,
        port=rlt.vla_server_port,
        logger=logger,
    )
    logger.info("Connected to VLA feature server at %s:%s", rlt.vla_server_host, rlt.vla_server_port)

    # Create env
    env = create_env(cfg, logger)
    task_description = env.current_instruction

    # Create agent (actor + critics for action selection)
    agent = create_rlt_agent_from_cfg(cfg)

    # Agentlace client
    data_store = QueuedDataStore(cfg.runtime.data_store_queue_size)
    client = TrainerClient(
        "actor_env",
        cfg.runtime.trainer_host,
        TrainerConfig(
            port_number=cfg.runtime.trainer_port,
            broadcast_port=cfg.runtime.broadcast_port,
            request_types=["send-stats"],
        ),
        data_store,
        wait_for_server=True,
    )

    def update_actor(payload: dict[str, Any]) -> None:
        apply_checkpoint_payload_to_agent(agent, dict(payload), load_optimizers=False)

    client.recv_network_callback(update_actor)

    timer = Timer()
    steps_per_update = cfg.training.steps_per_update
    log_period = cfg.training.log_period
    max_env_steps = cfg.training.max_env_steps

    env_steps = 0
    episode_id = 0
    success_count = 0
    recent_successes: deque[int] = deque(maxlen=20)
    actor_timer_log_path = run_dir / ACTOR_TIMER_LOG_FILE
    rollout_log_path = run_dir / (cfg.logging.episode_log_file or "episode_logs.jsonl")
    video_dir = run_dir / "videos"
    if cfg.logging.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(total=max_env_steps, desc="actor env_steps", dynamic_ncols=True, leave=True)

    try:
        while env_steps < max_env_steps:
            episode_id += 1
            obs = env.reset(seed=cfg.env.seed, init_episode_idx=episode_id - 1)
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            replay_images = []

            while env_steps < max_env_steps:
                timer.tick("total")

                with timer.context("vla_inference"):
                    processed_obs = prepare_vla_obs(obs, task_description)
                    if cfg.logging.save_videos:
                        replay_images.append(processed_obs["observation/image"])

                    features = vla_client.infer(processed_obs)
                    rlt_obs = build_rlt_obs(
                        z_rl=features["z_rl"],
                        proprio=features["proprio"],
                        reference_action=features["reference_action"],
                    )
                with timer.context("sample_action"):
                    if env_steps < warmup_steps:
                        action_chunk_flat = features["reference_action"]
                    else:
                        action_chunk_flat = agent.sample_action(rlt_obs, deterministic=False)

                action_chunk = action_chunk_flat.reshape(chunk_size, action_dim)

                chunk_reward = 0.0
                chunk_done = False
                env_done = False
                executed_steps = 0
                done = False
                truncated = False

                with timer.context("step_env"):
                    for step_idx in range(execute_horizon):
                        if env_steps >= max_env_steps:
                            break

                        next_obs, reward, done, truncated, info = env.step(action_chunk[step_idx])
                        chunk_reward += float(reward)
                        env_steps += 1
                        executed_steps += 1
                        progress_bar.update(1)
                        episode_steps += 1
                        episode_return += float(reward)

                        env_done_step = bool(info.get("env_done", False))
                        env_done = env_done or env_done_step
                        episode_success = episode_success or env_done_step

                        if done or truncated or env_done_step:
                            chunk_done = True
                            break

                transition_done = bool(chunk_done or env_done)
                if transition_done:
                    next_rlt_obs = rlt_obs
                else:
                    with timer.context("next_vla_inference"):
                        processed_next_obs = prepare_vla_obs(next_obs, task_description)
                        next_features = vla_client.infer(processed_next_obs)
                        next_rlt_obs = build_rlt_obs(
                            z_rl=next_features["z_rl"],
                            proprio=next_features["proprio"],
                            reference_action=next_features["reference_action"],
                        )

                transition = {
                    "observations": rlt_obs,
                    "actions": np.asarray(action_chunk_flat, dtype=np.float32),
                    "next_observations": next_rlt_obs,
                    "rewards": float(chunk_reward),
                    "masks": 0.0 if transition_done else 1.0,
                    "dones": transition_done,
                    "discounts": float(rlt.discount ** max(1, executed_steps)),
                    "executed_steps": float(executed_steps),
                }
                data_store.insert(transition)

                if env_steps % steps_per_update == 0:
                    client.update()

                timer.tock("total")

                if env_steps % log_period == 0:
                    append_jsonl(actor_timer_log_path, {
                        "source": "actor",
                        "env_steps": env_steps,
                        "episode_id": episode_id,
                        "timer": timer.get_average_times(),
                    })

                if chunk_done or env_steps >= max_env_steps:
                    obs = next_obs
                    break

                obs = next_obs

            logger.info("rollout episode=%d success=%s", episode_id, episode_success)

            if cfg.logging.save_videos and replay_images:
                try:
                    import imageio

                    suffix = "success" if episode_success else "failure"
                    task_segment = task_description.replace(" ", "_").replace("/", "_")
                    video_filename = video_dir / f"rollout_{task_segment}_{episode_id}_{suffix}.mp4"
                    imageio.mimwrite(
                        video_filename,
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                except Exception as e:
                    logger.error("Failed to save video for episode %s: %s", episode_id, e)

            client.update()
            success_count += int(episode_success)
            recent_successes.append(int(episode_success))
            recent_rate = float(sum(recent_successes)) / max(1, len(recent_successes))

            stats_payload = {
                "env_steps": env_steps,
                "rollout": {
                    "episode_id": episode_id,
                    "episode_return": episode_return,
                    "episode_steps": episode_steps,
                    "episode_success": int(episode_success),
                    "recent_success_rate_20": recent_rate,
                },
            }
            try:
                client.request("send-stats", stats_payload)
            except Exception:
                pass

            append_jsonl(rollout_log_path, {
                "episode_id": episode_id,
                "env_steps": env_steps,
                "episode_return": episode_return,
                "episode_steps": episode_steps,
                "success": int(episode_success),
            })

        try:
            client.request("send-stats", {
                "env_steps": env_steps,
                "actor_finished": True,
            })
        except Exception:
            pass

    finally:
        progress_bar.close()
        try:
            client.update()
        except Exception:
            pass
        try:
            client.stop()
        except Exception:
            pass
        try:
            env.close(clear_cache=False)
        except Exception:
            pass
        try:
            vla_client.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Learner
# ═══════════════════════════════════════════════════════════════════════════

def learner(cfg: LiberoRLTTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    rlt = cfg.rlt

    # Create agent
    agent = create_rlt_agent_from_cfg(cfg)

    # Create replay buffer (no images, pure vector observations)
    obs_space = build_rlt_observation_space(
        z_rl_dim=rlt.z_rl_dim,
        proprio_dim=rlt.proprio_dim,
        chunk_size=rlt.chunk_size,
        action_dim=rlt.action_dim,
    )
    action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(rlt.action_dim*rlt.chunk_size),),
            dtype=np.float32,
        )
    replay_buffer = ReplayBufferDataStore(
        capacity=cfg.replay.capacity,
        observation_space=obs_space,
        action_space=action_space,
        extra_fields={
            "discounts": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            "executed_steps": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
        },
    )

    # Wandb
    wandb_cfg = WandBLogger.get_default_config()
    wandb_cfg.update({
        "project": cfg.wandb.project,
        "exp_descriptor": cfg.wandb.exp_name,
        "tag": [cfg.wandb.exp_name],
        "group": cfg.wandb.group,
    })
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandBLogger(
        wandb_config=wandb_cfg,
        variant=cfg_to_log_payload(cfg),
        wandb_output_dir=str(wandb_dir),
        debug=cfg.wandb.debug,
    )

    try:
        if wandb.run is not None:
            wandb.define_metric("rollout/episode_id")
            wandb.define_metric("rollout/*", step_metric="rollout/episode_id")
    except Exception as e:
        logger.warning(f"Could not define custom wandb metrics for episode_id: {e}")

    # Async eval
    async_eval = start_async_eval_worker(cfg, run_dir=run_dir, logger=logger)

    # State
    update_steps = 0
    env_steps = 0
    latest_completed_episode_id = 0
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0
    actor_finished = False
    progress_state_lock = Lock()

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal env_steps, latest_completed_episode_id, actor_finished
        if request_type != "send-stats":
            return {}
        with progress_state_lock:
            env_steps = max(env_steps, int(payload.get("env_steps", 0)))
            actor_finished = actor_finished or bool(payload.get("actor_finished", False))
            rollout = payload.get("rollout", {})
            ep_id = int(rollout.get("episode_id", 0))
            if ep_id > 0:
                latest_completed_episode_id = max(latest_completed_episode_id, ep_id)
                completed_episode_env_steps[ep_id] = env_steps
        rollout = payload.get("rollout", {})
        if rollout:
            ep_id = int(rollout.get("episode_id", 0))
            wandb_logger.log(to_jsonable({
                "rollout/episode_id": ep_id,
                "rollout/episode_return": rollout.get("episode_return", 0),
                "rollout/episode_steps": rollout.get("episode_steps", 0),
                "rollout/success": rollout.get("episode_success", 0),
                "rollout/recent_success_rate_20": rollout.get("recent_success_rate_20", 0),
            }))
        return {}

    # Trainer server
    server = TrainerServer(
        TrainerConfig(
            port_number=cfg.runtime.trainer_port,
            broadcast_port=cfg.runtime.broadcast_port,
            request_types=["send-stats"],
        ),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Training params
    training_starts = cfg.training.training_starts
    checkpoint_every = cfg.training.checkpoint.every_steps
    checkpoint_keep = cfg.training.checkpoint.keep
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_period = cfg.training.log_period
    max_update_steps = cfg.training.max_update_steps
    steps_per_update = cfg.training.steps_per_update
    batch_size = cfg.replay.batch_size
    utd_ratio = rlt.utd_ratio

    timer = Timer()
    last_log_time = time.time()
    last_log_update_steps = 0

    def _pending_async_eval_checkpoint_paths() -> set[Path]:
        if (not async_eval.enabled) or async_eval.queue_path is None:
            return set()
        try:
            queued_requests, _ = load_async_eval_queue(async_eval.queue_path)
            completed_eval_indices = load_completed_async_eval_indices(
                async_eval.summary_jsonl_path
            )
        except Exception as exc:
            logger.warning(
                "Could not inspect async eval queue before pruning checkpoints: %s",
                exc,
            )
            return set()

        pending_paths: set[Path] = set()
        for request in queued_requests:
            eval_index_raw = request.get("eval_index", None)
            try:
                eval_index = None if eval_index_raw is None else int(eval_index_raw)
            except Exception:
                eval_index = None
            if eval_index is not None and eval_index in completed_eval_indices:
                continue
            checkpoint_path_raw = request.get("checkpoint_path", None)
            if checkpoint_path_raw:
                pending_paths.add(Path(str(checkpoint_path_raw)))
        return pending_paths

    def _maybe_queue_async_eval() -> None:
        nonlocal last_queued_async_eval_episode
        if not async_eval.enabled or async_eval.eval_checkpoint_dir is None:
            return
        every_episodes = int(async_eval.every_episodes)
        if every_episodes <= 0:
            return
        while True:
            with progress_state_lock:
                next_target = last_queued_async_eval_episode + every_episodes
                if latest_completed_episode_id < next_target:
                    return
                target_episode = next_target
                target_env_step = completed_episode_env_steps.get(target_episode, env_steps)

            ckpt_payload = snapshot_agent_checkpoint_payload(agent, step=update_steps)
            ckpt_path = save_async_eval_checkpoint_payload(
                async_eval.eval_checkpoint_dir, ckpt_payload, episode_id=target_episode,
            )
            append_async_eval_checkpoint_index(
                async_eval.eval_checkpoint_dir,
                episode_id=target_episode,
                checkpoint_step=update_steps,
                checkpoint_path=ckpt_path,
            )
            prune_async_eval_checkpoints(
                async_eval.eval_checkpoint_dir,
                keep=async_eval.eval_checkpoint_keep,
                protected_paths=_pending_async_eval_checkpoint_paths(),
            )
            append_async_eval_request(async_eval, {
                "eval_index": async_eval.triggered_count,
                "train_episode_id": target_episode,
                "train_update_step": update_steps,
                "train_env_step": target_env_step,
                "checkpoint_step": update_steps,
                "checkpoint_path": str(ckpt_path),
            })
            with progress_state_lock:
                last_queued_async_eval_episode = max(last_queued_async_eval_episode, target_episode)
            logger.info(
                "queued eval: episode=%s update_steps=%s env_steps=%s",
                target_episode, update_steps, target_env_step,
            )

    # ── Replay warmup ──
    logger.info("Waiting for replay buffer to reach %s transitions...", training_starts)
    warmup_bar = tqdm(total=training_starts, desc="learner replay warmup", dynamic_ncols=True, leave=True)
    try:
        while len(replay_buffer) < training_starts:
            current = min(len(replay_buffer), training_starts)
            warmup_bar.n = current
            warmup_bar.refresh()
            with progress_state_lock:
                should_stop = actor_finished and len(replay_buffer) < training_starts
                current_env_steps = env_steps
            if should_stop:
                raise RuntimeError(
                    "Actor finished before replay warmup completed: "
                    f"replay_size={len(replay_buffer)} training_starts={training_starts} "
                    f"env_steps={current_env_steps}"
                )
            time.sleep(FILL_WAIT_SLEEP_SEC)
    finally:
        warmup_bar.close()
    logger.info("Replay warmup complete: size=%s", len(replay_buffer))

    # Publish initial network
    server.publish_network(snapshot_agent_checkpoint_payload(agent, step=update_steps))
    logger.info("Published initial network")

    # ── Main training loop ──
    try:
        while update_steps < max_update_steps:
            _maybe_queue_async_eval()

            # Wait for actor to be ahead
            if update_steps >= env_steps:
                with progress_state_lock:
                    should_stop = actor_finished and update_steps >= env_steps
                    current_env_steps = env_steps
                if should_stop:
                    logger.info(
                        "Actor finished and learner caught up: update_steps=%s env_steps=%s",
                        update_steps,
                        current_env_steps,
                    )
                    break
                time.sleep(LEARNER_IDLE_SLEEP_SEC)
                continue

            with timer.context("sample_replay"):
                batch = replay_buffer.sample(batch_size)

            with timer.context("train"):
                agent, update_info = agent.update_high_utd(batch, utd_ratio=utd_ratio)

            update_steps += 1
            learner_metrics = {
                    f"learner/{key}": value for key, value in update_info.items()
                }
            wandb_logger.log(to_jsonable(learner_metrics), step=update_steps)
            # Broadcast params
            if update_steps % steps_per_update == 0:
                server.publish_network(snapshot_agent_checkpoint_payload(agent, step=update_steps))

            # Logging
            if update_steps % log_period == 0:
                check_async_eval_worker(async_eval, logger=logger)
                # Sync eval results
                eval_results = load_new_async_eval_results(async_eval)
                for result in eval_results:
                    wandb_logger.log(to_jsonable(result))

                now = time.time()
                elapsed = max(now - last_log_time, 1e-6)
                updates_since = max(1, update_steps - last_log_update_steps)
                updates_per_sec = updates_since / elapsed
                last_log_time = now
                last_log_update_steps = update_steps
                logger.info(
                    "step=%s env=%s replay=%s ups=%.1f %s",
                    update_steps, env_steps, len(replay_buffer), updates_per_sec,
                    " ".join(f"{k}={v:.4f}" for k, v in update_info.items()),
                )

                append_jsonl(run_dir / LEARNER_TIMER_LOG_FILE, {
                    "source": "learner",
                    "update_steps": update_steps,
                    "env_steps": env_steps,
                    "replay_size": len(replay_buffer),
                    "updates_per_sec": updates_per_sec,
                    "timer": timer.get_average_times(),
                })

            # Checkpoint
            if checkpoint_every > 0 and update_steps % checkpoint_every == 0:
                ckpt_path = save_agent_checkpoint(
                    checkpoint_dir, agent, step=update_steps, keep=checkpoint_keep,
                )
                logger.info("checkpoint: step=%s path=%s", update_steps, ckpt_path)

    finally:
        _maybe_queue_async_eval()
        if async_eval.enabled:
            append_async_eval_stop(async_eval)
            wait_for_async_eval_worker(async_eval, logger=logger)
            for result in load_new_async_eval_results(async_eval):
                wandb_logger.log(to_jsonable(result))

        summary = {
            "role": "learner",
            "update_steps": update_steps,
            "env_steps": env_steps,
            "replay_size": len(replay_buffer),
        }
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        try:
            if getattr(wandb_logger, "run", None) is not None:
                wandb_logger.run.finish()
        except Exception:
            pass
        try:
            server.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="../configs", config_name="train_rlt", version_base=None)
def main(cfg: DictConfig) -> None:
    parsed = parse_train_cfg(cfg)
    set_global_seeds(parsed.global_seed)

    run_dir = Path(hydra.utils.get_original_cwd()) if hydra.utils.HydraConfig.initialized() else Path(".")
    try:
        from hydra.core.hydra_config import HydraConfig
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        run_dir = Path("outputs/rlt_training")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Role: %s | Run dir: %s", parsed.runtime.role, run_dir)

    if parsed.runtime.role == "actor":
        actor(parsed, run_dir=run_dir, logger=logger)
    else:
        learner(parsed, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
