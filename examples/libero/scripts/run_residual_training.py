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
from tqdm.auto import tqdm

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
from serl_torch.examples.libero.async_eval import AsyncEvalRuntime
from serl_torch.examples.libero.async_eval import append_async_eval_request
from serl_torch.examples.libero.async_eval import append_async_eval_stop
from serl_torch.examples.libero.async_eval import check_async_eval_worker
from serl_torch.examples.libero.async_eval import load_new_async_eval_results
from serl_torch.examples.libero.async_eval import start_async_eval_worker
from serl_torch.examples.libero.async_eval import summarize_async_eval_results
from serl_torch.examples.libero.async_eval import wait_for_async_eval_worker
from serl_torch.examples.libero.env.factory import create_env
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


def _sync_async_eval_results_to_wandb(
    async_eval: AsyncEvalRuntime,
    *,
    wandb_logger: WandBLogger,
    logger: logging.Logger,
) -> None:
    if not async_eval.enabled:
        return
    records = load_new_async_eval_results(async_eval)
    for payload in records:
        status = str(payload.get("status", "")).strip().lower()
        train_update_step_raw = payload.get("train_update_step", None)
        try:
            if train_update_step_raw is None:
                continue
            train_update_step = int(train_update_step_raw)
        except Exception:
            continue

        metrics: dict[str, Any] = {
            "async_eval/status_ok": 1.0 if status == "ok" else 0.0,
            "async_eval/status_failed": 0.0 if status == "ok" else 1.0,
        }
        for payload_key, metric_key in (
            ("train_env_step", "async_eval/train_env_step"),
            ("checkpoint_step", "async_eval/checkpoint_step"),
            ("duration_sec", "async_eval/duration_sec"),
            ("eval_index", "async_eval/eval_index"),
        ):
            value = payload.get(payload_key, None)
            if isinstance(value, (int, float)):
                metrics[metric_key] = float(value)

        summary = payload.get("summary", None)
        if status == "ok" and isinstance(summary, dict):
            for payload_key, metric_key in (
                ("success_rate", "async_eval/success_rate"),
                ("mean_return", "async_eval/mean_return"),
                ("mean_episode_steps", "async_eval/mean_episode_steps"),
                ("successes", "async_eval/successes"),
                ("episodes_completed", "async_eval/episodes_completed"),
                ("env_steps", "async_eval/eval_env_steps"),
            ):
                value = summary.get(payload_key, None)
                if isinstance(value, (int, float)):
                    metrics[metric_key] = float(value)

        wandb_logger.log(to_jsonable(metrics), step=train_update_step)

        if status == "ok":
            logger.info(
                "async eval done: eval_index=%s update_steps=%s env_steps=%s success_rate=%s",
                payload.get("eval_index", None),
                train_update_step,
                payload.get("train_env_step", None),
                (
                    summary.get("success_rate", None)
                    if isinstance(summary, dict)
                    else None
                ),
            )
        else:
            logger.warning(
                "async eval failed: eval_index=%s update_steps=%s error=%s",
                payload.get("eval_index", None),
                train_update_step,
                payload.get("error", None),
            )


def _maybe_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_optional_metric(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_learner_heartbeat(
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
    updates_per_sec: float,
    update_info: dict[str, Any],
) -> str:
    return (
        "learner heartbeat: "
        f"update_steps={int(update_steps)} "
        f"env_steps={int(env_steps)} "
        f"replay_size={int(replay_size)} "
        f"updates_per_sec={updates_per_sec:.2f} "
        f"critic_loss={_format_optional_metric(_maybe_float(update_info, 'critic_loss'))} "
        f"critic_td_loss={_format_optional_metric(_maybe_float(update_info, 'critic_td_loss'))} "
        f"actor_loss={_format_optional_metric(_maybe_float(update_info, 'actor_loss'))} "
        f"temperature={_format_optional_metric(_maybe_float(update_info, 'temperature'))} "
        f"entropy={_format_optional_metric(_maybe_float(update_info, 'entropy'))} "
        f"predicted_qs={_format_optional_metric(_maybe_float(update_info, 'predicted_qs'))} "
        f"target_qs={_format_optional_metric(_maybe_float(update_info, 'target_qs'))} "
        f"actor_predicted_q={_format_optional_metric(_maybe_float(update_info, 'actor_predicted_q'))}"
    )


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
    progress_bar = tqdm(
        total=int(max_env_steps),
        desc="actor env_steps",
        dynamic_ncols=True,
        leave=True,
    )

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
                should_log_timer = False

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
                    progress_bar.update(1)
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
                        should_log_timer = True
                    if episode_done or env_steps >= max_env_steps:
                        break

                timer.tock("total")

                if should_log_timer:
                    client.request(
                        "send-stats",
                        {"actor_timer": to_jsonable(timer.get_average_times())},
                    )

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
            progress_bar.set_postfix(
                episode=int(episode_id),
                success=int(bool(episode_success)),
                refresh=False,
            )
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
            progress_bar.close()
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


def learner(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
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
    async_eval = start_async_eval_worker(
        cfg,
        run_dir=run_dir,
        logger=logger,
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
    warmup_bar = tqdm(
        total=int(training_starts),
        initial=min(int(len(replay_buffer)), int(training_starts)),
        desc="learner replay warmup",
        dynamic_ncols=True,
        leave=True,
    )
    warmup_replay_size = min(int(len(replay_buffer)), int(training_starts))
    try:
        while len(replay_buffer) < training_starts:
            current_replay_size = min(int(len(replay_buffer)), int(training_starts))
            if current_replay_size > warmup_replay_size:
                warmup_bar.update(current_replay_size - warmup_replay_size)
                warmup_replay_size = current_replay_size
            warmup_bar.set_postfix(
                replay=int(len(replay_buffer)),
                env_steps=int(env_steps),
                refresh=False,
            )
            time.sleep(FILL_WAIT_SLEEP_SEC)
        current_replay_size = min(int(len(replay_buffer)), int(training_starts))
        if current_replay_size > warmup_replay_size:
            warmup_bar.update(current_replay_size - warmup_replay_size)
    finally:
        warmup_bar.close()
    logger.info(
        "replay warmup complete: replay=%s training_starts=%s env_steps=%s",
        int(len(replay_buffer)),
        int(training_starts),
        int(env_steps),
    )

    server.publish_network(
        snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
    )
    logger.info(
        "publish network: step=%s env_steps=%s reason=initial",
        int(update_steps),
        int(env_steps),
    )

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
    last_log_time = time.time()
    last_log_update_steps = int(update_steps)

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

            update_steps += 1

            if update_steps % steps_per_update == 0:
                server.publish_network(
                    snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
                )
                logger.info(
                    "publish network: step=%s env_steps=%s reason=periodic",
                    int(update_steps),
                    int(env_steps),
                )

            if update_steps % log_period == 0:
                check_async_eval_worker(async_eval, logger=logger)
                _sync_async_eval_results_to_wandb(
                    async_eval,
                    wandb_logger=wandb_logger,
                    logger=logger,
                )
                update_metrics = to_jsonable(update_info)
                now = time.time()
                elapsed_sec = max(now - last_log_time, 1e-6)
                updates_since_last_log = max(1, int(update_steps - last_log_update_steps))
                updates_per_sec = float(updates_since_last_log) / float(elapsed_sec)
                last_log_time = now
                last_log_update_steps = int(update_steps)
                timer_metrics = to_jsonable(timer.get_average_times())
                wandb_logger.log(update_metrics, step=update_steps)
                wandb_logger.log(
                    {
                        "learner/updates_per_sec": float(updates_per_sec),
                        "learner/replay_size": int(len(replay_buffer)),
                    },
                    step=update_steps,
                )
                wandb_logger.log(
                    {"timer": timer_metrics},
                    step=update_steps,
                )
                logger.info(
                    _format_learner_heartbeat(
                        update_steps=int(update_steps),
                        env_steps=int(env_steps),
                        replay_size=int(len(replay_buffer)),
                        updates_per_sec=float(updates_per_sec),
                        update_info=dict(update_metrics),
                    )
                )

            if checkpoint_every > 0 and update_steps % checkpoint_every == 0:
                checkpoint_path = save_agent_checkpoint(
                    checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(checkpoint_keep),
                )
                logger.info(
                    "checkpoint saved: step=%s env_steps=%s path=%s",
                    int(update_steps),
                    int(env_steps),
                    checkpoint_path,
                )

            if (
                async_eval.enabled
                and update_steps % int(async_eval.every_steps) == 0
                and async_eval.eval_checkpoint_dir is not None
            ):
                async_eval_checkpoint_path = save_agent_checkpoint(
                    async_eval.eval_checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(async_eval.eval_checkpoint_keep),
                )
                append_async_eval_request(
                    async_eval,
                    {
                        "eval_index": int(async_eval.triggered_count),
                        "train_update_step": int(update_steps),
                        "train_env_step": int(env_steps),
                        "checkpoint_step": int(update_steps),
                        "checkpoint_path": str(async_eval_checkpoint_path),
                    },
                )
                logger.info(
                    "queued async eval: eval_index=%s update_steps=%s env_steps=%s checkpoint=%s",
                    int(max(0, async_eval.triggered_count - 1)),
                    int(update_steps),
                    int(env_steps),
                    async_eval_checkpoint_path,
                )

    finally:
        async_eval_return_code = None
        if async_eval.enabled:
            append_async_eval_stop(async_eval)
            async_eval_return_code = wait_for_async_eval_worker(
                async_eval,
                logger=logger,
            )
            _sync_async_eval_results_to_wandb(
                async_eval,
                wandb_logger=wandb_logger,
                logger=logger,
            )
            if async_eval_return_code not in (None, 0):
                logger.warning(
                    "async eval worker exited with returncode=%s; see %s",
                    async_eval_return_code,
                    async_eval.worker_log_path,
                )
        async_eval_counts = summarize_async_eval_results(async_eval.summary_jsonl_path)
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
                "async_eval": {
                    "enabled": bool(async_eval.enabled),
                    "every_steps": int(async_eval.every_steps),
                    "triggered": int(async_eval.triggered_count),
                    "results_total": int(async_eval_counts["total"]),
                    "results_ok": int(async_eval_counts["ok"]),
                    "results_failed": int(async_eval_counts["failed"]),
                    "queue_path": (
                        str(async_eval.queue_path)
                        if async_eval.queue_path is not None
                        else None
                    ),
                    "summary_jsonl_path": (
                        str(async_eval.summary_jsonl_path)
                        if async_eval.summary_jsonl_path is not None
                        else None
                    ),
                    "worker_log_path": (
                        str(async_eval.worker_log_path)
                        if async_eval.worker_log_path is not None
                        else None
                    ),
                    "worker_return_code": (
                        None
                        if async_eval_return_code is None
                        else int(async_eval_return_code)
                    ),
                    "eval_checkpoint_dir": (
                        str(async_eval.eval_checkpoint_dir)
                        if async_eval.eval_checkpoint_dir is not None
                        else None
                    ),
                },
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
