from __future__ import annotations

"""Chunk-boundary AgiBot residual DRQ training script.

This variant differs from ``run_residual_training.py`` in one important way:

- actor executes an entire action chunk via ``env.step_chunk(...)``
- replay stores one transition per executed chunk boundary
- no step-level backfill is performed for intermediate states within a chunk

Known limitation:

- the current learner loop is driven by replay insert count, but the default
  config still allows ``training.max_update_steps`` to exceed the theoretical
  maximum number of chunk transitions in a full actor run
- with the current default ``chunk_horizon=15``, ``max_env_steps=300000``, and
  ``max_episodes=2000`` / ``max_episode_steps=150``, the actor can produce at
  most about 20k chunk transitions
- if ``max_update_steps`` is left significantly larger than that, the learner
  can sit idle after the actor exits, waiting for replay growth that will never
  happen
"""

from collections import deque
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

from agentlace.data.data_store import DataStoreBase
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient
from agentlace.trainer import TrainerConfig
from agentlace.trainer import TrainerServer
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.common.training_observability import (
    configure_learner_wandb_metrics,
)
from serl_launcher.common.training_observability import (
    configure_rollout_wandb_metrics,
)
from serl_launcher.common.training_observability import (
    extract_learner_wandb_metrics,
)
from serl_launcher.common.training_observability import (
    extract_rollout_wandb_metrics,
)
from serl_launcher.common.training_payloads import build_rollout_payload
from serl_launcher.common.training_payloads import build_rollout_stats_payload
from serl_launcher.common.training_payloads import parse_rollout_stats_payload
from serl_launcher.common.training_reporting import format_learner_heartbeat
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.step_window_replay_buffer import (
    MemoryEfficientStepWindowReplayBuffer,
)
from serl_launcher.residual.chunk_window_replay import reshape_chunk_batch_for_training
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.config import AgiBotTrainConfig
from serl_torch.examples.agibot_real.config import cfg_to_log_payload
from serl_torch.examples.agibot_real.config import parse_train_cfg
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.env.factory import create_env
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_obs,
)
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_observation_space,
)
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_sample_obs,
)


class ChunkBoundaryReplayBufferDataStore(
    MemoryEfficientStepWindowReplayBuffer,
    DataStoreBase,
):
    """Replay for one macro transition per executed chunk boundary.

    We still reuse the step-window replay implementation so that sampling returns
    the same batch structure as the current residual learner expects. The
    difference is that each inserted "step" is already a whole executed chunk,
    and we store an explicit flat ``action_mask`` to handle partial terminal
    chunks safely.
    """

    def __init__(
        self,
        *,
        observation_space: gym.spaces.Dict,
        chunk_action_dim: int,
        discount: float,
        capacity: int,
        image_keys: tuple[str, ...],
    ) -> None:
        MemoryEfficientStepWindowReplayBuffer.__init__(
            self,
            observation_space=observation_space,
            action_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(chunk_action_dim),),
                dtype=np.float32,
            ),
            capacity=int(capacity),
            window_size=1,
            discount=float(discount),
            sample_stride=1,
            require_full_window=False,
            pixel_keys=tuple(image_keys),
        )
        DataStoreBase.__init__(self, int(capacity))
        self._lock = Lock()
        self._chunk_action_masks = np.zeros(
            (int(capacity), int(chunk_action_dim)),
            dtype=np.float32,
        )

    def insert(self, data: dict[str, Any]) -> None:
        if "action_mask" not in data:
            raise KeyError("chunk boundary replay insert requires 'action_mask'")
        action_mask = np.asarray(data["action_mask"], dtype=np.float32).reshape(-1)
        expected_dim = int(self._step_action_shape[0])
        if int(action_mask.shape[0]) != expected_dim:
            raise ValueError(
                "chunk boundary action_mask dim mismatch: "
                f"{action_mask.shape[0]} != {expected_dim}"
            )

        step_record = dict(data)
        step_record.pop("action_mask")

        with self._lock:
            insert_index = int(self._insert_index)
            MemoryEfficientStepWindowReplayBuffer.insert(self, step_record)
            self._chunk_action_masks[insert_index] = action_mask

    def _build_transition(self, start_step_id: int) -> dict[str, Any]:
        transition = MemoryEfficientStepWindowReplayBuffer._build_transition(
            self,
            start_step_id,
        )
        start_idx = self._buffer_index(start_step_id)
        transition["action_mask"] = np.expand_dims(
            np.array(self._chunk_action_masks[start_idx], copy=True),
            axis=0,
        )
        return transition

    def sample(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            return MemoryEfficientStepWindowReplayBuffer.sample(
                self,
                *args,
                **kwargs,
            )

    def latest_data_id(self) -> int:
        return int(self._insert_count)

    def get_latest_data(self, from_id: int) -> None:
        raise NotImplementedError


def create_chunk_boundary_replay_buffer(
    *,
    observation_space: gym.spaces.Dict,
    chunk_action_dim: int,
    discount: float,
    image_keys: tuple[str, ...],
    capacity: int,
) -> ChunkBoundaryReplayBufferDataStore:
    return ChunkBoundaryReplayBufferDataStore(
        observation_space=observation_space,
        chunk_action_dim=int(chunk_action_dim),
        discount=float(discount),
        image_keys=image_keys,
        capacity=int(capacity),
    )


def _compute_discounted_chunk_reward(
    rewards: list[float],
    *,
    discount: float,
) -> float:
    discounted_return = 0.0
    for offset, reward in enumerate(rewards):
        discounted_return += (float(discount) ** int(offset)) * float(reward)
    return float(discounted_return)


def _pad_chunk_action_and_mask(
    *,
    executed_actions: np.ndarray,
    chunk_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(executed_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2-D executed_actions, got {actions.shape}")
    if int(actions.shape[1]) != int(action_dim):
        raise ValueError(
            f"Unexpected action_dim for executed_actions: {actions.shape[1]} != {action_dim}"
        )
    padded_actions = np.zeros(
        (int(chunk_horizon), int(action_dim)),
        dtype=np.float32,
    )
    padded_mask = np.zeros_like(padded_actions, dtype=np.float32)
    executed_steps = int(actions.shape[0])
    padded_actions[:executed_steps] = actions
    padded_mask[:executed_steps] = 1.0
    return padded_actions.reshape(-1), padded_mask.reshape(-1)


def _crossed_interval(
    *,
    previous_count: int,
    current_count: int,
    period: int,
) -> bool:
    if int(period) <= 0:
        return False
    return int(previous_count // period) < int(current_count // period)


def _build_terminal_next_residual_obs(
    *,
    obs: dict[str, Any],
    chunk_horizon: int,
    action_dim: int,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> dict[str, np.ndarray]:
    terminal_base_actions = np.zeros(
        (int(chunk_horizon), int(action_dim)),
        dtype=np.float32,
    )
    return build_chunk_residual_obs(
        obs=obs,
        base_actions=terminal_base_actions,
        image_keys=image_keys,
        residual_alpha=residual_alpha,
    )


def actor(cfg: AgiBotTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    if cfg.offline.enabled:
        raise ValueError(
            "Chunk-boundary training script does not support offline replay mixing yet. "
            "Disable offline.enabled for this variant."
        )

    env = create_env(cfg, logger)
    base_policy = build_agibot_base_policy(cfg, logger=logger)
    logger.info("Chunk policy backend: %s", base_policy.describe())

    image_keys = cfg.obs.image_keys
    action_dim = int(cfg.env.action_dim)
    chunk_horizon = int(cfg.residual.chunk_horizon)
    discount = float(cfg.sac.discount)
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
    steps_per_update = int(cfg.training.steps_per_update)
    log_period = int(cfg.training.log_period)
    max_env_steps = int(cfg.training.max_env_steps)
    max_episodes = int(cfg.training.max_episodes)

    env_steps = 0
    episode_id = 0
    success_count = 0
    recent_episode_successes: deque[int] = deque(maxlen=20)
    actor_timer_log_path = run_dir / "actor_timers.jsonl"
    rollout_log_path = run_dir / str(cfg.logging.episode_log_file or "episode_logs.jsonl")

    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "residual_chunk_boundary",
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
        "timer_log_path": str(actor_timer_log_path),
        "episode_log_path": str(rollout_log_path),
    }

    progress_bar = tqdm(
        total=int(max_env_steps),
        desc="actor env_steps",
        dynamic_ncols=True,
        leave=True,
    )

    logger.info(
        "chunk-boundary actor active: replay inserts one transition per chunk boundary "
        "(chunk_horizon=%s); intermediate step transitions are not stored",
        int(chunk_horizon),
    )

    try:
        while env_steps < max_env_steps and episode_id < max_episodes:
            episode_id += 1
            obs = env.reset()
            task_prompt = str(env.task_description)
            prefetched: dict[str, Any] | None = None
            episode_return = 0.0
            episode_steps = 0
            episode_chunk_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}

            while env_steps < max_env_steps:
                remaining_env_budget = int(max_env_steps - env_steps)
                if remaining_env_budget <= 0:
                    break

                timer.tick("total")
                with timer.context("sample_actions"):
                    if prefetched is None:
                        base_actions, _ = base_policy.infer(obs, prompt=task_prompt)
                        residual_obs = build_chunk_residual_obs(
                            obs=obs,
                            base_actions=base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )
                    else:
                        base_actions = np.asarray(
                            prefetched["base_actions"],
                            dtype=np.float32,
                        )
                        residual_obs = dict(prefetched["residual_obs"])
                        prefetched = None
                    residual_actions = agent.sample_action(
                        residual_obs,
                        deterministic=False,
                    )
                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                execute_horizon = min(int(chunk_horizon), int(remaining_env_budget))
                planned_actions = np.asarray(
                    final_actions[:execute_horizon],
                    dtype=np.float32,
                )
                if int(planned_actions.shape[0]) <= 0:
                    break

                with timer.context("step_env_chunk"):
                    chunk_result = env.step_chunk(planned_actions)

                infos = [dict(info) for info in chunk_result["infos"]]
                observations = [dict(item) for item in chunk_result["observations"]]
                rewards = [float(v) for v in chunk_result["rewards"]]
                dones = [bool(v) for v in chunk_result["dones"]]
                executed_steps = 0
                for info in infos:
                    if bool(info.get("controller_action_executed", True)):
                        executed_steps += 1
                    else:
                        break

                should_log_timer = False
                previous_env_steps = int(env_steps)

                if executed_steps <= 0:
                    done_flag = bool(chunk_result["done"] or chunk_result["truncated"])
                    if not done_flag:
                        raise RuntimeError(
                            "chunk execution produced no executed action and no terminal outcome"
                        )
                    last_info = dict(infos[-1]) if infos else {}
                    episode_return += float(sum(rewards))
                    episode_success = bool(
                        episode_success or last_info.get("success", False)
                    )
                    obs = dict(chunk_result["obs"])
                    timer.tock("total")
                    break

                executed_rewards = rewards[:executed_steps]
                executed_dones = dones[:executed_steps]
                executed_infos = infos[:executed_steps]
                boundary_obs = dict(observations[executed_steps - 1])
                last_info = dict(executed_infos[-1])
                done_flag = bool(
                    executed_dones[-1] or bool(chunk_result["truncated"])
                )

                if done_flag:
                    with timer.context("build_terminal_boundary_obs"):
                        next_residual_obs = _build_terminal_next_residual_obs(
                            obs=boundary_obs,
                            chunk_horizon=int(chunk_horizon),
                            action_dim=int(action_dim),
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )
                    prefetched = None
                else:
                    with timer.context("build_boundary_obs"):
                        next_base_actions, _ = base_policy.infer(
                            boundary_obs,
                            prompt=task_prompt,
                        )
                        next_residual_obs = build_chunk_residual_obs(
                            obs=boundary_obs,
                            base_actions=next_base_actions,
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )
                    prefetched = {
                        "base_actions": np.asarray(next_base_actions, dtype=np.float32),
                        "residual_obs": dict(next_residual_obs),
                    }
                next_residual_obs = dict(next_residual_obs)
                chunk_actions, chunk_action_mask = _pad_chunk_action_and_mask(
                    executed_actions=np.asarray(
                        planned_actions[:executed_steps],
                        dtype=np.float32,
                    ),
                    chunk_horizon=int(chunk_horizon),
                    action_dim=int(action_dim),
                )
                chunk_reward = _compute_discounted_chunk_reward(
                    executed_rewards,
                    discount=float(discount),
                )
                chunk_mask = 0.0 if done_flag else float(
                    float(discount) ** max(0, int(executed_steps) - 1)
                )
                transition = {
                    "episode_id": int(episode_id),
                    "episode_step": int(episode_chunk_steps),
                    "observations": residual_obs,
                    "actions": chunk_actions,
                    "action_mask": chunk_action_mask,
                    "next_observations": next_residual_obs,
                    "rewards": float(chunk_reward),
                    "masks": float(chunk_mask),
                    "dones": bool(done_flag),
                }
                data_store.insert(transition)

                env_steps += int(executed_steps)
                progress_bar.update(int(executed_steps))
                episode_steps += int(executed_steps)
                episode_chunk_steps += 1
                episode_return += float(sum(executed_rewards))
                episode_success = bool(
                    episode_success or last_info.get("success", False)
                )
                obs = dict(boundary_obs)

                if _crossed_interval(
                    previous_count=int(previous_env_steps),
                    current_count=int(env_steps),
                    period=int(steps_per_update),
                ):
                    client.update()
                if _crossed_interval(
                    previous_count=int(previous_env_steps),
                    current_count=int(env_steps),
                    period=int(log_period),
                ):
                    should_log_timer = True

                timer.tock("total")

                if should_log_timer:
                    append_jsonl(
                        actor_timer_log_path,
                        {
                            "source": "actor",
                            "env_steps": int(env_steps),
                            "episode_id": int(episode_id),
                            "episode_steps": int(episode_steps),
                            "episode_chunk_steps": int(episode_chunk_steps),
                            "timer": timer.get_average_times(),
                        },
                    )

                if done_flag or env_steps >= max_env_steps:
                    break

            client.update()
            success_count += int(episode_success)
            recent_episode_successes.append(int(episode_success))
            recent_success_rate_20 = float(sum(recent_episode_successes)) / float(
                max(1, len(recent_episode_successes))
            )
            episode_stats = build_rollout_stats_payload(
                env_steps=int(env_steps),
                rollout=build_rollout_payload(
                    episode_id=int(episode_id),
                    episode_steps=int(episode_steps),
                    episode_return=float(episode_return),
                    success=bool(episode_success),
                    cumulative_success_rate=float(
                        success_count / max(1, episode_id)
                    ),
                    recent_success_rate_20=float(recent_success_rate_20),
                ),
                env_info=last_info,
            )

            append_jsonl(
                rollout_log_path,
                {
                    "source": "rollout",
                    **episode_stats,
                    "episode_chunk_steps": int(episode_chunk_steps),
                },
            )
            client.request("send-stats", episode_stats)
            progress_bar.set_postfix(
                episode=int(episode_id),
                success=int(bool(episode_success)),
                chunks=int(episode_chunk_steps),
                refresh=False,
            )
            logger.info(
                "episode=%s success=%s env_steps=%s chunk_transitions=%s return=%.3f total_env_steps=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                int(episode_chunk_steps),
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
            json.dump(summary, fp, indent=2, ensure_ascii=False)
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
        try:
            base_policy.close()
        except Exception:  # noqa: BLE001
            pass


def learner(cfg: AgiBotTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
    if cfg.offline.enabled:
        raise ValueError(
            "Chunk-boundary training script does not support offline replay mixing yet. "
            "Disable offline.enabled for this variant."
        )

    image_keys = cfg.obs.image_keys
    action_dim = int(cfg.env.action_dim)
    chunk_horizon = int(cfg.residual.chunk_horizon)
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
    replay_buffer = create_chunk_boundary_replay_buffer(
        observation_space=observation_space,
        chunk_action_dim=int(residual_action_spec.chunk_critic_action_dim),
        discount=float(cfg.sac.discount),
        image_keys=image_keys,
        capacity=int(cfg.replay.capacity),
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
    configure_rollout_wandb_metrics(wandb_logger=wandb_logger)
    configure_learner_wandb_metrics(wandb_logger=wandb_logger)

    update_steps = 0
    env_steps = 0
    latest_completed_episode_id = 0
    learner_timer_log_path = run_dir / "learner_timers.jsonl"
    progress_state_lock = Lock()
    summary: dict[str, Any] = {
        "role": "learner",
        "mode": "residual_chunk_boundary",
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
        "timer_log_path": str(learner_timer_log_path),
    }

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal env_steps
        nonlocal latest_completed_episode_id
        if request_type != "send-stats":
            raise ValueError(f"Invalid request type: {request_type}")
        rollout_stats = parse_rollout_stats_payload(payload)
        if rollout_stats is None:
            logger.warning(
                "ignore malformed rollout stats payload: keys=%s",
                sorted(payload.keys()),
            )
            return {}
        with progress_state_lock:
            env_steps = max(int(env_steps), int(rollout_stats["env_steps"]))
            latest_completed_episode_id = max(
                int(latest_completed_episode_id),
                int(rollout_stats["rollout"]["episode_id"]),
            )
        rollout_metrics = extract_rollout_wandb_metrics(rollout_stats)
        if rollout_metrics:
            wandb_logger.log(to_jsonable(rollout_metrics))
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

    logger.info(
        "chunk-boundary learner active: replay_size / training_starts now count chunk "
        "transitions rather than env steps (chunk_horizon=%s)",
        int(chunk_horizon),
    )

    training_starts = int(cfg.training.training_starts)
    if training_starts > 0 and chunk_horizon > 1:
        logger.warning(
            "chunk-boundary variant interprets training.training_starts as chunk transitions; "
            "current config training_starts=%s with chunk_horizon=%s implies roughly %s "
            "executed env steps before learner warmup completes",
            int(training_starts),
            int(chunk_horizon),
            int(training_starts * chunk_horizon),
        )
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
    replay_warmup_poll_interval_sec = 1.0
    idle_poll_interval_sec = 1.0
    timer = Timer()
    last_log_time = time.time()
    last_log_update_steps = int(update_steps)
    initial_network_published = False

    def _sample_training_batch() -> dict[str, Any]:
        batch = replay_buffer.sample(int(cfg.replay.batch_size))
        return reshape_chunk_batch_for_training(batch)

    def _run_training_update() -> dict[str, Any]:
        nonlocal agent
        for _ in range(max(0, critic_actor_ratio - 1)):
            with timer.context("sample_replay_buffer"):
                batch = _sample_training_batch()
            with timer.context("train_critics"):
                agent, _ = agent.update_critics(batch)

        with timer.context("train"):
            batch = _sample_training_batch()
            agent, update_info = agent.update_high_utd(
                batch,
                utd_ratio=cfg.sac.utd_ratio,
            )
        return update_info

    if int(update_steps) < int(max_update_steps) and int(training_starts) > 0:
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
                time.sleep(replay_warmup_poll_interval_sec)
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

    if not initial_network_published:
        server.publish_network(
            snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
        )
        logger.info(
            "publish network: step=%s env_steps=%s reason=initial",
            int(update_steps),
            int(env_steps),
        )
        initial_network_published = True

    try:
        while update_steps < max_update_steps:
            replay_insert_count = int(replay_buffer.latest_data_id())
            if int(update_steps) >= int(replay_insert_count):
                time.sleep(idle_poll_interval_sec)
                continue

            update_info = _run_training_update()
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
                update_metrics = to_jsonable(update_info)
                now = time.time()
                elapsed_sec = max(now - last_log_time, 1e-6)
                updates_since_last_log = max(1, int(update_steps - last_log_update_steps))
                updates_per_sec = float(updates_since_last_log) / float(elapsed_sec)
                last_log_time = now
                last_log_update_steps = int(update_steps)
                timer_metrics = to_jsonable(timer.get_average_times())
                learner_metrics = extract_learner_wandb_metrics(update_metrics)
                if learner_metrics:
                    wandb_logger.log(to_jsonable(learner_metrics), step=update_steps)
                append_jsonl(
                    learner_timer_log_path,
                    {
                        "source": "learner",
                        "update_steps": int(update_steps),
                        "env_steps": int(env_steps),
                        "replay_size": int(len(replay_buffer)),
                        "updates_per_sec": float(updates_per_sec),
                        "timer": timer_metrics,
                    },
                )
                logger.info(
                    format_learner_heartbeat(
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

    finally:
        with progress_state_lock:
            summary_last_completed_episode_id = int(latest_completed_episode_id)
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
                "last_completed_episode_id": int(summary_last_completed_episode_id),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
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
    config_name="train_residual_chunk_boundary",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_residual_chunk_boundary")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    set_global_seeds(typed_cfg.global_seed)

    if typed_cfg.runtime.role == "actor":
        actor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    learner(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
