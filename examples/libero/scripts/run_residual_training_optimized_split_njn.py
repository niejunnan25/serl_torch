from __future__ import annotations

"""Reference-style LIBERO residual DRQ training script."""

from collections import deque
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
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

from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.agent_acceleration import apply_torch_compile
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_actor_network_payload
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.common.trainer_session import TrainerClientSession
from serl_launcher.async_eval import append_async_eval_checkpoint_index
from serl_launcher.async_eval import save_async_eval_checkpoint_payload
from serl_launcher.common.training_observability import configure_eval_wandb_metrics
from serl_launcher.common.training_observability import configure_learner_wandb_metrics
from serl_launcher.common.training_observability import configure_rollout_wandb_metrics
from serl_launcher.common.training_observability import extract_learner_wandb_metrics
from serl_launcher.common.training_observability import extract_rollout_wandb_metrics
from serl_launcher.common.training_payloads import build_actor_progress_payload
from serl_launcher.common.training_payloads import build_rollout_payload
from serl_launcher.common.training_payloads import build_rollout_stats_payload
from serl_launcher.common.training_payloads import parse_actor_progress_payload
from serl_launcher.common.training_payloads import parse_rollout_stats_payload
from serl_launcher.common.training_reporting import format_learner_heartbeat
from serl_launcher.common.training_reporting import sync_eval_results_to_wandb
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.policy.typed_factory import build_policy_client
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import sample_mixed_training_batch
from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import build_chunk_residual_observation_space
from serl_launcher.residual.observation import build_chunk_residual_sample_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.rollout import ProcessorClient
from serl_launcher.rollout import ProcessorServer
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.config import LiberoTrainConfig
from serl_torch.examples.libero.config import cfg_to_log_payload
from serl_torch.examples.libero.config import get_runtime_role
from serl_torch.examples.libero.config import parse_train_cfg_allow_processor
from serl_torch.examples.libero.env.factory import create_env
from serl_torch.examples.libero.env.observation import build_libero_state
from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.observation import LIBERO_STATE_DIM
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_HEIGHT
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_WIDTH
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input
from serl_torch.examples.libero.env.offline_data import load_prepared_offline_replay
from serl_torch.examples.libero.env.offline_data import resolve_and_validate_prepared_paths
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    append_async_eval_request,
)
from serl_torch.examples.libero.runtime.async_eval_runtime import append_async_eval_stop
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    check_async_eval_worker,
)
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    load_new_async_eval_results,
)
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    start_async_eval_worker,
)
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    summarize_async_eval_results,
)
from serl_torch.examples.libero.runtime.async_eval_runtime import (
    wait_for_async_eval_worker,
)
from serl_torch.examples.libero.runtime.transition_assembly import (
    BatchAwareLiberoTransitionAssembler,
)
from serl_torch.examples.libero.runtime.processor_protocol import (
    build_processor_submission_payload,
)
from serl_torch.examples.libero.runtime.processor_protocol import (
    normalize_chunk_result,
)
from serl_torch.examples.libero.runtime.processor_protocol import (
    reconstruct_chunk_execution_record,
)

TRAINER_REQUEST_TYPES = ("send-stats", "actor-progress")


def actor(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    env = create_env(cfg, logger)
    task_prompt = str(env.task_description)
    policy_client = build_policy_client(cfg, logger=logger)
    policy_backend = describe_policy_backend(cfg)
    logger.info("Chunk policy backend: %s", policy_backend)

    processor_client = ProcessorClient(
        transport_config=cfg.runtime.processor_transport,
        logger=logger,
    )

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_alpha = cfg.residual.alpha
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
    runtime_cfg = cfg.runtime
    trainer_transport_cfg = runtime_cfg.trainer_transport

    sample_obs = build_chunk_residual_sample_obs(
        state_dim=LIBERO_STATE_DIM,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        image_height=RESIDUAL_IMAGE_HEIGHT,
        image_width=RESIDUAL_IMAGE_WIDTH,
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
        runtime_cfg.trainer_host,
        TrainerConfig(
            port_number=runtime_cfg.trainer_port,
            broadcast_port=runtime_cfg.broadcast_port,
            request_types=list(TRAINER_REQUEST_TYPES),
            transport_mode=trainer_transport_cfg.mode,
            data_port=trainer_transport_cfg.data_port,
            control_timeout_ms=trainer_transport_cfg.control_timeout_ms,
            data_queue_capacity=trainer_transport_cfg.data_queue_capacity,
            data_socket_hwm=trainer_transport_cfg.data_socket_hwm,
            commit_poll_ms=trainer_transport_cfg.commit_poll_ms,
        ),
        data_store=data_store,
        wait_for_server=True,
        log_level=logger.level,
    )

    trainer_session = TrainerClientSession(
        client=client,
        logger=logger,
        store_name="actor_env",
        status_fallback=lambda: {"transport_mode": trainer_transport_cfg.mode},
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
    chunk_seq = 0
    recent_episode_successes: deque[int] = deque(maxlen=20)
    actor_timer_log_path = run_dir / "actor_timers.jsonl"
    rollout_log_path = run_dir / str(
        cfg.logging.episode_log_file or "episode_logs.jsonl"
    )
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "residual",
        "transport_mode": trainer_transport_cfg.mode,
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
        "chunks_sent": 0,
        "timer_log_path": str(actor_timer_log_path),
        "episode_log_path": str(rollout_log_path),
    }

    processor_client.wait_until_ready()

    progress_bar = tqdm(
        total=max_env_steps,
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
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            episode_last_chunk_seq: int | None = None
            last_info: dict[str, Any] = {}

            while env_steps < max_env_steps:
                timer.tick("total")
                with timer.context("prepare_action_chunk"):
                    base_policy_input = build_libero_policy_input(obs, task_prompt)

                    base_actions, _ = policy_client.infer(base_policy_input)
                    base_actions = prepare_base_actions_chunk(
                        base_actions=base_actions,
                        chunk_horizon=chunk_horizon,
                    )
                    residual_obs = build_chunk_residual_obs(
                        robot_state=build_libero_state(obs),
                        images=extract_libero_images(obs),
                        image_keys=image_keys,
                        base_actions=base_actions,
                        residual_alpha=residual_alpha,
                    )
                    residual_actions = agent.sample_action(
                        residual_obs,
                        deterministic=False,
                    )
                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                should_log_timer = False
                remaining_env_steps = max(0, int(max_env_steps - env_steps))
                if remaining_env_steps <= 0:
                    timer.tock("total")
                    break

                action_chunk = final_actions[:remaining_env_steps]

                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)
                normalized_chunk = normalize_chunk_result(dict(chunk_result))

                current_chunk_seq = int(chunk_seq)
                submit_payload = build_processor_submission_payload(
                    chunk_seq=int(current_chunk_seq),
                    episode_id=int(episode_id),
                    episode_step_start=int(episode_steps),
                    task_prompt=task_prompt,
                    chunk_result=chunk_result,
                )
                with timer.context("submit_processor_chunk"):
                    processor_client.submit(
                        payload=submit_payload,
                        context=(
                            f"episode_{int(episode_id)}_chunk_{int(current_chunk_seq)}"
                        ),
                    )
                chunk_seq += 1
                episode_last_chunk_seq = int(current_chunk_seq)

                executed_steps = int(normalized_chunk.executed_steps)
                previous_env_steps = int(env_steps)
                env_steps += int(executed_steps)
                progress_bar.update(int(executed_steps))
                episode_steps += int(executed_steps)
                episode_return += float(normalized_chunk.reward_sum)
                infos = list(normalized_chunk.infos)
                episode_success = bool(
                    episode_success
                    or any(bool(info.get("env_done", False)) for info in infos)
                )
                last_info = dict(normalized_chunk.chunk_info)
                obs = dict(normalized_chunk.final_obs)
                episode_done = bool(
                    normalized_chunk.chunk_done or normalized_chunk.chunk_truncated
                )

                for step_offset in range(1, int(executed_steps) + 1):
                    next_env_step = int(previous_env_steps + step_offset)
                    if next_env_step % steps_per_update == 0:
                        trainer_session.update(
                            context=f"env_step_{int(next_env_step)}",
                            failure_message=(
                                "trainer transport update failed repeatedly; "
                                "aborting actor run"
                            ),
                        )
                    if next_env_step % log_period == 0:
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
                            "chunk_seq": int(current_chunk_seq),
                            "timer": timer.get_average_times(),
                            "transport": trainer_session.status(),
                            "processor": processor_client.last_status(),
                        },
                    )

                if episode_done:
                    break

            processor_client.finish(
                last_chunk_seq=episode_last_chunk_seq,
                episode_id=int(episode_id),
            )
            trainer_session.update(
                context="episode_end",
                failure_message=(
                    "trainer transport update failed repeatedly; aborting actor run"
                ),
            )
            trainer_session.request(
                "actor-progress",
                build_actor_progress_payload(
                    env_steps=int(env_steps),
                    episode_id=int(episode_id),
                    actor_done=bool(int(env_steps) >= int(max_env_steps)),
                ),
                context=f"episode_{int(episode_id)}_progress",
                failure_message=(
                    "trainer transport actor-progress failed repeatedly; "
                    "aborting actor run"
                ),
                retry_until_exhausted=True,
            )
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
                    init_episode_idx=int(init_episode_idx),
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
                    "transport": trainer_session.status(),
                    "processor": processor_client.last_status(),
                },
            )
            trainer_session.request(
                "send-stats",
                episode_stats,
                context=f"episode_{int(episode_id)}_stats",
                raise_on_exhaustion=False,
            )
            progress_bar.set_postfix(
                episode=int(episode_id),
                success=int(bool(episode_success)),
                refresh=False,
            )
            logger.info(
                "episode=%s success=%s steps=%s return=%.3f env_steps=%s chunks_sent=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(env_steps),
                int(chunk_seq),
            )

    finally:
        try:
            processor_client.shutdown(
                last_chunk_seq=(None if int(chunk_seq) <= 0 else int(chunk_seq - 1)),
            )
            trainer_session.update(
                context="shutdown",
                failure_message=(
                    "trainer transport update failed repeatedly; aborting actor run"
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
                "chunks_sent": int(chunk_seq),
                "transport": trainer_session.status(),
                "processor": processor_client.last_status(),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            client.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            processor_client.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass
        if cfg.env.backend != "remote":
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


def processor(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    runtime_cfg = cfg.runtime
    trainer_transport_cfg = runtime_cfg.trainer_transport
    if cfg.backfill_policy.enabled:
        endpoint_cfg = SimpleNamespace(
            policy=SimpleNamespace(
                type=cfg.policy.type,
                host=cfg.backfill_policy.host,
                port=cfg.backfill_policy.port,
            ),
            env=SimpleNamespace(action_dim=cfg.env.action_dim),
        )
        policy_client = build_policy_client(endpoint_cfg, logger=logger)
        backfill_backend = (
            f"{cfg.policy.type}:{cfg.backfill_policy.host}:"
            f"{cfg.backfill_policy.port}"
        )
    else:
        policy_client = build_policy_client(cfg, logger=logger)
        backfill_backend = describe_policy_backend(cfg)
    logger.info("Processor backfill backend: %s", backfill_backend)

    assembler = BatchAwareLiberoTransitionAssembler(
        policy_client=policy_client,
        chunk_horizon=cfg.residual.chunk_horizon,
        image_keys=cfg.obs.image_keys,
        residual_alpha=cfg.residual.alpha,
    )

    data_store = QueuedDataStore(cfg.runtime.data_store_queue_size)
    client = TrainerClient(
        "actor_env",
        runtime_cfg.trainer_host,
        TrainerConfig(
            port_number=runtime_cfg.trainer_port,
            broadcast_port=runtime_cfg.broadcast_port,
            request_types=list(TRAINER_REQUEST_TYPES),
            transport_mode=trainer_transport_cfg.mode,
            data_port=trainer_transport_cfg.data_port,
            control_timeout_ms=trainer_transport_cfg.control_timeout_ms,
            data_queue_capacity=trainer_transport_cfg.data_queue_capacity,
            data_socket_hwm=trainer_transport_cfg.data_socket_hwm,
            commit_poll_ms=trainer_transport_cfg.commit_poll_ms,
        ),
        data_store=data_store,
        wait_for_server=True,
        log_level=logger.level,
    )
    trainer_session = TrainerClientSession(
        client=client,
        logger=logger,
        store_name="actor_env",
        status_fallback=lambda: {"transport_mode": trainer_transport_cfg.mode},
        log_prefix="processor trainer transport",
    )

    committed_env_steps = 0
    processor_timer_log_path = run_dir / "processor_timers.jsonl"
    timer = Timer()
    steps_per_update = cfg.training.steps_per_update
    log_period = cfg.training.log_period
    summary: dict[str, Any] = {
        "role": "processor",
        "mode": "residual",
        "transport_mode": trainer_transport_cfg.mode,
        "accepted_chunk_seq": -1,
        "processed_chunk_seq": -1,
        "committed_env_steps": 0,
        "timer_log_path": str(processor_timer_log_path),
    }

    processor_server = ProcessorServer(
        transport_config=cfg.runtime.processor_transport,
        transport_status_fn=trainer_session.status,
        flush_transport_fn=lambda context, wait_until_committed: trainer_session.flush(
            context=context,
            wait_until_committed=wait_until_committed,
            update_failure_message=(
                "processor trainer transport update failed repeatedly; "
                "aborting processor run"
            ),
            wait_timeout_message=(
                f"processor wait_until_committed timed out: context={str(context)}"
            ),
        ),
        wait_committed_on_episode_end=trainer_transport_cfg.wait_committed_on_episode_end,
        wait_committed_on_shutdown=trainer_transport_cfg.wait_committed_on_shutdown,
        logger=logger,
    )
    processor_server.start()

    try:
        while True:
            payload = processor_server.get_chunk(timeout_s=0.1)
            if payload is None:
                if processor_server.should_stop():
                    break
                continue

            chunk_seq_value = int(payload["chunk_seq"])
            should_log_timer = False
            try:
                timer.tick("total")
                with timer.context("reconstruct_raw_chunk"):
                    raw_chunk = reconstruct_chunk_execution_record(
                        payload=payload,
                        assembler=assembler,
                    )
                with timer.context("assemble_transitions"):
                    assembled_chunk = assembler.process_chunk(
                        raw=raw_chunk,
                        task_prompt=str(payload["task_prompt"]),
                    )
                previous_committed_env_steps = int(committed_env_steps)
                with timer.context("commit_replay"):
                    for transition in assembled_chunk.transitions:
                        data_store.insert(transition)
                    for step_offset in range(1, assembled_chunk.env_steps_delta + 1):
                        next_committed_env_step = int(
                            previous_committed_env_steps + step_offset
                        )
                        if next_committed_env_step % steps_per_update == 0:
                            trainer_session.update(
                                context=(
                                    f"processor_commit_step_"
                                    f"{int(next_committed_env_step)}"
                                ),
                                failure_message=(
                                    "processor trainer transport update failed "
                                    "repeatedly; aborting processor run"
                                ),
                            )
                        if next_committed_env_step % log_period == 0:
                            should_log_timer = True
                    committed_env_steps = int(
                        previous_committed_env_steps + assembled_chunk.env_steps_delta
                    )
                timer.tock("total")
                processor_server.mark_chunk_committed(chunk_seq=int(chunk_seq_value))
                if should_log_timer:
                    append_jsonl(
                        processor_timer_log_path,
                        {
                            "source": "processor",
                            "chunk_seq": int(chunk_seq_value),
                            "committed_env_steps": int(committed_env_steps),
                            "timer": timer.get_average_times(),
                            "transport": trainer_session.status(),
                            "processor": processor_server.status_snapshot(),
                        },
                    )
            except Exception:
                logger.exception(
                    "processor failed: chunk_seq=%s episode_id=%s",
                    int(chunk_seq_value),
                    int(payload.get("episode_id", -1)),
                )
                raise
            finally:
                processor_server.task_done()
    except KeyboardInterrupt:
        logger.info("processor interrupted; shutting down gracefully")
    finally:
        try:
            processor_server.request_stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            trainer_session.flush(
                context="processor_finally",
                wait_until_committed=False,
                update_failure_message=(
                    "processor trainer transport update failed repeatedly; "
                    "aborting processor run"
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        summary.update(
            {
                "accepted_chunk_seq": int(
                    processor_server.status_snapshot()["accepted_chunk_seq"]
                ),
                "processed_chunk_seq": int(
                    processor_server.status_snapshot()["processed_chunk_seq"]
                ),
                "committed_env_steps": int(committed_env_steps),
                "transport": trainer_session.status(),
                "processor": processor_server.status_snapshot(),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            processor_server.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            client.stop()
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
    runtime_cfg = cfg.runtime
    trainer_transport_cfg = runtime_cfg.trainer_transport
    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(
        cfg,
        action_dim=action_dim,
    )
    sample_obs = build_chunk_residual_sample_obs(
        state_dim=LIBERO_STATE_DIM,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        image_height=RESIDUAL_IMAGE_HEIGHT,
        image_width=RESIDUAL_IMAGE_WIDTH,
    )
    agent = create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=image_keys,
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )
    agent = apply_torch_compile(
        agent,
        compile_cfg=cfg.training.torch_compile,
    )
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
    replay_buffer = create_chunk_replay_buffer(
        observation_space=observation_space,
        action_dim=cfg.env.action_dim,
        chunk_horizon=cfg.residual.chunk_horizon,
        discount=cfg.sac.discount,
        image_keys=image_keys,
        capacity=cfg.replay.capacity,
    )
    offline_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore | None = None
    offline_prepared_path: Path | None = None
    offline_manifest_path: Path | None = None
    offline_validation_stats: dict[str, Any] | None = None
    offline_load_stats: dict[str, Any] = {
        "files_total": 0,
        "episodes_loaded": 0,
        "steps_loaded": 0,
        "load_errors": 0,
    }
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
    configure_eval_wandb_metrics(wandb_logger=wandb_logger)
    configure_learner_wandb_metrics(wandb_logger=wandb_logger)
    async_eval = start_async_eval_worker(
        cfg,
        run_dir=run_dir,
        logger=logger,
    )

    update_steps = 0
    env_steps = 0
    latest_completed_episode_id = 0
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0
    learner_timer_log_path = run_dir / "learner_timers.jsonl"
    progress_state_lock = Lock()
    summary: dict[str, Any] = {
        "role": "learner",
        "mode": "residual",
        "transport_mode": trainer_transport_cfg.mode,
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
        "timer_log_path": str(learner_timer_log_path),
    }
    actor_reported_env_steps = 0
    actor_done = False

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(server.get_transport_status("actor_env"))
        except Exception:  # noqa: BLE001
            return {"transport_mode": trainer_transport_cfg.mode}

    def _committed_online_steps() -> int:
        return int(replay_buffer.latest_data_id())

    def _should_stop_after_actor_done(*, online_update_steps: int) -> bool:
        if not bool(actor_done):
            return False
        target_env_steps = int(actor_reported_env_steps)
        if target_env_steps <= 0:
            return False
        committed_online_steps = int(_committed_online_steps())
        if int(online_update_steps) < int(committed_online_steps):
            return False
        transport_status = _transport_status()
        accepted = int(transport_status.get("accepted_update_id", -1))
        committed = int(transport_status.get("committed_update_id", -1))
        target_last_online_id = max(0, int(target_env_steps) - 1)
        if accepted < int(target_last_online_id) or committed < int(target_last_online_id):
            return False
        if accepted >= 0 and committed >= 0 and accepted > committed:
            return False
        return True

    def _record_actor_progress(*, reported_env_steps: int, episode_id: int) -> None:
        nonlocal env_steps
        nonlocal latest_completed_episode_id
        env_steps = max(int(env_steps), int(reported_env_steps))
        if int(episode_id) > 0:
            latest_completed_episode_id = max(
                int(latest_completed_episode_id),
                int(episode_id),
            )
            completed_episode_env_steps[int(episode_id)] = max(
                int(completed_episode_env_steps.get(int(episode_id), 0)),
                int(reported_env_steps),
            )

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal actor_reported_env_steps
        nonlocal actor_done
        if request_type != "send-stats":
            if request_type != "actor-progress":
                raise ValueError(f"Invalid request type: {request_type}")
            actor_progress = parse_actor_progress_payload(payload)
            if actor_progress is None:
                logger.warning(
                    "ignore malformed actor progress payload: keys=%s",
                    sorted(payload.keys()),
                )
                return {}
            with progress_state_lock:
                actor_reported_env_steps = max(
                    int(actor_reported_env_steps),
                    int(actor_progress["env_steps"]),
                )
                actor_done = bool(actor_done or actor_progress["actor_done"])
                _record_actor_progress(
                    reported_env_steps=int(actor_progress["env_steps"]),
                    episode_id=int(actor_progress["episode_id"]),
                )
            return {}
        rollout_stats = parse_rollout_stats_payload(payload)
        if rollout_stats is None:
            logger.warning(
                "ignore malformed rollout stats payload: keys=%s",
                sorted(payload.keys()),
            )
            return {}
        with progress_state_lock:
            _record_actor_progress(
                reported_env_steps=int(rollout_stats["env_steps"]),
                episode_id=int(rollout_stats["rollout"]["episode_id"]),
            )
        rollout_metrics = extract_rollout_wandb_metrics(rollout_stats)
        if rollout_metrics:
            wandb_logger.log(to_jsonable(rollout_metrics))
        return {}

    server = TrainerServer(
        TrainerConfig(
            port_number=runtime_cfg.trainer_port,
            broadcast_port=runtime_cfg.broadcast_port,
            request_types=list(TRAINER_REQUEST_TYPES),
            transport_mode=trainer_transport_cfg.mode,
            data_port=trainer_transport_cfg.data_port,
            control_timeout_ms=trainer_transport_cfg.control_timeout_ms,
            data_queue_capacity=trainer_transport_cfg.data_queue_capacity,
            data_socket_hwm=trainer_transport_cfg.data_socket_hwm,
            commit_poll_ms=trainer_transport_cfg.commit_poll_ms,
        ),
        request_callback=stats_callback,
        log_level=logger.level,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    if cfg.offline.enabled:
        offline_resolution = resolve_and_validate_prepared_paths(
            cfg,
            logger=logger,
        )
        if offline_resolution.prepared_paths:
            offline_prepared_path = offline_resolution.prepared_paths[0]
        if offline_resolution.manifest_paths:
            offline_manifest_path = offline_resolution.manifest_paths[0]
        offline_validation_stats = dict(offline_resolution.validation_stats)
        offline_replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=cfg.env.action_dim,
            chunk_horizon=cfg.residual.chunk_horizon,
            discount=cfg.sac.discount,
            image_keys=image_keys,
            capacity=cfg.offline.capacity,
        )
        offline_load_stats = load_prepared_offline_replay(
            replay_buffer=offline_replay_buffer,
            prepared_paths=tuple()
            if offline_prepared_path is None
            else (offline_prepared_path,),
            logger=logger,
            max_episodes=cfg.offline.load_max_episodes,
            max_transitions=cfg.offline.load_max_transitions,
        )
        if len(offline_replay_buffer) <= 0:
            logger.warning(
                "offline replay prepared but empty; continuing with online-only training"
            )
            offline_replay_buffer = None
        else:
            logger.info(
                "offline replay ready: prepared_path=%s replay_size=%s ratio=%.3f pretrain_steps=%s",
                None if offline_prepared_path is None else str(offline_prepared_path),
                int(len(offline_replay_buffer)),
                cfg.offline.ratio,
                cfg.offline.pretrain_steps,
            )

    training_starts = cfg.training.training_starts
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
    replay_warmup_poll_interval_sec = 1.0
    idle_poll_interval_sec = 1.0
    timer = Timer()
    last_log_time = time.time()
    last_log_update_steps = int(update_steps)
    offline_pretrain_steps_done = 0
    initial_network_published = False

    def _run_training_update(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal agent
        train_batch_mix = {
            "online_batch_size": cfg.replay.batch_size,
            "offline_batch_size": 0,
        }
        for _ in range(max(0, critic_actor_ratio - 1)):
            with timer.context("sample_replay_buffer"):
                batch, _ = sample_mixed_training_batch(
                    online_replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    batch_size=cfg.replay.batch_size,
                    offline_ratio=offline_ratio,
                )
            with timer.context("train_critics"):
                agent, _critics_info = agent.update_critics(batch)

        with timer.context("train"):
            batch, train_batch_mix = sample_mixed_training_batch(
                online_replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                batch_size=cfg.replay.batch_size,
                offline_ratio=offline_ratio,
            )
            agent, update_info = agent.update_high_utd(
                batch,
                utd_ratio=cfg.sac.utd_ratio,
            )
        return update_info, train_batch_mix

    def _maybe_queue_async_eval() -> None:
        nonlocal last_queued_async_eval_episode
        if (not async_eval.enabled) or async_eval.eval_checkpoint_dir is None:
            return
        every_episodes = int(async_eval.every_episodes)
        if every_episodes <= 0:
            return
        while True:
            with progress_state_lock:
                next_target_episode = (
                    int(last_queued_async_eval_episode) + every_episodes
                )
                if int(latest_completed_episode_id) < int(next_target_episode):
                    return
                target_episode = int(next_target_episode)
                target_env_step = int(
                    completed_episode_env_steps.get(target_episode, env_steps)
                )
            async_eval_checkpoint_payload = snapshot_agent_checkpoint_payload(
                agent,
                step=int(update_steps),
            )
            async_eval_checkpoint_path = save_async_eval_checkpoint_payload(
                async_eval.eval_checkpoint_dir,
                async_eval_checkpoint_payload,
                episode_id=int(target_episode),
            )
            append_async_eval_checkpoint_index(
                async_eval.eval_checkpoint_dir,
                episode_id=int(target_episode),
                checkpoint_step=int(update_steps),
                checkpoint_path=async_eval_checkpoint_path,
            )
            append_async_eval_request(
                async_eval,
                {
                    "eval_index": int(async_eval.triggered_count),
                    "train_episode_id": int(target_episode),
                    "train_update_step": int(update_steps),
                    "train_env_step": int(target_env_step),
                    "checkpoint_step": int(update_steps),
                    "checkpoint_path": str(async_eval_checkpoint_path),
                },
            )
            with progress_state_lock:
                last_queued_async_eval_episode = max(
                    int(last_queued_async_eval_episode),
                    int(target_episode),
                )
                stale_episode_ids = [
                    episode_id
                    for episode_id in completed_episode_env_steps
                    if int(episode_id) <= int(target_episode)
                ]
                for stale_episode_id in stale_episode_ids:
                    completed_episode_env_steps.pop(int(stale_episode_id), None)
            logger.info(
                "queued eval: eval_index=%s episode=%s update_steps=%s env_steps=%s checkpoint=%s",
                int(max(0, async_eval.triggered_count - 1)),
                int(target_episode),
                int(update_steps),
                int(target_env_step),
                async_eval_checkpoint_path,
            )

    if (
        offline_replay_buffer is not None
        and cfg.offline.pretrain_steps > 0
        and int(update_steps) < int(max_update_steps)
    ):
        logger.info(
            "starting offline pretrain: steps=%s offline_replay_size=%s",
            cfg.offline.pretrain_steps,
            int(len(offline_replay_buffer)),
        )
        pretrain_bar = tqdm(
            total=min(cfg.offline.pretrain_steps, max_update_steps),
            desc="learner offline pretrain",
            dynamic_ncols=True,
            leave=True,
        )
        last_pretrain_info: dict[str, Any] | None = None
        try:
            while (
                int(update_steps) < int(max_update_steps)
                and int(offline_pretrain_steps_done) < cfg.offline.pretrain_steps
            ):
                last_pretrain_info, _ = _run_training_update(offline_ratio=1.0)
                update_steps += 1
                offline_pretrain_steps_done += 1
                pretrain_bar.update(1)
        finally:
            pretrain_bar.close()

        if last_pretrain_info is not None:
            pretrain_metrics = extract_learner_wandb_metrics(last_pretrain_info)
            if pretrain_metrics:
                wandb_logger.log(to_jsonable(pretrain_metrics), step=update_steps)
        logger.info(
            "offline pretrain complete: completed=%s update_steps=%s offline_replay_size=%s",
            int(offline_pretrain_steps_done),
            int(update_steps),
            0 if offline_replay_buffer is None else int(len(offline_replay_buffer)),
        )
        server.publish_network(
            snapshot_actor_network_payload(agent, step=int(update_steps))
        )
        initial_network_published = True
        logger.info(
            "publish network: step=%s env_steps=%s reason=offline_pretrain_complete",
            int(update_steps),
            int(env_steps),
        )

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
            snapshot_actor_network_payload(agent, step=int(update_steps))
        )
        logger.info(
            "publish network: step=%s env_steps=%s reason=initial",
            int(update_steps),
            int(env_steps),
        )

    interrupted = False
    try:
        while update_steps < max_update_steps:
            _maybe_queue_async_eval()
            online_update_steps = max(0, int(update_steps - offline_pretrain_steps_done))
            if _should_stop_after_actor_done(online_update_steps=int(online_update_steps)):
                logger.info(
                    "stopping learner after actor env limit: update_steps=%s env_steps=%s replay_size=%s transport=%s",
                    int(update_steps),
                    int(env_steps),
                    int(len(replay_buffer)),
                    _transport_status(),
                )
                break
            if not online_update_steps < _committed_online_steps():
                time.sleep(idle_poll_interval_sec)
                continue

            update_info, batch_mix = _run_training_update(
                offline_ratio=cfg.offline.ratio,
            )
            update_steps += 1
            _maybe_queue_async_eval()

            if update_steps % steps_per_update == 0:
                server.publish_network(
                    snapshot_actor_network_payload(agent, step=int(update_steps))
                )
                logger.info(
                    "publish network: step=%s env_steps=%s reason=periodic",
                    int(update_steps),
                    int(env_steps),
                )

            if update_steps % log_period == 0:
                check_async_eval_worker(async_eval, logger=logger)
                sync_eval_results_to_wandb(
                    records=load_new_async_eval_results(async_eval),
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
                        "transport": _transport_status(),
                    },
                )
                offline_suffix = ""
                if offline_replay_buffer is not None:
                    offline_total = max(
                        1,
                        int(batch_mix["online_batch_size"])
                        + int(batch_mix["offline_batch_size"]),
                    )
                    offline_ratio_actual = float(batch_mix["offline_batch_size"]) / float(
                        offline_total
                    )
                    offline_suffix = (
                        " "
                        f"offline_replay={int(len(offline_replay_buffer))} "
                        f"offline_ratio_actual={offline_ratio_actual:.3f} "
                        f"offline_batch={int(batch_mix['offline_batch_size'])}/{offline_total} "
                        f"online_updates={max(0, int(update_steps - offline_pretrain_steps_done))} "
                        f"offline_pretrain={int(offline_pretrain_steps_done)}"
                    )
                logger.info(
                    format_learner_heartbeat(
                        update_steps=int(update_steps),
                        env_steps=int(env_steps),
                        replay_size=int(len(replay_buffer)),
                        updates_per_sec=float(updates_per_sec),
                        update_info=dict(update_metrics),
                        offline_suffix=offline_suffix,
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
    except KeyboardInterrupt:
        interrupted = True
        logger.info("learner interrupted; shutting down gracefully")

    finally:
        _maybe_queue_async_eval()
        async_eval_return_code = None
        if async_eval.enabled:
            append_async_eval_stop(async_eval)
            async_eval_return_code = wait_for_async_eval_worker(
                async_eval,
                logger=logger,
            )
            sync_eval_results_to_wandb(
                records=load_new_async_eval_results(async_eval),
                wandb_logger=wandb_logger,
                logger=logger,
            )
            if async_eval_return_code not in (None, 0):
                logger.warning(
                    "eval worker exited with returncode=%s; see %s",
                    async_eval_return_code,
                    async_eval.worker_log_path,
                )
        async_eval_counts = summarize_async_eval_results(async_eval.summary_jsonl_path)
        with progress_state_lock:
            summary_last_completed_episode_id = int(latest_completed_episode_id)
            summary_last_queued_episode_id = int(last_queued_async_eval_episode)
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
                "transport": _transport_status(),
                "offline": {
                    "enabled": cfg.offline.enabled,
                    "ratio": cfg.offline.ratio,
                    "prepared_path": (
                        None
                        if offline_prepared_path is None
                        else str(offline_prepared_path)
                    ),
                    "manifest_path": (
                        None
                        if offline_manifest_path is None
                        else str(offline_manifest_path)
                    ),
                    "validation_stats": (
                        None
                        if offline_validation_stats is None
                        else to_jsonable(offline_validation_stats)
                    ),
                    "load_stats": to_jsonable(offline_load_stats),
                    "replay_size": (
                        0
                        if offline_replay_buffer is None
                        else int(len(offline_replay_buffer))
                    ),
                    "pretrain_steps_requested": cfg.offline.pretrain_steps,
                    "pretrain_steps_completed": int(offline_pretrain_steps_done),
                },
                "eval": {
                    "enabled": bool(async_eval.enabled),
                    "every_episodes": int(async_eval.every_episodes),
                    "triggered": int(async_eval.triggered_count),
                    "last_completed_episode_id": int(summary_last_completed_episode_id),
                    "last_queued_episode_id": int(summary_last_queued_episode_id),
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
    if interrupted:
        return


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_residual_optimized",
)
def main(cfg: DictConfig) -> None:
    raw_role = get_runtime_role(cfg)
    typed_cfg = parse_train_cfg_allow_processor(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_residual")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Runtime role: %s", raw_role)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    set_global_seeds(typed_cfg.global_seed)

    if raw_role == "actor":
        actor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    if raw_role == "processor":
        processor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    learner(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
