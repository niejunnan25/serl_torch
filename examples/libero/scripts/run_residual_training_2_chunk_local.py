from __future__ import annotations

"""Reference-style LIBERO residual DRQ training script."""

from collections import deque
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

from agentlace.data.data_store import QueuedDataStore
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
from serl_launcher.common.trainer_transport import build_actor_trainer_transport
from serl_launcher.common.trainer_transport import build_learner_trainer_transport
from serl_launcher.async_eval import append_async_eval_checkpoint_index
from serl_launcher.async_eval import save_async_eval_checkpoint_payload
from serl_launcher.common.training_observability import configure_eval_wandb_metrics
from serl_launcher.common.training_observability import configure_learner_wandb_metrics
from serl_launcher.common.training_observability import configure_rollout_wandb_metrics
from serl_launcher.common.training_observability import extract_learner_wandb_metrics
from serl_launcher.common.training_observability import extract_rollout_wandb_metrics
from serl_launcher.common.training_payloads import build_rollout_payload
from serl_launcher.common.training_payloads import build_rollout_stats_payload
from serl_launcher.common.training_payloads import parse_rollout_stats_payload
from serl_launcher.common.training_reporting import format_learner_heartbeat
from serl_launcher.common.training_reporting import sync_eval_results_to_wandb
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.policy.typed_factory import build_policy_client
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import PrefetchingMixedBatchSampler
from serl_launcher.residual.chunk_window_replay import ProfileAccumulator
from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import build_chunk_residual_observation_space
from serl_launcher.residual.observation import build_chunk_residual_sample_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec
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
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.env.factory import create_env
from serl_torch.examples.libero.env.observation import build_libero_state
from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.observation import LIBERO_STATE_DIM
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_HEIGHT
from serl_torch.examples.libero.env.observation import RESIDUAL_IMAGE_WIDTH
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input
from serl_torch.examples.libero.env.offline_data import load_prepared_offline_replay
from serl_torch.examples.libero.env.offline_data import (
    resolve_and_validate_prepared_paths,
)
from serl_torch.examples.libero.runtime.raw_rollout_recorder import RawRolloutRecorder
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
    AssemblyResult,
)
from serl_torch.examples.libero.runtime.transition_assembly import (
    ChunkExecutionRecord,
)
from serl_torch.examples.libero.runtime.transition_assembly import (
    LiberoActorTransitionAssembler,
)


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

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_alpha = float(cfg.residual.alpha)
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
    transition_assembler = LiberoActorTransitionAssembler(
        cfg=cfg,
        policy_client=policy_client,
        logger=logger,
    )
    # Optimized actor dataflow:
    # chunk execute -> post-hoc step transition assembly -> step-window replay.
    # Replay still stores per-step transitions; this is not direct chunk replay.
    if transition_assembler.async_transition_assembly_enabled:
        logger.info(
            "Async backfill enabled: mode=%s endpoint=%s:%s max_pending_chunks=%s",
            str(cfg.backfill_policy.mode),
            str(cfg.backfill_policy.host),
            int(cfg.backfill_policy.port),
            int(cfg.backfill_policy.max_pending_chunks),
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

    data_store = QueuedDataStore(cfg.runtime.data_store_queue_size)

    client = build_actor_trainer_transport(
        store_name="actor_env",
        server_ip=cfg.runtime.trainer_host,
        trainer_port=cfg.runtime.trainer_port,
        broadcast_port=cfg.runtime.broadcast_port,
        transport_cfg=cfg.runtime.trainer_transport,
        data_store=data_store,
        request_types=("send-stats",),
        wait_for_server=True,
        log_level=logger.level,
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
    committed_env_steps = 0
    episode_id = 0
    success_count = 0
    current_task_prompt: str | None = None
    recent_episode_successes: deque[int] = deque(maxlen=20)
    actor_timer_log_path = run_dir / "actor_timers.jsonl"
    rollout_log_path = run_dir / str(
        cfg.logging.episode_log_file or "episode_logs.jsonl"
    )
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "residual",
        "transport_mode": str(cfg.runtime.trainer_transport.mode),
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
        "timer_log_path": str(actor_timer_log_path),
        "episode_log_path": str(rollout_log_path),
    }
    recycle_output_root = Path(cfg.recycle.output_root)
    if not recycle_output_root.is_absolute():
        recycle_output_root = (run_dir / recycle_output_root).resolve()
    recycle_recorder: RawRolloutRecorder | None = None
    if cfg.recycle.enabled:
        recycle_recorder = RawRolloutRecorder(
            output_root=recycle_output_root,
            logger=logger,
        )
        logger.info(
            "raw rollout recycle enabled: output_root=%s",
            str(recycle_output_root),
        )

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(client.get_transport_status("actor_env"))
        except Exception:  # noqa: BLE001
            return {"transport_mode": str(cfg.runtime.trainer_transport.mode)}

    consecutive_update_failures = 0
    consecutive_stats_failures = 0

    def _update_trainer_transport(*, context: str) -> bool:
        nonlocal consecutive_update_failures
        ok = bool(client.update())
        if ok:
            if int(consecutive_update_failures) > 0:
                logger.info(
                    "trainer transport update recovered: context=%s "
                    "consecutive_failures=%s status=%s",
                    str(context),
                    int(consecutive_update_failures),
                    _transport_status(),
                )
            consecutive_update_failures = 0
            return True

        consecutive_update_failures += 1
        logger.warning(
            "trainer transport update skipped after best-effort attempt: "
            "context=%s consecutive_failures=%s status=%s",
            str(context),
            int(consecutive_update_failures),
            _transport_status(),
        )
        # Keep scripts2 aligned with the original SERL actor loop: update() is
        # a best-effort datastore flush, not a training-liveness condition.  A
        # missed ack is retried on the next cadence from the learner's
        # last_update_id, so aborting here is more harmful than continuing.
        return False

    def _send_rollout_stats(*, payload: dict[str, Any]) -> None:
        nonlocal consecutive_stats_failures
        response = client.request("send-stats", payload)
        if response is not None:
            consecutive_stats_failures = 0
            return
        consecutive_stats_failures += 1
        logger.warning(
            "trainer transport send-stats failed: consecutive_failures=%s status=%s",
            int(consecutive_stats_failures),
            _transport_status(),
        )

    wait_for_episode_commit = bool(
        cfg.runtime.trainer_transport.wait_committed_on_episode_end
    )
    progress_bar = tqdm(
        total=int(max_env_steps),
        desc="actor env_steps",
        dynamic_ncols=True,
        leave=True,
    )

    def _commit_assembled_chunks(assembled_chunks: list[AssemblyResult]) -> None:
        nonlocal committed_env_steps
        for assembled_chunk in assembled_chunks:
            for transition in assembled_chunk.transitions:
                data_store.insert(transition)
            for step_offset in range(1, assembled_chunk.env_steps_delta + 1):
                next_committed_env_step = int(committed_env_steps + step_offset)
                if next_committed_env_step % steps_per_update == 0:
                    _update_trainer_transport(
                        context=f"commit_step_{int(next_committed_env_step)}"
                    )
            committed_env_steps += int(assembled_chunk.env_steps_delta)

    try:
        while env_steps < max_env_steps:
            episode_id += 1
            reset_seed = env_seed
            init_episode_idx = episode_id - 1
            obs = env.reset(seed=reset_seed, init_episode_idx=init_episode_idx)
            current_task_prompt = str(task_prompt)
            prefetched = None
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}
            episode_chunk_seq = 0

            while env_steps < max_env_steps:
                if transition_assembler.async_transition_assembly_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(transition_assembler.drain_ready())

                timer.tick("total")
                with timer.context("prepare_action_chunk"):
                    if prefetched is None:
                        robot_state = build_libero_state(obs)
                        image_observations = extract_libero_images(obs)
                        base_policy_input = build_libero_policy_input(
                            prompt=task_prompt,
                            state=robot_state,
                            images=image_observations,
                        )
                        base_actions, _ = policy_client.infer(base_policy_input)
                        base_actions = prepare_base_actions_chunk(
                            base_actions=base_actions,
                            chunk_horizon=chunk_horizon,
                        )
                        residual_obs = build_chunk_residual_obs(
                            robot_state=robot_state,
                            images=image_observations,
                            image_keys=image_keys,
                            base_actions=base_actions,
                            residual_alpha=residual_alpha,
                        )
                    else:
                        base_actions = prefetched.base_actions
                        residual_obs = prefetched.residual_obs
                        prefetched = None

                    residual_actions = agent.sample_action(
                        residual_obs,
                        deterministic=False,
                    )

                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                episode_done = False
                should_log_timer = False
                remaining_env_steps = max(0, int(max_env_steps - env_steps))
                if remaining_env_steps <= 0:
                    timer.tock("total")
                    break

                action_chunk = np.asarray(final_actions, dtype=np.float32)[
                    :remaining_env_steps
                ]

                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)

                raw_chunk = ChunkExecutionRecord.from_env_chunk_result(
                    episode_id=int(episode_id),
                    episode_step_start=int(episode_steps),
                    residual_obs_before_chunk=residual_obs,
                    action_chunk=action_chunk,
                    chunk_result=chunk_result,
                )
                if recycle_recorder is not None:
                    try:
                        recycle_recorder.append_chunk(
                            payload={
                                "episode_id": int(episode_id),
                                "chunk_seq": int(episode_chunk_seq),
                                "episode_step_start": int(episode_steps),
                                "task_prompt": str(task_prompt),
                                "chunk_result": dict(chunk_result),
                            }
                        )
                    except Exception:
                        recycle_recorder.record_append_error()
                        logger.exception(
                            "raw rollout recycle append failed: episode_id=%s chunk_seq=%s",
                            int(episode_id),
                            int(episode_chunk_seq),
                        )
                episode_chunk_seq += 1
                previous_env_steps = int(env_steps)
                with timer.context("assemble_transitions"):
                    assembled_chunks = transition_assembler.handle_chunk(
                        raw=raw_chunk,
                        task_prompt=task_prompt,
                    )
                if transition_assembler.async_transition_assembly_enabled:
                    prefetched = None
                elif assembled_chunks:
                    prefetched = assembled_chunks[-1].prefetched

                env_steps += int(raw_chunk.executed_steps)
                progress_bar.update(int(raw_chunk.executed_steps))
                episode_steps += int(raw_chunk.executed_steps)
                episode_return += float(raw_chunk.reward_sum)
                episode_success = bool(
                    episode_success
                    or any(
                        bool(info.get("env_done", False)) for info in raw_chunk.infos
                    )
                )
                last_info = dict(raw_chunk.chunk_info)
                obs = dict(raw_chunk.final_obs)
                episode_done = bool(raw_chunk.chunk_done or raw_chunk.chunk_truncated)

                for step_offset in range(1, int(raw_chunk.executed_steps) + 1):
                    next_env_step = int(previous_env_steps + step_offset)
                    if next_env_step % log_period == 0:
                        should_log_timer = True

                if transition_assembler.async_transition_assembly_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(assembled_chunks)
                else:
                    _commit_assembled_chunks(assembled_chunks)

                timer.tock("total")

                if should_log_timer:
                    append_jsonl(
                        actor_timer_log_path,
                        {
                            "source": "actor",
                            "env_steps": int(env_steps),
                            "episode_id": int(episode_id),
                            "episode_steps": int(episode_steps),
                            "timer": timer.get_average_times(),
                            "transport": _transport_status(),
                        },
                    )

                if episode_done:
                    break

            if transition_assembler.async_transition_assembly_enabled:
                with timer.context("commit_replay"):
                    _commit_assembled_chunks(
                        transition_assembler.finish_episode(
                            block=bool(wait_for_episode_commit),
                        )
                    )
            _update_trainer_transport(context="episode_end")
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
                    cumulative_success_rate=float(success_count / max(1, episode_id)),
                    recent_success_rate_20=float(recent_success_rate_20),
                ),
                env_info=last_info,
            )

            append_jsonl(
                rollout_log_path,
                {
                    "source": "rollout",
                    **episode_stats,
                    "transport": _transport_status(),
                },
            )
            if wait_for_episode_commit:
                client.wait_until_committed()
            _send_rollout_stats(payload=episode_stats)
            if recycle_recorder is not None:
                try:
                    recycle_recorder.finalize_episode(
                        marker={
                            "episode_id": int(episode_id),
                            "rollout_stats": dict(episode_stats),
                        }
                    )
                except Exception:
                    logger.exception(
                        "raw rollout recycle finalize failed: episode_id=%s",
                        int(episode_id),
                    )
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
        try:
            if current_task_prompt is not None:
                assembled_chunks = transition_assembler.finish_episode(
                    block=True,
                )
                if assembled_chunks:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(assembled_chunks)
            _update_trainer_transport(context="shutdown")
            if bool(cfg.runtime.trainer_transport.wait_committed_on_shutdown):
                client.wait_until_committed()
        except Exception:  # noqa: BLE001
            pass
        pending_recycle_episodes = 0
        recycle_status: dict[str, Any] = {
            "enabled": bool(cfg.recycle.enabled),
            "output_root": str(recycle_output_root),
            "episodes_written": 0,
            "steps_written": 0,
            "append_errors": 0,
            "write_errors": 0,
            "pending_episodes": 0,
        }
        if recycle_recorder is not None:
            try:
                pending_recycle_episodes = recycle_recorder.discard_pending()
                recycle_status = dict(recycle_recorder.status_snapshot())
                recycle_status["pending_episodes"] = int(pending_recycle_episodes)
                logger.info(
                    "raw rollout recycle summary: output_root=%s episodes_written=%s "
                    "steps_written=%s append_errors=%s write_errors=%s pending_episodes=%s",
                    str(recycle_status["output_root"]),
                    int(recycle_status["episodes_written"]),
                    int(recycle_status["steps_written"]),
                    int(recycle_status["append_errors"]),
                    int(recycle_status["write_errors"]),
                    int(recycle_status["pending_episodes"]),
                )
            except Exception:  # noqa: BLE001
                logger.exception("raw rollout recycle cleanup failed")
                recycle_status["write_errors"] = (
                    int(recycle_status.get("write_errors", 0)) + 1
                )
        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
                "transport": _transport_status(),
                "recycle_enabled": bool(recycle_status.get("enabled", False)),
                "recycle_output_root": recycle_status.get("output_root", None),
                "recycle_episodes_written": int(
                    recycle_status.get("episodes_written", 0)
                ),
                "recycle_steps_written": int(recycle_status.get("steps_written", 0)),
                "recycle_append_errors": int(recycle_status.get("append_errors", 0)),
                "recycle_write_errors": int(recycle_status.get("write_errors", 0)),
                "recycle": recycle_status,
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            client.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass
        if str(cfg.env.backend) != "remote":
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
        transition_assembler.close()


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
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        discount=float(cfg.sac.discount),
        image_keys=image_keys,
        capacity=int(cfg.replay.capacity),
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
            "entity": cfg.wandb.entity,
            "exp_descriptor": run_name,
            "tag": [run_name],
            "group": cfg.wandb.group,
            "mode": cfg.wandb.mode,
        }
    )
    wandb_variant = cfg_to_log_payload(cfg)
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandBLogger(
        wandb_config=wandb_cfg,
        variant=wandb_variant,
        wandb_output_dir=str(wandb_dir),
        mode=cfg.wandb.mode,
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
        "transport_mode": str(cfg.runtime.trainer_transport.mode),
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
        "timer_log_path": str(learner_timer_log_path),
    }

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(server.get_transport_status("actor_env"))
        except Exception:  # noqa: BLE001
            return {"transport_mode": str(cfg.runtime.trainer_transport.mode)}

    def _committed_online_steps() -> int:
        return int(replay_buffer.latest_data_id())

    def _should_stop_after_actor_done(*, online_update_steps: int) -> bool:
        if int(env_steps) < int(cfg.training.max_env_steps):
            return False
        committed_online_steps = int(_committed_online_steps())
        if int(online_update_steps) < int(committed_online_steps):
            return False
        transport_status = _transport_status()
        accepted = int(transport_status.get("accepted_update_id", -1))
        committed = int(transport_status.get("committed_update_id", -1))
        target_last_online_id = max(0, int(env_steps) - 1)
        if accepted < int(target_last_online_id) or committed < int(
            target_last_online_id
        ):
            return False
        if accepted >= 0 and committed >= 0 and accepted > committed:
            return False
        return True

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
            episode_id = int(rollout_stats["rollout"]["episode_id"])
            if episode_id > 0:
                latest_completed_episode_id = max(
                    int(latest_completed_episode_id),
                    int(episode_id),
                )
                completed_episode_env_steps[int(episode_id)] = int(env_steps)
        rollout_metrics = extract_rollout_wandb_metrics(rollout_stats)
        if rollout_metrics:
            wandb_logger.log(to_jsonable(rollout_metrics))
        return {}

    server = build_learner_trainer_transport(
        trainer_port=cfg.runtime.trainer_port,
        broadcast_port=cfg.runtime.broadcast_port,
        transport_cfg=cfg.runtime.trainer_transport,
        request_callback=stats_callback,
        request_types=("send-stats",),
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
            action_dim=int(cfg.env.action_dim),
            chunk_horizon=int(cfg.residual.chunk_horizon),
            discount=float(cfg.sac.discount),
            image_keys=image_keys,
            capacity=int(cfg.offline.capacity),
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
                float(cfg.offline.ratio),
                int(cfg.offline.pretrain_steps),
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
    batch_sampler = PrefetchingMixedBatchSampler(
        online_replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        batch_size=int(cfg.replay.batch_size),
        device=agent.device,
        pack_obs_and_next_obs=True,
        prefer_device_concat=True,
        thread_name_prefix="libero-learner-prefetch",
    )
    update_profile_accumulator = ProfileAccumulator()
    logger.info("learner replay batch prefetch enabled: depth=1")

    def _next_training_batch(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        with timer.context("sample_replay_buffer"):
            batch, batch_mix = batch_sampler.next_batch(
                offline_ratio=float(offline_ratio)
            )
        return batch, batch_mix

    def _run_training_update(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal agent
        train_batch_mix = {
            "online_batch_size": int(cfg.replay.batch_size),
            "offline_batch_size": 0,
        }
        update_profile: dict[str, float] = {}
        for _ in range(max(0, critic_actor_ratio - 1)):
            batch, _ = _next_training_batch(offline_ratio=float(offline_ratio))
            with timer.context("train_critics"):
                agent, _critics_info = agent.update_critics(
                    batch,
                    profile=update_profile,
                )

        batch, train_batch_mix = _next_training_batch(
            offline_ratio=float(offline_ratio)
        )
        with timer.context("train"):
            agent, update_info = agent.update_high_utd(
                batch,
                utd_ratio=cfg.sac.utd_ratio,
                profile=update_profile,
            )
        update_profile_accumulator.record(update_profile)
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

    def _publish_actor_network(*, step: int, reason: str) -> None:
        with timer.context("publish_snapshot"):
            payload = snapshot_actor_network_payload(agent, step=int(step))
        with timer.context("publish_network"):
            server.publish_network(payload)
        logger.info(
            "publish network: step=%s env_steps=%s reason=%s",
            int(step),
            int(env_steps),
            str(reason),
        )

    if (
        offline_replay_buffer is not None
        and int(cfg.offline.pretrain_steps) > 0
        and int(update_steps) < int(max_update_steps)
    ):
        logger.info(
            "starting offline pretrain: steps=%s offline_replay_size=%s",
            int(cfg.offline.pretrain_steps),
            int(len(offline_replay_buffer)),
        )
        pretrain_bar = tqdm(
            total=min(int(cfg.offline.pretrain_steps), int(max_update_steps)),
            desc="learner offline pretrain",
            dynamic_ncols=True,
            leave=True,
        )
        last_pretrain_info: dict[str, Any] | None = None
        try:
            while int(update_steps) < int(max_update_steps) and int(
                offline_pretrain_steps_done
            ) < int(cfg.offline.pretrain_steps):
                check_async_eval_worker(async_eval, logger=logger)
                last_pretrain_info, _ = _run_training_update(offline_ratio=1.0)
                update_steps += 1
                offline_pretrain_steps_done += 1
                check_async_eval_worker(async_eval, logger=logger)
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
        _publish_actor_network(
            step=int(update_steps),
            reason="offline_pretrain_complete",
        )
        initial_network_published = True

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
                check_async_eval_worker(async_eval, logger=logger)
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
        _publish_actor_network(step=int(update_steps), reason="initial")

    interrupted = False
    try:
        while update_steps < max_update_steps:
            check_async_eval_worker(async_eval, logger=logger)
            _maybe_queue_async_eval()
            online_update_steps = max(
                0, int(update_steps - offline_pretrain_steps_done)
            )
            if _should_stop_after_actor_done(
                online_update_steps=int(online_update_steps)
            ):
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
                offline_ratio=float(cfg.offline.ratio),
            )
            update_steps += 1
            check_async_eval_worker(async_eval, logger=logger)
            _maybe_queue_async_eval()

            if update_steps % steps_per_update == 0:
                _publish_actor_network(step=int(update_steps), reason="periodic")

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
                updates_since_last_log = max(
                    1, int(update_steps - last_log_update_steps)
                )
                updates_per_sec = float(updates_since_last_log) / float(elapsed_sec)
                last_log_time = now
                last_log_update_steps = int(update_steps)
                timer_metrics = to_jsonable(timer.get_average_times())
                sample_profile_metrics = to_jsonable(
                    batch_sampler.drain_sample_profile()
                )
                update_profile_metrics = to_jsonable(update_profile_accumulator.drain())
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
                        "sample_profile": sample_profile_metrics,
                        "update_profile": update_profile_metrics,
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
                    offline_ratio_actual = float(
                        batch_mix["offline_batch_size"]
                    ) / float(offline_total)
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
        batch_sampler.close()
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
                    "enabled": bool(cfg.offline.enabled),
                    "ratio": float(cfg.offline.ratio),
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
                    "pretrain_steps_requested": int(cfg.offline.pretrain_steps),
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
