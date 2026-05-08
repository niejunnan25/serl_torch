from __future__ import annotations

"""Copy-style AgiBot residual DRQ training script."""

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
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from serl_launcher.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_actor_network_payload
from serl_launcher.common.trainer_transport import build_actor_trainer_transport
from serl_launcher.common.trainer_transport import build_learner_trainer_transport
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
from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import sample_mixed_training_batch
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.utils.seeding import set_global_seeds
from serl_launcher.utils.serialization import to_jsonable
from serl_launcher.utils.timer_utils import Timer

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.config import AgiBotTrainConfig
from serl_torch.examples.agibot_real.config import cfg_to_log_payload
from serl_torch.examples.agibot_real.config import parse_train_cfg
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.env.factory import create_env
from serl_torch.examples.agibot_real.offline_data import (
    load_prepared_offline_replay,
)
from serl_torch.examples.agibot_real.offline_data import (
    resolve_and_validate_prepared_paths,
)
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_observation_space,
)
from serl_torch.examples.agibot_real.residual_observation import (
    build_chunk_residual_sample_obs,
)
from serl_torch.examples.agibot_real.torch_compile import maybe_enable_torch_compile
from serl_torch.examples.agibot_real.transition_assembly import (
    AgiBotTransitionAssembler,
)
from serl_torch.examples.agibot_real.transition_assembly import AssemblyResult
from serl_torch.examples.agibot_real.transition_assembly import RawChunkRecord
from serl_torch.examples.agibot_real.transition_assembly import (
    count_executed_steps_from_infos,
)
from serl_torch.examples.agibot_real.video_recorder import AsyncImageVideoRecorder
from serl_torch.examples.agibot_real.video_recorder import AsyncVideoRecorderConfig


def actor(
    cfg: AgiBotTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    env = create_env(cfg, logger)
    base_policy = build_agibot_base_policy(cfg, logger=logger)
    logger.info("Chunk policy backend: %s", base_policy.describe())
    video_recorder: AsyncImageVideoRecorder | None = None
    if bool(cfg.video.enabled):
        video_recorder = AsyncImageVideoRecorder(
            config=AsyncVideoRecorderConfig(
                camera_key=str(cfg.video.camera_key),
                fps=float(cfg.video.fps),
                output_dir=run_dir / str(cfg.video.output_dir),
                max_pending_frames=int(cfg.video.max_pending_frames),
                drop_frames_when_busy=bool(cfg.video.drop_frames_when_busy),
            ),
            logger=logger,
        )
        logger.info(
            "Rollout video recording enabled: camera_key=%s fps=%.3f output_dir=%s max_pending_frames=%s drop_frames_when_busy=%s",
            str(cfg.video.camera_key),
            float(cfg.video.fps),
            run_dir / str(cfg.video.output_dir),
            int(cfg.video.max_pending_frames),
            bool(cfg.video.drop_frames_when_busy),
        )

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
    transition_assembler = AgiBotTransitionAssembler(
        cfg=cfg,
        base_policy=base_policy,
        logger=logger,
    )

    if transition_assembler.async_backfill_enabled:
        logger.info(
            "Async backfill enabled: mode=%s endpoint=%s:%s max_pending_chunks=%s",
            str(cfg.backfill_policy.mode),
            str(cfg.backfill_policy.host),
            int(cfg.backfill_policy.port),
            int(cfg.backfill_policy.max_pending_chunks),
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

    if bool(cfg.training.torch_compile.enabled):
        logger.info(
            "training.torch_compile.enabled=true is ignored on actor; compile applies "
            "to learner updates only"
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

    prepare_episode_reset_fn = getattr(env, "prepare_episode_reset", None)
    start_episode_after_reset_fn = getattr(env, "start_episode_after_reset", None)
    supports_staged_reset = bool(
        callable(prepare_episode_reset_fn) and callable(start_episode_after_reset_fn)
    )

    timer = Timer()
    steps_per_update = cfg.training.steps_per_update
    log_period = cfg.training.log_period
    max_env_steps = cfg.training.max_env_steps
    max_episodes = cfg.training.max_episodes

    env_steps = 0
    committed_env_steps = 0
    episode_id = 0
    success_count = 0
    recent_episode_successes: deque[int] = deque(maxlen=20)
    actor_timer_log_path = run_dir / "actor_timers.jsonl"
    rollout_log_path = run_dir / str(cfg.logging.episode_log_file or "episode_logs.jsonl")

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
            consecutive_update_failures = 0
            return True
        consecutive_update_failures += 1
        logger.warning(
            "trainer transport update failed: context=%s consecutive_failures=%s status=%s",
            str(context),
            int(consecutive_update_failures),
            _transport_status(),
        )
        if int(consecutive_update_failures) >= 5:
            raise RuntimeError(
                "trainer transport update failed repeatedly; aborting actor run"
            )
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
    current_task_prompt: str | None = None
    pending_last_transition: dict[str, Any] | None = None

    def _flush_pending_last_transition() -> None:
        nonlocal pending_last_transition
        if pending_last_transition is None:
            return
        data_store.insert(pending_last_transition)
        pending_last_transition = None

    def _finalize_pending_last_transition(
        *,
        terminal_reward: float,
        boundary_flag: bool,
    ) -> None:
        nonlocal pending_last_transition
        if pending_last_transition is None:
            return
        pending_last_transition["rewards"] = float(
            pending_last_transition["rewards"]
        ) + float(terminal_reward)
        pending_last_transition["dones"] = bool(boundary_flag)
        pending_last_transition["masks"] = 0.0
        data_store.insert(pending_last_transition)
        pending_last_transition = None

    def _commit_assembled_chunks(assembled_chunks: list[AssemblyResult]) -> None:
        nonlocal committed_env_steps
        nonlocal pending_last_transition
        for assembled_chunk in assembled_chunks:
            _flush_pending_last_transition()
            if bool(assembled_chunk.episode_done):
                transitions_to_insert = assembled_chunk.transitions
            else:
                transitions_to_insert = assembled_chunk.transitions[:-1]
                pending_last_transition = assembled_chunk.transitions[-1]
            for transition in transitions_to_insert:
                data_store.insert(transition)
            for step_offset in range(1, assembled_chunk.env_steps_delta + 1):
                next_committed_env_step = int(committed_env_steps + step_offset)
                if next_committed_env_step % steps_per_update == 0:
                    _update_trainer_transport(
                        context=f"commit_step_{int(next_committed_env_step)}"
                    )
            committed_env_steps += int(assembled_chunk.env_steps_delta)

    progress_bar = tqdm(
        total=int(max_env_steps),
        desc="actor env_steps",
        dynamic_ncols=True,
        leave=True,
    )
    prefetched_reset_prepared = False

    try:
        while env_steps < max_env_steps and episode_id < max_episodes:
            episode_id += 1
            if prefetched_reset_prepared:
                with timer.context("reset_obs"):
                    assert callable(start_episode_after_reset_fn)
                    obs = dict(start_episode_after_reset_fn())
                task_prompt = str(env.task_description)
                prefetched_reset_prepared = False
            else:
                with timer.context("reset_env"):
                    obs = dict(env.reset())
                task_prompt = str(env.task_description)
            current_task_prompt = str(task_prompt)
            if video_recorder is not None:
                video_recorder.start_episode(int(episode_id))
                video_recorder.add_obs_frame(obs)

            pending_last_transition = None
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}

            while env_steps < max_env_steps:
                if transition_assembler.async_backfill_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(transition_assembler.drain_ready())
                timer.tick("total")
                with timer.context("sample_actions"):
                    decision_obs = transition_assembler.next_decision_obs(
                        obs=obs,
                        task_prompt=task_prompt,
                    )
                    residual_actions = agent.sample_action(
                        decision_obs.residual_obs,
                        deterministic=False,
                    )
                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=decision_obs.base_actions,
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
                if video_recorder is not None:
                    for post_step_obs in chunk_result.get("observations", ()):
                        if isinstance(post_step_obs, dict):
                            video_recorder.add_obs_frame(post_step_obs)

                chunk_infos = [dict(v) for v in chunk_result["infos"]]
                executed_steps = count_executed_steps_from_infos(chunk_infos)
                last_info = dict(chunk_result["info"])
                obs = dict(chunk_result["obs"])

                if executed_steps <= 0:
                    done_flag = bool(chunk_result["done"] or chunk_result["truncated"])
                    if not done_flag:
                        raise RuntimeError(
                            "step_chunk returned no executed actions without a terminal outcome"
                        )
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(
                            transition_assembler.finish_episode(
                                task_prompt=task_prompt,
                                block=bool(wait_for_episode_commit),
                            )
                        )
                    _finalize_pending_last_transition(
                        terminal_reward=float(chunk_result["reward_sum"]),
                        boundary_flag=bool(
                            chunk_result["done"] or chunk_result["truncated"]
                        ),
                    )
                    episode_return += float(chunk_result["reward_sum"])
                    episode_success = bool(
                        episode_success or chunk_result["info"].get("success", False)
                    )
                    episode_done = True
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
                    break

                raw_chunk = RawChunkRecord.from_step_chunk_result(
                    episode_id=int(episode_id),
                    episode_step_start=int(episode_steps),
                    residual_obs_before_chunk=decision_obs.residual_obs,
                    action_chunk=action_chunk,
                    chunk_result=chunk_result,
                )
                previous_env_steps = int(env_steps)
                with timer.context("build_decision_obs"):
                    assembled_chunks = transition_assembler.handle_chunk(
                        raw=raw_chunk,
                        task_prompt=task_prompt,
                        env_steps=int(env_steps),
                        max_env_steps=int(max_env_steps),
                    )

                env_steps += int(raw_chunk.executed_steps)
                progress_bar.update(int(raw_chunk.executed_steps))
                episode_steps += int(raw_chunk.executed_steps)
                episode_return += float(raw_chunk.reward_sum)
                episode_success = bool(
                    episode_success
                    or any(bool(info.get("success", False)) for info in raw_chunk.infos)
                )
                last_info = dict(raw_chunk.chunk_info)
                obs = dict(raw_chunk.final_obs)

                for step_offset in range(1, int(raw_chunk.executed_steps) + 1):
                    next_env_step = int(previous_env_steps + step_offset)
                    if next_env_step % log_period == 0:
                        should_log_timer = True

                if transition_assembler.async_backfill_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(assembled_chunks)
                else:
                    _commit_assembled_chunks(assembled_chunks)

                if (
                    bool(chunk_result["done"] or chunk_result["truncated"])
                    or env_steps >= max_env_steps
                ):
                    episode_done = True

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

            next_reset_error: Exception | None = None
            should_prefetch_next_reset = bool(
                transition_assembler.async_backfill_enabled
                and supports_staged_reset
                and env_steps < max_env_steps
                and episode_id < max_episodes
            )
            if should_prefetch_next_reset:
                try:
                    with timer.context("reset_env"):
                        assert callable(prepare_episode_reset_fn)
                        prepare_episode_reset_fn()
                except Exception as exc:  # noqa: BLE001
                    next_reset_error = exc
            if transition_assembler.async_backfill_enabled:
                with timer.context("commit_replay"):
                    _commit_assembled_chunks(
                        transition_assembler.finish_episode(
                            task_prompt=task_prompt,
                            block=bool(wait_for_episode_commit),
                        )
                    )
            else:
                _commit_assembled_chunks(
                    transition_assembler.finish_episode(
                        task_prompt=task_prompt,
                    )
                )
            _flush_pending_last_transition()
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
                    "transport": _transport_status(),
                },
            )
            if wait_for_episode_commit:
                client.wait_until_committed()
            _send_rollout_stats(payload=episode_stats)
            progress_bar.set_postfix(
                episode=int(episode_id),
                success=int(bool(episode_success)),
                refresh=False,
            )
            if next_reset_error is not None:
                raise next_reset_error
            prefetched_reset_prepared = bool(should_prefetch_next_reset)
            logger.info(
                "episode=%s success=%s steps=%s return=%.3f env_steps=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(env_steps),
            )
            if video_recorder is not None:
                video_recorder.end_episode(
                    episode_id=int(episode_id),
                    success=bool(episode_success),
                    episode_steps=int(episode_steps),
                )

    finally:
        if current_task_prompt is not None:
            try:
                _commit_assembled_chunks(
                    transition_assembler.finish_episode(
                        task_prompt=current_task_prompt,
                        block=True,
                    )
                )
                _flush_pending_last_transition()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "ignored final actor replay flush error",
                    exc_info=True,
                )
        try:
            _update_trainer_transport(context="shutdown")
            if bool(cfg.runtime.trainer_transport.wait_committed_on_shutdown):
                client.wait_until_committed()
        except Exception:  # noqa: BLE001
            pass
        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
                "transport": _transport_status(),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
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
        try:
            transition_assembler.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if video_recorder is not None:
                video_recorder.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            client.stop()
        except Exception:  # noqa: BLE001
            pass


def learner(cfg: AgiBotTrainConfig, *, run_dir: Path, logger: logging.Logger) -> None:
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
    agent = maybe_enable_torch_compile(
        agent,
        compile_cfg=cfg.training.torch_compile,
        logger=logger,
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
        "files_loaded": 0,
        "episodes_loaded": 0,
        "inserted": 0,
        "errors": 0,
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
    configure_learner_wandb_metrics(wandb_logger=wandb_logger)

    update_steps = 0
    env_steps = 0
    latest_completed_episode_id = 0
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
        if accepted < int(target_last_online_id) or committed < int(target_last_online_id):
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
            latest_completed_episode_id = max(
                int(latest_completed_episode_id),
                int(rollout_stats["rollout"]["episode_id"]),
            )
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

    def _run_training_update(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal agent
        train_batch_mix = {
            "online_batch_size": int(cfg.replay.batch_size),
            "offline_batch_size": 0,
        }
        for _ in range(max(0, critic_actor_ratio - 1)):
            with timer.context("sample_replay_buffer"):
                batch, _ = sample_mixed_training_batch(
                    online_replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    batch_size=int(cfg.replay.batch_size),
                    offline_ratio=float(offline_ratio),
                )
            with timer.context("train_critics"):
                agent, _ = agent.update_critics(batch)

        with timer.context("train"):
            batch, train_batch_mix = sample_mixed_training_batch(
                online_replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                batch_size=int(cfg.replay.batch_size),
                offline_ratio=float(offline_ratio),
            )
            agent, update_info = agent.update_high_utd(
                batch,
                utd_ratio=cfg.sac.utd_ratio,
            )
        return update_info, train_batch_mix

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
            while (
                int(update_steps) < int(max_update_steps)
                and int(offline_pretrain_steps_done) < int(cfg.offline.pretrain_steps)
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
                offline_ratio=float(cfg.offline.ratio),
            )
            update_steps += 1

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
        with progress_state_lock:
            summary_last_completed_episode_id = int(latest_completed_episode_id)
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
                "last_completed_episode_id": int(summary_last_completed_episode_id),
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
    config_name="train_residual_copy",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_residual")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    set_global_seeds(typed_cfg.global_seed)

    if typed_cfg.runtime.role == "actor":
        actor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    learner(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
