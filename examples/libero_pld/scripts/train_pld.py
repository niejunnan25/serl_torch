from __future__ import annotations

"""Clean PLD Stage-1 residual RL training for LIBERO."""

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
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.agents.continuous.drq_typed_config import (  # noqa: E402
    create_drq_agent_from_typed_cfg,
)
from serl_launcher.common.agent_acceleration import apply_torch_compile  # noqa: E402
from serl_launcher.common.checkpoint_codec import (  # noqa: E402
    apply_checkpoint_payload_to_agent,
)
from serl_launcher.common.checkpoint_codec import (  # noqa: E402
    snapshot_actor_network_payload,
)
from serl_launcher.common.trainer_transport import (  # noqa: E402
    build_actor_trainer_transport,
)
from serl_launcher.common.trainer_transport import (  # noqa: E402
    build_learner_trainer_transport,
)
from serl_launcher.residual.chunk_window_replay import (  # noqa: E402
    PrefetchingMixedBatchSampler,
)
from serl_launcher.residual.chunk_window_replay import (  # noqa: E402
    PreparedStepWindowReplayBufferSampler,
)
from serl_launcher.residual.chunk_window_replay import ProfileAccumulator  # noqa: E402
from serl_launcher.residual.chunk_window_replay import (  # noqa: E402
    create_chunk_replay_buffer,
)
from serl_launcher.residual.observation import (  # noqa: E402
    build_chunk_residual_sample_obs,
)
from serl_launcher.residual.observation import (  # noqa: E402
    build_chunk_residual_observation_space,
)
from serl_launcher.residual.observation import prepare_base_actions_chunk  # noqa: E402
from serl_launcher.residual.observation import build_chunk_residual_obs  # noqa: E402
from serl_launcher.residual.typed_action import ResidualActionSpec  # noqa: E402
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint  # noqa: E402
from serl_launcher.utils.jsonl import append_jsonl  # noqa: E402
from serl_launcher.utils.seeding import set_global_seeds  # noqa: E402
from serl_launcher.utils.serialization import to_jsonable  # noqa: E402
from serl_launcher.utils.timer_utils import Timer  # noqa: E402

from serl_torch.examples.libero.config import LiberoTrainConfig  # noqa: E402
from serl_torch.examples.libero.config import cfg_to_log_payload  # noqa: E402
from serl_torch.examples.libero.config import parse_train_cfg  # noqa: E402
from serl_torch.examples.libero.env.factory import create_env  # noqa: E402
from serl_torch.examples.libero.env.observation import LIBERO_STATE_DIM  # noqa: E402
from serl_torch.examples.libero.env.observation import (  # noqa: E402
    RESIDUAL_IMAGE_HEIGHT,
)
from serl_torch.examples.libero.env.observation import (  # noqa: E402
    RESIDUAL_IMAGE_WIDTH,
)
from serl_torch.examples.libero.env.observation import build_libero_state  # noqa: E402
from serl_torch.examples.libero.env.observation import extract_libero_images  # noqa: E402
from serl_torch.examples.libero.env.policy_input import (  # noqa: E402
    build_libero_policy_input,
)
from serl_torch.examples.libero.env.offline_data import (  # noqa: E402
    load_prepared_offline_replay,
)
from serl_torch.examples.libero.env.offline_data import (  # noqa: E402
    resolve_and_validate_prepared_paths,
)
from serl_torch.examples.libero.runtime.transition_assembly import (  # noqa: E402
    AssemblyResult,
)
from serl_torch.examples.libero.runtime.transition_assembly import (  # noqa: E402
    ChunkExecutionRecord,
)
from serl_torch.examples.libero.runtime.transition_assembly import (  # noqa: E402
    LiberoActorTransitionAssembler,
)
from serl_launcher.policy.typed_factory import build_policy_client  # noqa: E402


def _pld_section(raw_cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(raw_cfg.get("pld", {}) or {}, resolve=True)
    return dict(value or {})


def _nested(payload: dict[str, Any], key: str, default: Any) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _make_agent(cfg: LiberoTrainConfig):
    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
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
    return agent, residual_action_spec, sample_obs


def _commit_chunks(
    *,
    assembled_chunks: list[AssemblyResult],
    data_store: QueuedDataStore,
    replay_inserted_steps: int,
    steps_per_update: int,
    update_transport,
) -> int:
    for assembled_chunk in assembled_chunks:
        for transition in assembled_chunk.transitions:
            data_store.insert(transition)
        inserted_delta = int(len(assembled_chunk.transitions))
        for step_offset in range(1, inserted_delta + 1):
            next_step = int(replay_inserted_steps + step_offset)
            if next_step % int(steps_per_update) == 0:
                update_transport(context=f"commit_replay_step_{next_step}")
        replay_inserted_steps += inserted_delta
    return int(replay_inserted_steps)


def actor(
    cfg: LiberoTrainConfig,
    raw_cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    pld_cfg = _pld_section(raw_cfg)
    warmup_episodes = int(_nested(pld_cfg, "base_warmup_episodes", 100))
    env = create_env(cfg, logger)
    task_prompt = str(env.task_description)
    policy_client = build_policy_client(cfg, logger=logger)
    agent, residual_action_spec, _sample_obs = _make_agent(cfg)
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
        apply_checkpoint_payload_to_agent(agent, dict(payload), load_optimizers=False)

    client.recv_network_callback(update_actor)
    transition_assembler = LiberoActorTransitionAssembler(
        cfg=cfg,
        policy_client=policy_client,
        logger=logger,
    )
    timer = Timer()
    max_env_steps = int(cfg.training.max_env_steps)
    steps_per_update = int(cfg.training.steps_per_update)
    log_period = int(cfg.training.log_period)
    env_steps = 0
    replay_inserted_steps = 0
    episode_id = 0
    success_count = 0
    recent_episode_successes: deque[int] = deque(maxlen=20)
    rollout_log_path = run_dir / str(cfg.logging.episode_log_file or "episode_logs.jsonl")
    actor_timer_log_path = run_dir / "actor_timers.jsonl"

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(client.get_transport_status("actor_env"))
        except Exception:
            return {"transport_mode": str(cfg.runtime.trainer_transport.mode)}

    def _update_transport(*, context: str) -> bool:
        ok = bool(client.update())
        if not ok:
            logger.warning("trainer transport update skipped: context=%s status=%s", context, _transport_status())
        return ok

    def _send_stats(payload: dict[str, Any]) -> None:
        response = client.request("send-stats", payload)
        if response is None:
            logger.warning("trainer transport send-stats failed: status=%s", _transport_status())

    progress_bar = tqdm(total=max_env_steps, desc="pld actor env_steps", dynamic_ncols=True, leave=True)
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "pld_stage1",
        "base_warmup_episodes": int(warmup_episodes),
        "env_steps": 0,
        "replay_inserted_steps": 0,
        "episodes": 0,
        "successes": 0,
    }
    current_task_prompt: str | None = None
    try:
        while env_steps < max_env_steps:
            episode_id += 1
            base_warmup = int(episode_id) <= int(warmup_episodes)
            obs = env.reset(seed=int(cfg.env.seed), init_episode_idx=int(episode_id - 1))
            current_task_prompt = str(task_prompt)
            episode_steps = 0
            episode_return = 0.0
            episode_success = False
            last_info: dict[str, Any] = {}
            prefetched = None
            while env_steps < max_env_steps:
                if transition_assembler.async_transition_assembly_enabled:
                    replay_inserted_steps = _commit_chunks(
                        assembled_chunks=transition_assembler.drain_ready(),
                        data_store=data_store,
                        replay_inserted_steps=replay_inserted_steps,
                        steps_per_update=steps_per_update,
                        update_transport=_update_transport,
                    )

                timer.tick("total")
                with timer.context("prepare_action"):
                    if prefetched is None:
                        robot_state = build_libero_state(obs)
                        image_observations = extract_libero_images(obs)
                        policy_input = build_libero_policy_input(
                            prompt=task_prompt,
                            state=robot_state,
                            images=image_observations,
                        )
                        base_actions, _ = policy_client.infer(policy_input)
                        base_actions = prepare_base_actions_chunk(
                            base_actions=base_actions,
                            chunk_horizon=int(cfg.residual.chunk_horizon),
                        )
                        residual_obs = build_chunk_residual_obs(
                            robot_state=robot_state,
                            images=image_observations,
                            image_keys=cfg.obs.image_keys,
                            base_actions=base_actions,
                            residual_alpha=float(cfg.residual.alpha),
                        )
                    else:
                        base_actions = prefetched.base_actions
                        residual_obs = prefetched.residual_obs
                        prefetched = None

                    if base_warmup:
                        residual_actions = np.zeros(
                            (int(residual_action_spec.chunk_policy_action_dim),),
                            dtype=np.float32,
                        )
                        final_actions = np.asarray(base_actions, dtype=np.float32)
                    else:
                        residual_actions = agent.sample_action(
                            residual_obs,
                            deterministic=False,
                        )
                        final_actions = residual_action_spec.compose_chunk(
                            base_action_chunk=base_actions,
                            residual_action=residual_actions,
                        )

                remaining_env_steps = max(0, max_env_steps - env_steps)
                if remaining_env_steps <= 0:
                    timer.tock("total")
                    break
                action_chunk = np.asarray(final_actions, dtype=np.float32)[:remaining_env_steps]
                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)
                raw_chunk = ChunkExecutionRecord.from_env_chunk_result(
                    episode_id=int(episode_id),
                    episode_step_start=int(episode_steps),
                    residual_obs_before_chunk=residual_obs,
                    action_chunk=action_chunk,
                    chunk_result=chunk_result,
                )
                with timer.context("assemble_transitions"):
                    assembled_chunks = transition_assembler.handle_chunk(
                        raw=raw_chunk,
                        task_prompt=task_prompt,
                    )
                if transition_assembler.async_transition_assembly_enabled:
                    prefetched = None
                elif assembled_chunks:
                    prefetched = assembled_chunks[-1].prefetched

                previous_env_steps = int(env_steps)
                env_steps += int(raw_chunk.executed_steps)
                episode_steps += int(raw_chunk.executed_steps)
                episode_return += float(raw_chunk.reward_sum)
                episode_success = bool(
                    episode_success
                    or any(bool(info.get("env_done", False) or info.get("success", False)) for info in raw_chunk.infos)
                )
                last_info = dict(raw_chunk.chunk_info)
                obs = dict(raw_chunk.final_obs)
                episode_done = bool(raw_chunk.chunk_done or raw_chunk.chunk_truncated)
                progress_bar.update(int(raw_chunk.executed_steps))
                replay_inserted_steps = _commit_chunks(
                    assembled_chunks=assembled_chunks,
                    data_store=data_store,
                    replay_inserted_steps=replay_inserted_steps,
                    steps_per_update=steps_per_update,
                    update_transport=_update_transport,
                )
                timer.tock("total")

                if any((previous_env_steps + i) % log_period == 0 for i in range(1, int(raw_chunk.executed_steps) + 1)):
                    append_jsonl(
                        actor_timer_log_path,
                        {
                            "source": "actor",
                            "env_steps": int(env_steps),
                            "replay_inserted_steps": int(replay_inserted_steps),
                            "episode_id": int(episode_id),
                            "base_warmup": bool(base_warmup),
                            "timer": timer.get_average_times(),
                            "transport": _transport_status(),
                        },
                    )
                if episode_done:
                    break

            if transition_assembler.async_transition_assembly_enabled:
                replay_inserted_steps = _commit_chunks(
                    assembled_chunks=transition_assembler.finish_episode(block=True),
                    data_store=data_store,
                    replay_inserted_steps=replay_inserted_steps,
                    steps_per_update=steps_per_update,
                    update_transport=_update_transport,
                )
            _update_transport(context="episode_end")
            success_count += int(episode_success)
            recent_episode_successes.append(int(episode_success))
            recent_success_rate_20 = float(sum(recent_episode_successes)) / float(max(1, len(recent_episode_successes)))
            episode_payload = {
                "source": "rollout",
                "env_steps": int(env_steps),
                "replay_inserted_steps": int(replay_inserted_steps),
                "episode_id": int(episode_id),
                "episode_steps": int(episode_steps),
                "episode_return": float(episode_return),
                "success": bool(episode_success),
                "cumulative_success_rate": float(success_count / max(1, episode_id)),
                "recent_success_rate_20": float(recent_success_rate_20),
                "base_warmup": bool(base_warmup),
                "env_info": last_info,
                "transport": _transport_status(),
            }
            append_jsonl(rollout_log_path, episode_payload)
            _send_stats(episode_payload)
            logger.info(
                "episode=%s warmup=%s success=%s steps=%s env_steps=%s replay=%s return=%.3f recent20=%.3f",
                int(episode_id),
                bool(base_warmup),
                bool(episode_success),
                int(episode_steps),
                int(env_steps),
                int(replay_inserted_steps),
                float(episode_return),
                float(recent_success_rate_20),
            )
    finally:
        try:
            if current_task_prompt is not None:
                replay_inserted_steps = _commit_chunks(
                    assembled_chunks=transition_assembler.finish_episode(block=True),
                    data_store=data_store,
                    replay_inserted_steps=replay_inserted_steps,
                    steps_per_update=steps_per_update,
                    update_transport=_update_transport,
                )
            _update_transport(context="shutdown")
            if bool(cfg.runtime.trainer_transport.wait_committed_on_shutdown):
                client.wait_until_committed()
        except Exception:
            logger.exception("actor shutdown flush failed")
        summary.update(
            {
                "env_steps": int(env_steps),
                "replay_inserted_steps": int(replay_inserted_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
                "transport": _transport_status(),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            client.stop()
        except Exception:
            pass
        try:
            progress_bar.close()
        except Exception:
            pass
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:
                pass
        transition_assembler.close()


def learner(
    cfg: LiberoTrainConfig,
    raw_cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    pld_cfg = _pld_section(raw_cfg)
    require_offline = bool(_nested(pld_cfg, "require_offline", True))
    calql_enabled = bool(_nested(pld_cfg, "calql_pretrain.enabled", True))
    calql_steps = int(_nested(pld_cfg, "calql_pretrain.steps", 150))
    calql_alpha = float(_nested(pld_cfg, "calql_pretrain.alpha", 1.0))
    calql_n_actions = int(_nested(pld_cfg, "calql_pretrain.n_actions", cfg.sac.cql_n_actions))
    calql_temperature = float(_nested(pld_cfg, "calql_pretrain.temperature", cfg.sac.cql_temperature))
    agent, _residual_action_spec, sample_obs = _make_agent(cfg)
    agent = apply_torch_compile(agent, compile_cfg=cfg.training.torch_compile)
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=cfg.obs.image_keys,
    )
    replay_buffer = create_chunk_replay_buffer(
        observation_space=observation_space,
        action_dim=int(cfg.env.action_dim),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        discount=float(cfg.sac.discount),
        image_keys=cfg.obs.image_keys,
        capacity=int(cfg.replay.capacity),
        sample_stride=int(cfg.replay.sample_stride),
    )
    offline_replay_buffer: Any | None = None
    if bool(cfg.offline.enabled):
        offline_resolution = resolve_and_validate_prepared_paths(cfg, logger=logger)
        offline_prepared_path = offline_resolution.prepared_paths[0] if offline_resolution.prepared_paths else None
        offline_replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=int(cfg.env.action_dim),
            chunk_horizon=int(cfg.residual.chunk_horizon),
            discount=float(cfg.sac.discount),
            image_keys=cfg.obs.image_keys,
            capacity=int(cfg.offline.capacity),
            sample_stride=int(cfg.replay.sample_stride),
        )
        if offline_prepared_path is not None:
            load_stats = load_prepared_offline_replay(
                replay_buffer=offline_replay_buffer,
                prepared_paths=(offline_prepared_path,),
                logger=logger,
                max_episodes=cfg.offline.load_max_episodes,
                max_transitions=cfg.offline.load_max_transitions,
            )
        else:
            load_stats = {"steps_loaded": 0, "episodes_loaded": 0}
        if len(offline_replay_buffer) <= 0:
            offline_replay_buffer = None
        elif bool(cfg.replay.prepared_chunk.offline_enabled):
            offline_replay_buffer = PreparedStepWindowReplayBufferSampler(
                offline_replay_buffer,
                name="offline",
            )
        logger.info(
            "offline replay load: path=%s size=%s stats=%s",
            None if offline_prepared_path is None else str(offline_prepared_path),
            0 if offline_replay_buffer is None else int(len(offline_replay_buffer)),
            to_jsonable(load_stats),
        )
    if require_offline and offline_replay_buffer is None:
        raise RuntimeError(
            "PLD config requires offline base-success replay, but no offline replay was loaded. "
            "Set offline.prepared_path or override pld.require_offline=false for smoke tests."
        )

    latest_episode_id = 0
    env_steps = 0
    actor_replay_steps = 0
    progress_lock = Lock()

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal latest_episode_id, env_steps, actor_replay_steps
        if request_type != "send-stats":
            raise ValueError(f"invalid request type: {request_type}")
        with progress_lock:
            latest_episode_id = max(latest_episode_id, int(payload.get("episode_id", 0)))
            env_steps = max(env_steps, int(payload.get("env_steps", 0)))
            actor_replay_steps = max(actor_replay_steps, int(payload.get("replay_inserted_steps", 0)))
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

    batch_sampler = PrefetchingMixedBatchSampler(
        online_replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        batch_size=int(cfg.replay.batch_size),
        device=agent.device,
        pack_obs_and_next_obs=True,
        prefer_device_concat=True,
        thread_name_prefix="pld-learner-prefetch",
    )
    profile_accumulator = ProfileAccumulator()
    timer = Timer()
    update_steps = 0
    calql_steps_done = 0
    training_starts = int(cfg.training.training_starts)
    max_update_steps = int(cfg.training.max_update_steps)
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    learner_log_path = run_dir / "learner_timers.jsonl"

    def _committed_online_steps() -> int:
        return int(replay_buffer.latest_data_id())

    def _next_batch(*, offline_ratio: float) -> tuple[dict[str, Any], dict[str, int]]:
        with timer.context("sample_replay_buffer"):
            return batch_sampler.next_batch(offline_ratio=float(offline_ratio))

    def _run_update(*, offline_ratio: float) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal agent
        update_profile: dict[str, float] = {}
        for _ in range(max(0, int(cfg.training.critic_actor_ratio) - 1)):
            batch, _ = _next_batch(offline_ratio=float(offline_ratio))
            with timer.context("train_critics"):
                agent, _ = agent.update_critics(batch, profile=update_profile)
        batch, mix = _next_batch(offline_ratio=float(offline_ratio))
        with timer.context("train"):
            agent, info = agent.update_high_utd(
                batch,
                utd_ratio=int(cfg.sac.utd_ratio),
                profile=update_profile,
            )
        profile_accumulator.record(update_profile)
        return info, mix

    try:
        if calql_enabled and calql_steps > 0 and offline_replay_buffer is not None:
            logger.info(
                "starting Cal-QL critic pretrain: steps=%s alpha=%.4f n_actions=%s temperature=%.3f",
                int(calql_steps),
                float(calql_alpha),
                int(calql_n_actions),
                float(calql_temperature),
            )
            pretrain_bar = tqdm(total=int(calql_steps), desc="Cal-QL critic pretrain", dynamic_ncols=True, leave=True)
            while calql_steps_done < calql_steps and update_steps < max_update_steps:
                batch, _ = _next_batch(offline_ratio=1.0)
                with timer.context("calql_pretrain"):
                    agent, info = agent.update_critics_calql(
                        batch,
                        calql_alpha=float(calql_alpha),
                        calql_n_actions=int(calql_n_actions),
                        calql_temperature=float(calql_temperature),
                    )
                update_steps += 1
                calql_steps_done += 1
                if calql_steps_done % 50 == 0:
                    logger.info("calql step=%s info=%s", int(calql_steps_done), to_jsonable(info))
                pretrain_bar.update(1)
            pretrain_bar.close()
            logger.info("Cal-QL critic pretrain complete: steps=%s", int(calql_steps_done))

        if training_starts > 0:
            warmup_bar = tqdm(total=training_starts, initial=min(len(replay_buffer), training_starts), desc="online replay warmup", dynamic_ncols=True, leave=True)
            last_size = min(len(replay_buffer), training_starts)
            while len(replay_buffer) < training_starts:
                current_size = min(len(replay_buffer), training_starts)
                if current_size > last_size:
                    warmup_bar.update(current_size - last_size)
                    last_size = current_size
                time.sleep(1.0)
            current_size = min(len(replay_buffer), training_starts)
            if current_size > last_size:
                warmup_bar.update(current_size - last_size)
            warmup_bar.close()

        server.publish_network(snapshot_actor_network_payload(agent, step=int(update_steps)))
        logger.info("published initial actor network: step=%s", int(update_steps))
        last_log_time = time.time()
        last_log_update_steps = int(update_steps)
        while update_steps < max_update_steps:
            online_updates = max(0, int(update_steps - calql_steps_done))
            if not online_updates < _committed_online_steps():
                time.sleep(1.0)
                continue
            info, mix = _run_update(offline_ratio=float(cfg.offline.ratio))
            update_steps += 1
            if update_steps % int(cfg.training.steps_per_update) == 0:
                server.publish_network(snapshot_actor_network_payload(agent, step=int(update_steps)))
            if int(cfg.training.checkpoint.every_steps) > 0 and update_steps % int(cfg.training.checkpoint.every_steps) == 0:
                save_agent_checkpoint(
                    checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(cfg.training.checkpoint.keep),
                )
            if update_steps % int(cfg.training.log_period) == 0:
                now = time.time()
                elapsed = max(now - last_log_time, 1e-6)
                updates_per_sec = float(update_steps - last_log_update_steps) / elapsed
                last_log_time = now
                last_log_update_steps = int(update_steps)
                with progress_lock:
                    current_env_steps = int(env_steps)
                    current_actor_replay_steps = int(actor_replay_steps)
                    current_episode_id = int(latest_episode_id)
                payload = {
                    "source": "learner",
                    "update_steps": int(update_steps),
                    "online_update_steps": int(max(0, update_steps - calql_steps_done)),
                    "calql_pretrain_steps": int(calql_steps_done),
                    "env_steps": int(current_env_steps),
                    "actor_replay_steps": int(current_actor_replay_steps),
                    "latest_episode_id": int(current_episode_id),
                    "replay_size": int(len(replay_buffer)),
                    "offline_replay_size": 0 if offline_replay_buffer is None else int(len(offline_replay_buffer)),
                    "updates_per_sec": float(updates_per_sec),
                    "batch_mix": dict(mix),
                    "update_info": to_jsonable(info),
                    "timer": to_jsonable(timer.get_average_times()),
                    "sample_profile": to_jsonable(batch_sampler.drain_sample_profile()),
                    "update_profile": to_jsonable(profile_accumulator.drain()),
                }
                append_jsonl(learner_log_path, payload)
                logger.info(
                    "update=%s online=%s env=%s replay=%s offline=%s upd/s=%.3f mix=%s",
                    int(update_steps),
                    int(payload["online_update_steps"]),
                    int(current_env_steps),
                    int(len(replay_buffer)),
                    int(payload["offline_replay_size"]),
                    float(updates_per_sec),
                    dict(mix),
                )
    finally:
        summary = {
            "role": "learner",
            "mode": "pld_stage1",
            "update_steps": int(update_steps),
            "online_update_steps": int(max(0, update_steps - calql_steps_done)),
            "calql_pretrain_steps": int(calql_steps_done),
            "env_steps": int(env_steps),
            "actor_replay_steps": int(actor_replay_steps),
            "replay_size": int(len(replay_buffer)),
            "offline_replay_size": 0 if offline_replay_buffer is None else int(len(offline_replay_buffer)),
        }
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        batch_sampler.close()
        try:
            server.stop()
        except Exception:
            pass


@hydra.main(version_base=None, config_path="../configs", config_name="pld_libero_spatial_task4")
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("libero_pld")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))
    logger.info("PLD:\n%s", json.dumps(_pld_section(cfg), indent=2))
    set_global_seeds(typed_cfg.global_seed)
    raw_key_rl = cfg.get("key_rl")
    if raw_key_rl is not None:
        if OmegaConf.is_config(raw_key_rl):
            raw_key_rl = OmegaConf.to_container(raw_key_rl, resolve=True)
        if isinstance(raw_key_rl, dict) and bool(raw_key_rl.get("enabled", False)):
            raise ValueError("PLD Stage-1 does not support key_rl.enabled=true")
    if typed_cfg.runtime.role == "actor":
        actor(typed_cfg, cfg, run_dir=run_dir, logger=logger)
        return
    if typed_cfg.runtime.role == "learner":
        learner(typed_cfg, cfg, run_dir=run_dir, logger=logger)
        return
    raise ValueError(f"Unsupported runtime.role for PLD: {typed_cfg.runtime.role}")


if __name__ == "__main__":
    main()
