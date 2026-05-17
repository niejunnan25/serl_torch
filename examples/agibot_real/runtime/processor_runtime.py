from __future__ import annotations

"""Standalone AgiBot rollout processor runtime."""

from collections import deque
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from agentlace.data.data_store import QueuedDataStore

from serl_launcher.common.trainer_session import TrainerClientSession
from serl_launcher.common.trainer_transport import build_actor_trainer_transport
from serl_launcher.rollout import ProcessorServer
from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.utils.timer_utils import Timer

from serl_torch.examples.agibot_real.config import AgiBotTrainConfig
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.runtime.processor_pipeline import (
    AgiBotRolloutProcessor,
)
from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RawRolloutRecorder,
)
from serl_torch.examples.agibot_real.runtime.transition_assembly import (
    AgiBotTransitionAssembler,
)

RECYCLE_RETRY_INITIAL_DELAY_S = 1.0
RECYCLE_RETRY_MAX_DELAY_S = 30.0


def _processor_backfill_obs_count(payload: dict[str, Any]) -> int:
    chunk_result = dict(payload.get("chunk_result", {}))
    if bool(payload.get("zero_step_terminal", False)):
        return 1
    observations = list(chunk_result.get("observations", ()))
    executed_steps = int(chunk_result.get("num_steps", len(observations)))
    if executed_steps <= 0:
        infos = list(chunk_result.get("infos", ()))
        executed_steps = len(infos)
    return max(1, int(executed_steps) + 1)


def _collect_processor_payload_batch(
    *,
    processor_server: ProcessorServer,
    first_payload: dict[str, Any],
    batching_cfg: Any,
) -> list[dict[str, Any]]:
    payload_batch = [dict(first_payload)]
    if not bool(getattr(batching_cfg, "enabled", False)):
        return payload_batch

    max_batch_chunks = max(1, int(getattr(batching_cfg, "max_batch_chunks", 1)))
    max_batch_obs = max(1, int(getattr(batching_cfg, "max_batch_obs", 1)))
    max_wait_s = max(0.0, float(getattr(batching_cfg, "max_wait_ms", 0)) / 1000.0)
    batch_obs = _processor_backfill_obs_count(first_payload)
    if len(payload_batch) >= max_batch_chunks or batch_obs >= max_batch_obs:
        return payload_batch

    deadline = time.monotonic() + float(max_wait_s)
    while len(payload_batch) < int(max_batch_chunks) and batch_obs < int(max_batch_obs):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        next_payload = processor_server.get_chunk(timeout_s=float(remaining_s))
        if next_payload is None:
            break
        payload_batch.append(dict(next_payload))
        batch_obs += _processor_backfill_obs_count(next_payload)
    return payload_batch


def run_processor(
    cfg: AgiBotTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    if cfg.processor.mode != "standalone":
        raise ValueError(
            "AgiBot processor role requires processor.mode=standalone; "
            "use processor.mode=in_process only for the actor process"
        )

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.backfill_policy.enabled):
        base_policy = build_agibot_base_policy(
            cfg,
            logger=logger,
            host=str(cfg.backfill_policy.host),
            port=int(cfg.backfill_policy.port),
        )
        logger.info(
            "Processor base policy backend: %s endpoint=%s:%s",
            base_policy.describe(),
            str(cfg.backfill_policy.host),
            int(cfg.backfill_policy.port),
        )
    else:
        base_policy = build_agibot_base_policy(cfg, logger=logger)
        logger.info(
            "Processor base policy backend: %s endpoint=%s:%s",
            base_policy.describe(),
            str(cfg.policy.host),
            int(cfg.policy.port),
        )
    processor_assembler_cfg = dataclasses.replace(
        cfg,
        backfill_policy=dataclasses.replace(
            cfg.backfill_policy,
            enabled=False,
        ),
    )
    transition_assembler = AgiBotTransitionAssembler(
        cfg=processor_assembler_cfg,
        base_policy=base_policy,
        logger=logger,
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
    trainer_session = TrainerClientSession(
        client=client,
        logger=logger,
        store_name="actor_env",
        status_fallback=lambda: {
            "transport_mode": str(cfg.runtime.trainer_transport.mode)
        },
        log_prefix="processor trainer transport",
    )

    rollout_processor = AgiBotRolloutProcessor(
        transition_assembler=transition_assembler,
        data_store=data_store,
        trainer_update_fn=lambda context: trainer_session.update_best_effort(
            context=str(context),
            log_prefix="processor trainer transport",
        ),
        steps_per_update=int(cfg.training.steps_per_update),
    )

    recycle_output_root = Path(cfg.recycle.output_root)
    if not recycle_output_root.is_absolute():
        recycle_output_root = (run_dir / recycle_output_root).resolve()
    recycle_recorder: RawRolloutRecorder | None = None
    if bool(cfg.recycle.enabled):
        recycle_recorder = RawRolloutRecorder(
            output_root=recycle_output_root,
            logger=logger,
            metadata={
                "task_key": str(cfg.task.task_key),
                "task_name": str(cfg.task.name),
                "policy": base_policy.describe(),
                "processor_schema_version": 1,
            },
        )
        logger.info("raw rollout recorder enabled: output_root=%s", recycle_output_root)
    recycle_retry_markers: deque[dict[str, Any]] = deque()

    def _publish_flushed_episode_markers(flushed_markers: list[dict[str, Any]]) -> None:
        for marker in flushed_markers:
            rollout_stats = marker.get("rollout_stats", None)
            if isinstance(rollout_stats, dict) and rollout_stats:
                trainer_session.request(
                    "send-stats",
                    rollout_stats,
                    context=f"episode_{int(marker.get('episode_id', -1))}_stats",
                    raise_on_exhaustion=False,
                )

    processor_server: ProcessorServer

    def _drain_flushed_episode_markers() -> None:
        processor_server.flush_ready_episode_markers()
        flushed_markers = processor_server.consume_flushed_episode_markers()
        if recycle_recorder is not None:
            ready_retry_markers: list[dict[str, Any]] = []
            deferred_retry_markers: deque[dict[str, Any]] = deque()
            now = time.monotonic()
            while recycle_retry_markers:
                retry_marker = recycle_retry_markers.popleft()
                retry_after_s = float(retry_marker.get("_recycle_retry_after_s", 0.0))
                if retry_after_s <= now:
                    ready_retry_markers.append(retry_marker)
                else:
                    deferred_retry_markers.append(retry_marker)
            recycle_retry_markers.extend(deferred_retry_markers)
            for marker in [*ready_retry_markers, *flushed_markers]:
                try:
                    recycle_recorder.finalize_episode(marker=marker)
                except Exception:
                    retry_attempt = int(marker.get("_recycle_retry_attempt", 0)) + 1
                    retry_delay_s = min(
                        float(RECYCLE_RETRY_MAX_DELAY_S),
                        float(RECYCLE_RETRY_INITIAL_DELAY_S)
                        * (2.0 ** float(max(0, retry_attempt - 1))),
                    )
                    retry_marker = dict(marker)
                    retry_marker["_recycle_retry_attempt"] = int(retry_attempt)
                    retry_marker["_recycle_retry_after_s"] = time.monotonic() + float(
                        retry_delay_s
                    )
                    recycle_retry_markers.append(retry_marker)
                    logger.exception(
                        "raw rollout finalize failed: episode_id=%s retry_attempt=%s",
                        int(marker.get("episode_id", -1)),
                        int(retry_attempt),
                    )
        _publish_flushed_episode_markers(flushed_markers)

    processor_timer_log_path = run_dir / "processor_timers.jsonl"
    timer = Timer()
    committed_env_steps = 0
    summary: dict[str, Any] = {
        "role": "processor",
        "mode": "residual",
        "transport_mode": str(cfg.runtime.trainer_transport.mode),
        "accepted_chunk_seq": -1,
        "processed_chunk_seq": -1,
        "committed_env_steps": 0,
        "timer_log_path": str(processor_timer_log_path),
        "recycle_enabled": bool(cfg.recycle.enabled),
        "recycle_output_root": str(recycle_output_root),
    }

    processor_server = ProcessorServer(
        transport_config=cfg.runtime.processor_transport,
        transport_status_fn=trainer_session.status,
        flush_transport_fn=lambda context, wait_until_committed: trainer_session.flush(
            context=str(context),
            wait_until_committed=bool(wait_until_committed),
            update_failure_message=(
                "processor trainer transport update failed repeatedly; "
                "aborting processor run"
            ),
            wait_timeout_message=(
                f"processor wait_until_committed timed out: context={str(context)}"
            ),
        ),
        wait_committed_on_episode_end=bool(
            cfg.runtime.trainer_transport.wait_committed_on_episode_end
        ),
        wait_committed_on_shutdown=bool(
            cfg.runtime.trainer_transport.wait_committed_on_shutdown
        ),
        logger=logger,
    )
    processor_server.start()
    logger.info(
        "AgiBot standalone processor started: port=%s batching=%s",
        int(cfg.runtime.processor_transport.port),
        bool(cfg.processor_batching.enabled),
    )

    try:
        while True:
            payload = processor_server.get_chunk(timeout_s=0.1)
            if payload is None:
                _drain_flushed_episode_markers()
                if processor_server.should_stop():
                    break
                continue

            payload_batch = _collect_processor_payload_batch(
                processor_server=processor_server,
                first_payload=payload,
                batching_cfg=cfg.processor_batching,
            )
            batch_chunk_seqs = [
                int(current_payload["chunk_seq"]) for current_payload in payload_batch
            ]
            batch_start = time.time()
            try:
                for current_payload in payload_batch:
                    chunk_seq = int(current_payload["chunk_seq"])
                    if bool(current_payload.get("zero_step_terminal", False)):
                        rollout_processor.finalize_zero_step_terminal(
                            terminal_reward=float(
                                current_payload.get("terminal_reward", 0.0)
                            ),
                            boundary_flag=bool(
                                current_payload.get("terminal_boundary", True)
                            ),
                            wait_for_episode_commit=False,
                        )
                        if recycle_recorder is not None:
                            try:
                                recycle_recorder.append_chunk(payload=current_payload)
                            except Exception:
                                recycle_recorder.record_append_error()
                                logger.exception(
                                    "raw rollout append failed for zero-step terminal: "
                                    "chunk_seq=%s episode_id=%s",
                                    int(chunk_seq),
                                    int(current_payload.get("episode_id", -1)),
                                )
                        processor_server.mark_chunk_committed(chunk_seq=chunk_seq)
                        continue

                    with timer.context("processor_step_chunk"):
                        processed = rollout_processor.process_payload_batch(
                            (current_payload,),
                            base_policy=base_policy,
                            image_keys=tuple(cfg.obs.image_keys),
                            residual_alpha=float(cfg.residual.alpha),
                            arm_layout=str(cfg.env.arm_layout),
                        )[0]
                    committed_env_steps += int(processed.raw.executed_steps)
                    if recycle_recorder is not None:
                        try:
                            recycle_recorder.append_chunk(payload=current_payload)
                        except Exception:
                            recycle_recorder.record_append_error()
                            logger.exception(
                                "raw rollout append failed: chunk_seq=%s episode_id=%s",
                                int(chunk_seq),
                                int(current_payload.get("episode_id", -1)),
                            )
                    processor_server.mark_chunk_committed(chunk_seq=chunk_seq)
                    _drain_flushed_episode_markers()

                timer.times["total"] += float(time.time() - batch_start)
                timer.counts["total"] += max(1, int(len(payload_batch)))
                if committed_env_steps > 0 and (
                    committed_env_steps % int(cfg.training.log_period) == 0
                ):
                    append_jsonl(
                        processor_timer_log_path,
                        {
                            "source": "processor",
                            "chunk_seq": int(batch_chunk_seqs[-1]),
                            "committed_env_steps": int(committed_env_steps),
                            "timer": timer.get_average_times(),
                            "transport": trainer_session.status(),
                            "processor": processor_server.status_snapshot(),
                            "recycle": (
                                None
                                if recycle_recorder is None
                                else recycle_recorder.status_snapshot()
                            ),
                        },
                    )
            except Exception:
                logger.exception(
                    "processor failed while processing chunk batch: chunk_seqs=%s",
                    batch_chunk_seqs,
                )
                raise
            finally:
                for _ in payload_batch:
                    try:
                        processor_server.task_done()
                    except Exception:  # noqa: BLE001
                        break

        _drain_flushed_episode_markers()
    finally:
        summary.update(
            {
                "committed_env_steps": int(committed_env_steps),
                "processor": processor_server.status_snapshot(),
                "transport": trainer_session.status(),
                "recycle": (
                    None
                    if recycle_recorder is None
                    else recycle_recorder.status_snapshot()
                ),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            if recycle_recorder is not None:
                recycle_recorder.discard_pending()
        except Exception:  # noqa: BLE001
            pass
        try:
            rollout_processor.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            base_policy.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            processor_server.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            client.stop()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "run_processor",
]
