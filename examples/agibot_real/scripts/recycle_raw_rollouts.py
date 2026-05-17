from __future__ import annotations

"""Recycle AgiBot raw rollout records through the processor pipeline."""

import argparse
import dataclasses
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.data.offline_prepared import MANIFEST_FILENAME
from serl_torch.examples.agibot_real.config import parse_train_cfg
from serl_torch.examples.agibot_real.env.base_policy import build_agibot_base_policy
from serl_torch.examples.agibot_real.env.offline_data import (
    training_compatibility_signature,
)
from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RAW_ROLLOUT_FORMAT_VERSION,
)
from serl_torch.examples.agibot_real.runtime.raw_rollout_recorder import (
    RAW_ROLLOUT_MANIFEST_FILENAME,
)
from serl_torch.examples.agibot_real.runtime.transition_assembly import (
    AgiBotTransitionAssembler,
)
from serl_torch.examples.agibot_real.runtime.transition_assembly import RawChunkRecord


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _load_raw_manifest(raw_rollout_root: Path) -> dict[str, Any]:
    manifest_path = raw_rollout_root / RAW_ROLLOUT_MANIFEST_FILENAME
    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)
    if not isinstance(manifest, dict):
        raise ValueError(f"raw rollout manifest must be a JSON object: {manifest_path}")
    if manifest.get("format_version") != RAW_ROLLOUT_FORMAT_VERSION:
        raise ValueError(
            "unsupported raw rollout format: "
            f"{manifest.get('format_version')!r} expected {RAW_ROLLOUT_FORMAT_VERSION!r}"
        )
    return manifest


def recycle_raw_rollouts(
    *,
    config_file: Path,
    raw_rollout_root: Path,
    output_root: Path,
    dry_run: bool,
    limit_episodes: int | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    cfg = parse_train_cfg(OmegaConf.load(config_file))
    if bool(cfg.replay.prepared_chunk.online_enabled):
        raise ValueError(
            "AgiBot raw rollout recycle refuses replay.prepared_chunk.online_enabled=true; "
            "only offline prepared paths are supported"
        )
    cfg = dataclasses.replace(
        cfg,
        backfill_policy=dataclasses.replace(cfg.backfill_policy, enabled=False),
    )

    raw_rollout_root = Path(raw_rollout_root).resolve()
    output_root = Path(output_root).resolve()
    manifest = _load_raw_manifest(raw_rollout_root)
    episode_files = [str(path) for path in list(manifest.get("episode_files", ()))]
    if limit_episodes is not None:
        episode_files = episode_files[: int(limit_episodes)]

    base_policy = build_agibot_base_policy(cfg, logger=logger)
    assembler = AgiBotTransitionAssembler(
        cfg=cfg,
        base_policy=base_policy,
        logger=logger,
    )

    summary = {
        "raw_rollout_root": str(raw_rollout_root),
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "input_episodes": 0,
        "input_chunks": 0,
        "input_steps": 0,
        "generated_transitions": 0,
        "skipped_episodes": 0,
        "failures": [],
        "episode_files": [],
    }

    try:
        for episode_name in episode_files:
            episode_path = raw_rollout_root / episode_name
            try:
                episode_payload = _load_pickle(episode_path)
                if episode_payload.get("format_version") != RAW_ROLLOUT_FORMAT_VERSION:
                    raise ValueError(
                        f"unsupported episode format: {episode_payload.get('format_version')!r}"
                    )
                chunks = [dict(chunk) for chunk in list(episode_payload.get("chunks", ()))]
                if not chunks:
                    raise ValueError("raw rollout episode has no chunks")

                transitions: list[dict[str, Any]] = []
                input_steps = 0
                for chunk_payload in chunks:
                    if bool(chunk_payload.get("zero_step_terminal", False)):
                        if transitions:
                            transitions[-1]["rewards"] = float(
                                transitions[-1]["rewards"]
                            ) + float(chunk_payload.get("terminal_reward", 0.0))
                            transitions[-1]["dones"] = bool(
                                chunk_payload.get("terminal_boundary", True)
                            )
                            transitions[-1]["masks"] = 0.0
                        continue
                    raw = RawChunkRecord.from_submission_payload(
                        chunk_payload,
                        base_policy=base_policy,
                        image_keys=tuple(cfg.obs.image_keys),
                        residual_alpha=float(cfg.residual.alpha),
                        arm_layout=str(cfg.env.arm_layout),
                    )
                    assembled = assembler.process_chunk(
                        raw=raw,
                        task_prompt=str(chunk_payload["task_prompt"]),
                    )
                    transitions.extend(assembled.transitions)
                    input_steps += int(raw.executed_steps)

                summary["input_episodes"] += 1
                summary["input_chunks"] += len(chunks)
                summary["input_steps"] += int(input_steps)
                summary["generated_transitions"] += len(transitions)
                if not bool(dry_run):
                    output_episode_name = f"episode_{int(episode_payload['episode_id']):06d}.pkl"
                    _write_pickle(output_root / output_episode_name, transitions)
                    summary["episode_files"].append(output_episode_name)
            except Exception as exc:  # noqa: BLE001
                summary["skipped_episodes"] += 1
                summary["failures"].append(
                    {
                        "episode_file": str(episode_name),
                        "reason": str(exc),
                    }
                )
                logger.exception("failed to recycle raw rollout episode: %s", episode_name)

        if not bool(dry_run):
            output_root.mkdir(parents=True, exist_ok=True)
            prepared_manifest = {
                "fingerprint": {
                    "format_version": "agibot_recycled_offline_step_transitions_v1",
                    "task_key": str(cfg.task.task_key),
                    "task_description": str(cfg.task.prompt),
                    "raw_dataset_path": str(raw_rollout_root),
                    **training_compatibility_signature(cfg),
                },
                "episode_files": list(summary["episode_files"]),
                "prepare_stats": {
                    "episodes_written": int(summary["input_episodes"]),
                    "transitions_written": int(summary["generated_transitions"]),
                    "source_raw_rollout_root": str(raw_rollout_root),
                },
            }
            with open(output_root / MANIFEST_FILENAME, "w", encoding="utf-8") as fp:
                json.dump(prepared_manifest, fp, indent=2, ensure_ascii=False)
    finally:
        try:
            assembler.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            base_policy.close()
        except Exception:  # noqa: BLE001
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--raw-rollout-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("agibot_recycle_raw_rollouts")
    summary = recycle_raw_rollouts(
        config_file=args.config_file,
        raw_rollout_root=args.raw_rollout_root,
        output_root=args.output_root,
        dry_run=bool(args.dry_run),
        limit_episodes=args.limit_episodes,
        logger=logger,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
