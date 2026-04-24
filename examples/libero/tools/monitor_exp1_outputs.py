#!/usr/bin/env python3
"""Monitor LIBERO exp1 runs from shared-disk output activity."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUTS_ROOT = Path(
    "/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp1"
)
DEFAULT_RUNS = (
    "train_residual_task4_exp1_scripts_2",
    "train_residual_task4_exp1_scripts_2_no_offline",
    "train_residual_task4_exp1_scripts_5",
    "train_residual_task4_exp1_scripts_5_no_offline",
    "train_residual_task9_exp1_scripts_2",
    "train_residual_task9_exp1_scripts_2_no_offline",
    "train_residual_task9_exp1_scripts_5",
    "train_residual_task9_exp1_scripts_5_no_offline",
)
WATCH_REL_PATHS = (
    "actor/launcher.log",
    "learner/launcher.log",
    "processor/launcher.log",
    "actor/run_residual_training_2_chunk_local.log",
    "learner/run_residual_training_2_chunk_local.log",
    "actor/run_residual_training_5_split_pipeline.log",
    "learner/run_residual_training_5_split_pipeline.log",
    "processor/run_residual_training_5_split_pipeline.log",
    "actor/episode_logs.jsonl",
    "actor/actor_timers.jsonl",
    "processor/processor_timers.jsonl",
    "rollout/manifest.json",
)


@dataclass(frozen=True)
class LatestFile:
    rel_path: str
    size_bytes: int
    mtime_s: float


@dataclass(frozen=True)
class RunSnapshot:
    name: str
    exists: bool
    status: str
    latest: LatestFile | None
    rollout_pkls: int
    episodes_written: int
    steps_written: int


def _load_rollout_stats(run_dir: Path) -> tuple[int, int, int]:
    manifest_path = run_dir / "rollout" / "manifest.json"
    if not manifest_path.exists():
        return 0, 0, 0
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return 0, 0, 0
    episode_files = payload.get("episode_files", [])
    recycle_stats = dict(payload.get("recycle_stats", {}))
    return (
        len(episode_files),
        int(recycle_stats.get("episodes_written", 0)),
        int(recycle_stats.get("steps_written", 0)),
    )


def _collect_latest(run_dir: Path) -> LatestFile | None:
    latest: LatestFile | None = None
    for rel_path in WATCH_REL_PATHS:
        path = run_dir / rel_path
        if not path.exists():
            continue
        stat = path.stat()
        candidate = LatestFile(
            rel_path=rel_path,
            size_bytes=int(stat.st_size),
            mtime_s=float(stat.st_mtime),
        )
        if latest is None or candidate.mtime_s > latest.mtime_s:
            latest = candidate
    return latest


def _status_for_run(run_dir: Path, active_window_s: float) -> RunSnapshot:
    if not run_dir.exists():
        return RunSnapshot(
            name=run_dir.name,
            exists=False,
            status="MISSING",
            latest=None,
            rollout_pkls=0,
            episodes_written=0,
            steps_written=0,
        )

    latest = _collect_latest(run_dir)
    rollout_pkls, episodes_written, steps_written = _load_rollout_stats(run_dir)

    if latest is None:
        status = "EMPTY"
    else:
        age_s = max(0.0, time.time() - latest.mtime_s)
        has_training_artifacts = rollout_pkls > 0 or (run_dir / "actor" / "actor_timers.jsonl").exists()
        if age_s <= active_window_s:
            status = "ACTIVE"
        elif has_training_artifacts:
            status = "STALE"
        else:
            status = "STARTUP_ONLY"

    return RunSnapshot(
        name=run_dir.name,
        exists=True,
        status=status,
        latest=latest,
        rollout_pkls=rollout_pkls,
        episodes_written=episodes_written,
        steps_written=steps_written,
    )


def _format_age(age_s: float) -> str:
    if age_s < 60:
        return f"{age_s:4.1f}s"
    if age_s < 3600:
        return f"{age_s / 60.0:4.1f}m"
    return f"{age_s / 3600.0:4.1f}h"


def _render_table(snapshots: list[RunSnapshot], *, active_window_s: float) -> str:
    now = time.time()
    lines = []
    lines.append(
        f"shared-disk exp1 monitor  now={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}  active_window={int(active_window_s)}s"
    )
    lines.append(
        "status       episodes  steps   pkls  age    latest_file"
    )
    lines.append(
        "-----------  --------  ------  ----  -----  ----------------------------------------------"
    )
    for snapshot in snapshots:
        if snapshot.latest is None:
            lines.append(
                f"{snapshot.name}\n  {snapshot.status:<11} {'-':>8}  {'-':>6}  {'-':>4}  {'-':>5}  -"
            )
            continue
        age_s = max(0.0, now - snapshot.latest.mtime_s)
        lines.append(
            f"{snapshot.name}\n"
            f"  {snapshot.status:<11} {snapshot.episodes_written:>8}  {snapshot.steps_written:>6}  {snapshot.rollout_pkls:>4}  "
            f"{_format_age(age_s):>5}  {snapshot.latest.rel_path}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor LIBERO exp1 runs using shared-disk file activity.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help=f"Outputs root to scan (default: {DEFAULT_OUTPUTS_ROOT})",
    )
    parser.add_argument(
        "--active-window-sec",
        type=float,
        default=30.0,
        help="Mark a run ACTIVE if any watched file updated within this window.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=10.0,
        help="Refresh interval when --watch is enabled.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh continuously instead of printing one snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()

    while True:
        snapshots = [
            _status_for_run(outputs_root / run_name, active_window_s=float(args.active_window_sec))
            for run_name in DEFAULT_RUNS
        ]
        rendered = _render_table(
            snapshots,
            active_window_s=float(args.active_window_sec),
        )
        if args.watch:
            os.system("clear")
        print(rendered, flush=True)
        if not args.watch:
            return 0
        time.sleep(max(1.0, float(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
