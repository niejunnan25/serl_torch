#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_NAMES = [
    "train_residual_task4_exp1_scripts_2",
    "train_residual_task4_exp1_scripts_2_no_offline",
    "train_residual_task4_exp1_scripts_5",
    "train_residual_task4_exp1_scripts_5_no_offline",
    "train_residual_task9_exp1_scripts_2",
    "train_residual_task9_exp1_scripts_2_no_offline",
    "train_residual_task9_exp1_scripts_5",
    "train_residual_task9_exp1_scripts_5_no_offline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success_rate_20 for all exp1 training runs."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("examples/libero/outputs/exp1"),
        help="Root directory that contains exp1 run outputs.",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs"),
        help="Directory where PNG plots will be written.",
    )
    return parser.parse_args()


def load_points(log_path: Path) -> tuple[list[int], list[float]]:
    episodes: list[int] = []
    success_rate_20: list[float] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rollout = payload.get("rollout") or {}
            episode_id = rollout.get("episode_id")
            recent_success_rate_20 = rollout.get("recent_success_rate_20")
            if episode_id is None or recent_success_rate_20 is None:
                continue
            episodes.append(int(episode_id))
            success_rate_20.append(float(recent_success_rate_20))
    return episodes, success_rate_20


def plot_run(run_name: str, log_path: Path, docs_root: Path) -> Path:
    episodes, success_rate_20 = load_points(log_path)
    if not episodes:
        raise ValueError(f"no recent_success_rate_20 data found in {log_path}")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.plot(episodes, success_rate_20, color="#1768AC", linewidth=1.8)
    ax.set_title(run_name.replace("_", " "))
    ax.set_xlabel("Episode")
    ax.set_ylabel("success_rate_20")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    latest_episode = episodes[-1]
    latest_value = success_rate_20[-1]
    ax.scatter([latest_episode], [latest_value], color="#D7263D", s=24, zorder=3)
    ax.text(
        latest_episode,
        latest_value,
        f"  ep={latest_episode}, sr20={latest_value:.2f}",
        fontsize=8,
        va="bottom",
    )

    fig.tight_layout()
    output_path = docs_root / f"2026-04-24-{run_name}-success-rate-20.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    docs_root = args.docs_root.resolve()
    docs_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for run_name in RUN_NAMES:
        log_path = outputs_root / run_name / "actor" / "episode_logs.jsonl"
        if not log_path.exists():
            raise FileNotFoundError(f"missing episode log: {log_path}")
        written.append(plot_run(run_name, log_path, docs_root))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
