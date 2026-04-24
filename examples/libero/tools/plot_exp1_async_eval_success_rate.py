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
        description="Plot async eval success rates for all exp1 training runs."
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


def load_points(results_path: Path) -> tuple[list[int], list[float]]:
    train_episode_ids: list[int] = []
    success_rates: list[float] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("status") != "ok":
                continue
            train_episode_id = payload.get("train_episode_id")
            summary = payload.get("summary") or {}
            success_rate = summary.get("success_rate")
            if train_episode_id is None or success_rate is None:
                continue
            train_episode_ids.append(int(train_episode_id))
            success_rates.append(float(success_rate))
    return train_episode_ids, success_rates


def plot_run(run_name: str, results_path: Path, docs_root: Path) -> Path:
    train_episode_ids, success_rates = load_points(results_path)
    if not train_episode_ids:
        raise ValueError(f"no async eval success_rate data found in {results_path}")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.plot(
        train_episode_ids,
        success_rates,
        color="#0B6E4F",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
    )
    ax.set_title(f"{run_name.replace('_', ' ')} async eval")
    ax.set_xlabel("Train episode")
    ax.set_ylabel("async_eval_success_rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    latest_episode = train_episode_ids[-1]
    latest_value = success_rates[-1]
    ax.scatter([latest_episode], [latest_value], color="#D7263D", s=28, zorder=3)
    ax.text(
        latest_episode,
        latest_value,
        f"  ep={latest_episode}, eval_sr={latest_value:.2f}",
        fontsize=8,
        va="bottom",
    )

    fig.tight_layout()
    output_path = docs_root / f"2026-04-24-{run_name}-async-eval-success-rate.png"
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
        results_path = outputs_root / run_name / "learner" / "async_eval_results.jsonl"
        if not results_path.exists():
            raise FileNotFoundError(f"missing async eval results: {results_path}")
        written.append(plot_run(run_name, results_path, docs_root))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
