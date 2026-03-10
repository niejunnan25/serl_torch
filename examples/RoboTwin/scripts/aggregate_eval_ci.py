#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def _load_success_rate(eval_dir: Path) -> float:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "success_rate" not in summary:
        raise KeyError(f"success_rate missing in {summary_path}")
    return float(summary["success_rate"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed eval success rates (mean + 95% CI).")
    parser.add_argument("--eval-dirs", nargs="+", required=True, help="List of eval output directories")
    parser.add_argument("--out", required=True, help="Output json path")
    args = parser.parse_args()

    eval_dirs: List[Path] = [Path(p).resolve() for p in args.eval_dirs]
    success_rates = np.asarray([_load_success_rate(p) for p in eval_dirs], dtype=np.float64)

    n = int(success_rates.size)
    mean = float(np.mean(success_rates)) if n > 0 else 0.0
    std = float(np.std(success_rates, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(max(1, n)))

    payload = {
        "num_seeds": n,
        "eval_dirs": [str(p) for p in eval_dirs],
        "success_rates": success_rates.tolist(),
        "mean_success_rate": mean,
        "std_success_rate": std,
        "ci95": ci95,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
