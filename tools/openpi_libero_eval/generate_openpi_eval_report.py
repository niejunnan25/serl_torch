#!/usr/bin/env python3
"""Render one strategy's LIBERO evaluation results into a Markdown report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for one OpenPI LIBERO eval run")
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--strategy-root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--policy-script", required=True)
    parser.add_argument("--policy-port", required=True)
    parser.add_argument(
        "--suite",
        dest="suites",
        action="append",
        choices=EXPECTED_SUITES,
        help="Limit the report to one or more suites. Defaults to all canonical suites.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strategy_root = Path(args.strategy_root).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    suites = tuple(args.suites) if args.suites else EXPECTED_SUITES

    suite_payloads: list[dict[str, Any]] = []
    for suite_name in suites:
        suite_summary_path = strategy_root / suite_name / "suite_summary.json"
        if not suite_summary_path.is_file():
            raise FileNotFoundError(f"Missing suite summary: {suite_summary_path}")
        suite_payloads.append(_load_json(suite_summary_path))

    total_episodes = sum(int(payload["episodes_total"]) for payload in suite_payloads)
    total_successes = sum(int(payload["successes_total"]) for payload in suite_payloads)
    weighted_success_rate = float(total_successes) / float(max(1, total_episodes))
    mean_suite_success_rate = sum(
        float(payload["success_rate_total"]) for payload in suite_payloads
    ) / float(max(1, len(suite_payloads)))

    lines: list[str] = []
    lines.append(f"# {args.strategy_name} LIBERO Eval Report")
    lines.append("")
    lines.append(f"生成时间（北京时间）: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append("## 评测配置")
    lines.append("")
    lines.append(f"- 策略名称: `{args.strategy_name}`")
    lines.append(f"- GPU: `{args.gpu_id}`")
    lines.append(f"- Policy script: `{args.policy_script}`")
    lines.append(f"- Policy port: `{args.policy_port}`")
    lines.append(f"- 结果目录: `{strategy_root}`")
    lines.append(f"- 套件数: `{len(suite_payloads)}`")
    lines.append(f"- 套件列表: `{', '.join(suites)}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 总 episode 数: `{total_episodes}`")
    lines.append(f"- 总成功数: `{total_successes}`")
    lines.append(f"- 加权总成功率: `{_format_pct(weighted_success_rate)}`")
    lines.append(f"- Suite 平均成功率: `{_format_pct(mean_suite_success_rate)}`")
    lines.append("")
    lines.append("| Suite | Success Rate | Successes | Episodes |")
    lines.append("| --- | --- | --- | --- |")
    for payload in suite_payloads:
        lines.append(
            f"| `{payload['suite_name']}` | `{_format_pct(float(payload['success_rate_total']))}` | "
            f"`{int(payload['successes_total'])}` | `{int(payload['episodes_total'])}` |"
        )
    lines.append("")

    for payload in suite_payloads:
        suite_name = str(payload["suite_name"])
        lines.append(f"## {suite_name}")
        lines.append("")
        lines.append(
            f"- Suite 成功率: `{_format_pct(float(payload['success_rate_total']))}` "
            f"(`{int(payload['successes_total'])}/{int(payload['episodes_total'])}`)"
        )
        lines.append("")
        lines.append("| Task | Success Rate | Successes | Episodes | Description |")
        lines.append("| --- | --- | --- | --- | --- |")
        tasks = sorted(payload["tasks"], key=lambda item: int(item["task_id"]))
        for task in tasks:
            lines.append(
                f"| `{int(task['task_id'])}` | `{_format_pct(float(task['success_rate']))}` | "
                f"`{int(task['successes'])}` | `{int(task['episodes'])}` | "
                f"{str(task['task_description']).replace('|', '/')} |"
            )
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
