"""Repo-local robot-service launcher for AgiBot residual RL.

This is infrastructure bootstrap only; policy inference is owned by actor/eval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for path in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from serl_torch.examples.agibot_real.robot.sdk_bootstrap import (
    ensure_repo_local_a2d_sdk,
)
from serl_torch.examples.agibot_real.robot.sdk_bootstrap import (
    ensure_repo_local_forwarder,
)


def _print_help() -> None:
    print(
        "Usage: python scripts/start_robot_service.py [-s|-t] [-c CONFIG] [--no-ros]"
    )
    print()
    print("Repo-local wrapper around vendored a2d_sdk.tools.robot_service.")
    print(
        "It bootstraps the vendored AgiBot SDK wheels from examples/agibot_real/vendor/"
    )
    print("and optionally resolves forwarder assets before delegating to")
    print("a2d_sdk.tools.robot_service.main().")
    print()
    print("Forwarder resolution order when ROS is enabled:")
    print("  1. AGIBOT_FORWARDER_DIR=/path/to/extracted/forwarder")
    print("  2. AGIBOT_FORWARDER_TAR=/path/to/forwarder_x86_v1.7.0.tar.gz")
    print("  3. existing examples/agibot_real/robot/service/forwarder")
    print(
        "  4. local cache examples/agibot_real/vendor/a2d_sdk/forwarder_x86_v1.7.0.tar.gz"
    )
    print(
        "Use --no-ros (or AGIBOT_NO_ROS=1 via the shell wrapper) to skip forwarder startup."
    )


def _peek_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-ros", action="store_true")
    return parser.parse_known_args()[0]


def _append_no_ros_flag() -> None:
    if "--no-ros" not in sys.argv[1:]:
        sys.argv.append("--no-ros")


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _print_help()
        return
    args = _peek_args()
    ensure_repo_local_a2d_sdk()
    if not args.no_ros:
        try:
            ensure_repo_local_forwarder()
        except RuntimeError as exc:
            if "AgiBot forwarder bundle not found" not in str(exc):
                raise
            print(
                "Warning: forwarder bundle is missing; starting robot-service with --no-ros.",
                file=sys.stderr,
            )
            _append_no_ros_flag()
    from a2d_sdk.tools.robot_service import main as robot_service_main

    robot_service_main()


if __name__ == "__main__":
    main()
