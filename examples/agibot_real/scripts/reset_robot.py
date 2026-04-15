"""One-shot real-robot reset using the current AgiBot reset hook.

This script takes only three optional CLI parameters:

- ``--task-name``: task key used to look up the init pose table
- ``--prompt``: prompt string passed through to the reset hook
- ``--hz``: robot interface frequency used to construct ``AgiBotRobotNode``

Default behavior:

- ``--task-name agibot_real_default``
- ``--prompt "Pick up the object with the right hand and place it at the target location."``
- ``--hz 20.0``

Examples:

    python examples/agibot_real/scripts/reset_robot.py

    python examples/agibot_real/scripts/reset_robot.py --task-name office_setting

    python examples/agibot_real/scripts/reset_robot.py --task-name pour_water --hz 10

Notes:

- ``agibot_real_default`` is normalized to ``office_setting`` by the current
  init-position table.
- This script only resets the robot posture (head / waist / 16D joints).
  It does not reset task objects or scene state.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for path in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from serl_torch.examples.agibot_real.robot.interface import AgiBotRobotNode
from serl_torch.examples.agibot_real.robot.init_positions import (
    normalize_task_name_for_init_pose,
)
from serl_torch.examples.agibot_real.robot.reset_hooks import (
    reset_to_task_initial_pose,
)


DEFAULT_TASK_NAME = "agibot_real_default"
DEFAULT_PROMPT = (
    "Pick up the object with the right hand and place it at the target location."
)
DEFAULT_HZ = 20.0


class _ResetEnvShim:
    def __init__(self, robot_node: AgiBotRobotNode) -> None:
        self.robot_node = robot_node


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the AgiBot robot to the current task initial pose.",
    )
    parser.add_argument(
        "--task-name",
        default=DEFAULT_TASK_NAME,
        help=(
            "Task name used by the current init-position table. "
            f"Default: {DEFAULT_TASK_NAME}"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Task prompt passed through to the reset hook.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_HZ,
        help=f"Robot interface frequency. Default: {DEFAULT_HZ}",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    resolved_task_name = normalize_task_name_for_init_pose(str(args.task_name))
    logger.info(
        "Resetting robot to task initial pose: task_name=%s resolved_init_key=%s hz=%.3f",
        args.task_name,
        resolved_task_name,
        float(args.hz),
    )

    robot_node = AgiBotRobotNode(hz=float(args.hz))
    env = _ResetEnvShim(robot_node)
    try:
        reset_to_task_initial_pose(
            env=env,
            task_name=str(args.task_name),
            prompt=str(args.prompt),
        )
        logger.info("Robot reset completed.")
    finally:
        robot_node.shutdown()


if __name__ == "__main__":
    main()
