"""Episode reset hooks aligned with the reference AgiRobot.reset flow."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from .init_positions import get_task_initial_pose
from .init_positions import normalize_task_name_for_init_pose

logger = logging.getLogger(__name__)


def reset_to_task_initial_pose(
    *,
    env: Any,
    task_name: str,
    prompt: str,
) -> None:
    """Move robot to task-specific initial pose (head/waist/joint).

    Uses the bundled init-position table and AgiBotRobotNode.publish_*.
    """
    robot_node = getattr(env, "robot_node", None)
    if robot_node is None:
        logger.warning(
            "reset_to_task_initial_pose skipped: env has no robot_node (wrong env type?)",
        )
        return

    # Optional tuning (seconds), same order as agi_robot.AgiRobot.reset
    sleep_hw = float(os.environ.get("AGIBOT_RESET_SLEEP_HEAD_WAIST_SEC", "2.0"))
    sleep_arm = float(os.environ.get("AGIBOT_RESET_SLEEP_ARM_SEC", "1.0"))

    init_key = normalize_task_name_for_init_pose(str(task_name))
    try:
        head_action, waist_action, joint_action = get_task_initial_pose(init_key)
    except ValueError as exc:
        logger.warning(
            "reset_to_task_initial_pose skipped (unknown task_name=%s): %s",
            task_name,
            exc,
        )
        return
    head = np.asarray(head_action, dtype=np.float32)
    waist = np.asarray(waist_action, dtype=np.float32)
    joint = np.asarray(joint_action, dtype=np.float32).reshape(-1)
    if joint.shape[0] != 16:
        raise ValueError(f"joint_action must be 16D, got shape {joint.shape}")

    logger.info(
        "reset_to_task_initial_pose: task_name=%s init_key=%s",
        task_name,
        init_key,
    )
    robot_node.publish_head_command(head)
    robot_node.publish_waist_command(waist)
    time.sleep(sleep_hw)
    robot_node.publish_joint_command_reset(joint)
    time.sleep(sleep_arm)
