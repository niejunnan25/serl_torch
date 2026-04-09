"""Episode reset hooks aligned with tangyili/code/agibot/agi_robot.py AgiRobot.reset."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from .task_init_pos import init_node_pos

logger = logging.getLogger(__name__)

# SERL 默认 task.name；与 tangyili 中 office_setting 初始位姿一致（见 init_node_pos_data）。
_DEFAULT_TASK_ALIAS = "office_setting"


def _normalize_task_name_for_init(task_name: str) -> str:
    """Map SERL-only names so external init_node_pos.py (AGIBOT_CODE_ROOT) also resolves."""
    if str(task_name).strip() == "agibot_real_default":
        return _DEFAULT_TASK_ALIAS
    return str(task_name)


def reset_to_task_initial_pose(
    *,
    env: Any,
    seed: int,
    init_episode_idx: int,
    episode_info: dict[str, Any] | None,
    task_name: str,
    prompt: str,
) -> None:
    """Move robot to task-specific initial pose (head/waist/joint), matching inference_camera_position.

    Uses init_node_pos(task_name) and AgiBotRobotNode.publish_*; see task_init_pos.py for data source.

    Only valid for local :class:`AgiBotTaskEnv` (has ``robot_node``). Remote env runs this hook on the server.
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

    init_key = _normalize_task_name_for_init(str(task_name))
    try:
        head_action, waist_action, joint_action = init_node_pos(init_key)
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
        "reset_to_task_initial_pose: task_name=%s init_key=%s seed=%s init_episode_idx=%s",
        task_name,
        init_key,
        seed,
        init_episode_idx,
    )
    robot_node.publish_head_command(head)
    robot_node.publish_waist_command(waist)
    time.sleep(sleep_hw)
    robot_node.publish_joint_command_reset(joint)
    time.sleep(sleep_arm)
