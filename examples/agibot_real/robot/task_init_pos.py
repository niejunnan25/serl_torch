"""Compatibility shim for the merged AgiBot init-positions module."""

from .init_positions import get_task_initial_pose
from .init_positions import init_node_pos
from .init_positions import normalize_task_name_for_init_pose

__all__ = [
    "get_task_initial_pose",
    "init_node_pos",
    "normalize_task_name_for_init_pose",
]
