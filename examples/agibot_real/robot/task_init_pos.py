"""Resolve init_node_pos(task_name) for episode reset using the bundled table."""

from __future__ import annotations

def init_node_pos(task_name: str):
    # Align with init_node_pos_data: default SERL task name uses office_setting pose.
    if str(task_name).strip() == "agibot_real_default":
        task_name = "office_setting"
    from .init_node_pos_data import init_node_pos as _builtin

    return _builtin(task_name)
