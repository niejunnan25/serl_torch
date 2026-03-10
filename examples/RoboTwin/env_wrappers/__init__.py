"""环境封装：任务初始化、环境交互、指令生成。"""
from env_wrappers.setup import (
    instantiate_task,
    load_embodiment_config,
    load_task_args,
    resolve_robo_root,
    setup_robotwin_pythonpath,
)
from env_wrappers.instruction import generate_instruction_from_episode_info
from env_wrappers.task_env import RoboTwinTaskEnv
from env_wrappers.remote_task_env import RemoteRoboTwinTaskEnv

__all__ = [
    "instantiate_task",
    "load_embodiment_config",
    "load_task_args",
    "resolve_robo_root",
    "setup_robotwin_pythonpath",
    "generate_instruction_from_episode_info",
    "RoboTwinTaskEnv",
    "RemoteRoboTwinTaskEnv",
]
