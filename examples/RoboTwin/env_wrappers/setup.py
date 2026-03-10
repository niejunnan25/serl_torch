"""RoboTwin 环境初始化：路径解析、任务参数加载、环境实例化。"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.paths import _find_serl_repo_root


# ---------------------------------------------------------------------------
# RoboTwin 项目根目录解析
# ---------------------------------------------------------------------------


def resolve_robo_root(robo_root: Optional[str]) -> Path:
    """
    解析 RoboTwin 项目根目录路径。

    优先使用用户传入的 ``robo_root``；否则自动搜索
    ``serl_torch`` 仓库的同级目录或子目录。
    """
    if robo_root:
        root = Path(robo_root).expanduser().resolve()
    else:
        repo_root = _find_serl_repo_root()
        sibling_candidate = (repo_root.parent / "RoboTwin").resolve()
        local_candidate = (repo_root / "RoboTwin").resolve()
        if sibling_candidate.exists():
            root = sibling_candidate
        elif local_candidate.exists():
            root = local_candidate
        else:
            root = sibling_candidate

    if not root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {root}")
    return root


def setup_robotwin_pythonpath(robo_root: Path) -> None:
    """将 RoboTwin 依赖的子目录添加到 ``sys.path``。"""
    os.environ["ROBOTWIN_ROOT"] = str(robo_root)
    candidates = [
        robo_root,
        robo_root / "policy",
        robo_root / "description" / "utils",
        robo_root / "packages" / "openpi-client" / "src",
    ]
    for item in candidates:
        item_str = str(item)
        if item_str not in sys.path:
            sys.path.append(item_str)


# ---------------------------------------------------------------------------
# RoboTwin envs 模块导入
# ---------------------------------------------------------------------------


def _get_robotwin_configs_path(robo_root: Path) -> str:
    """从 RoboTwin 的 envs 包中获取 CONFIGS_PATH。"""
    from envs import CONFIGS_PATH
    return str(CONFIGS_PATH)


# ---------------------------------------------------------------------------
# 任务参数加载
# ---------------------------------------------------------------------------


def _resolve_robot_file(robo_root: Path, maybe_relative: str) -> str:
    """把可能的相对路径转为绝对路径。"""
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return str(candidate)
    return str((robo_root / candidate).resolve())


def load_embodiment_config(robot_file: str) -> Dict[str, Any]:
    """加载机器人实体配置（YAML）。"""
    robot_cfg_file = Path(robot_file) / "config.yml"
    with open(robot_cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task_args(robo_root: Path, task_name: str, task_config: str) -> Dict[str, Any]:
    """
    加载任务参数：读取 task_config YAML、解析 embodiment 与 camera 配置。

    返回一个完整的 args dict，可直接传给 ``env.setup_demo(**args)``。
    """
    CONFIGS_PATH = _get_robotwin_configs_path(robo_root)

    task_cfg_path = robo_root / "task_config" / f"{task_config}.yml"
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["eval_video_log"] = False

    embodiment_type = args.get("embodiment")
    embodiment_cfg_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    with open(embodiment_cfg_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(item: str) -> str:
        robot_file = embodiment_types[item]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {item}")
        return _resolve_robot_file(robo_root, robot_file)

    camera_cfg_path = Path(CONFIGS_PATH) / "_camera_config.yml"
    with open(camera_cfg_path, "r", encoding="utf-8") as f:
        camera_cfg = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should have length 1 or 3")

    args["left_embodiment_config"] = load_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = load_embodiment_config(args["right_robot_file"])

    return args


# ---------------------------------------------------------------------------
# 环境实例化
# ---------------------------------------------------------------------------


def instantiate_task(task_name: str):
    """根据任务名动态 import 并实例化 RoboTwin 环境类。"""
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()
