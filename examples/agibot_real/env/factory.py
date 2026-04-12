"""Local environment factory helpers for AgiBot real training/eval."""
from __future__ import annotations

import logging

from omegaconf import DictConfig
from omegaconf import OmegaConf

from .task_env import AgiBotTaskEnv


def _build_common_kwargs(cfg: DictConfig, logger: logging.Logger) -> dict[str, object]:
    task_cfg = cfg.get("task", {})
    robot_cfg = cfg.get("robot", {})
    controller_cfg = cfg.get("controller", {})
    return {
        "task_name": str(task_cfg.get("name", "agibot_real_task")),
        "prompt": str(task_cfg.get("prompt", task_cfg.get("name", "agibot_real_task"))),
        "action_dim": int(cfg.get("env", {}).get("action_dim", 14)),
        "control_mode": str(task_cfg.get("control_mode", "camera_position")),
        "hz": float(task_cfg.get("hz", 20.0)),
        "use_smooth_trajectory": bool(task_cfg.get("use_smooth_trajectory", False)),
        "trajectory_time": task_cfg.get("trajectory_time", None),
        "max_episode_steps": task_cfg.get("max_episode_steps", None),
        "assets_root": robot_cfg.get("assets_root", None),
        "retargeter_urdf_path": robot_cfg.get("retargeter_urdf_path", None),
        "retargeter_camera_extrinsic_path": robot_cfg.get(
            "retargeter_camera_extrinsic_path", None
        ),
        "controller": OmegaConf.to_container(controller_cfg, resolve=True),
        "reset_hook": task_cfg.get("reset_hook", None),
        "success_hook": task_cfg.get("success_hook", None),
        "expert_precheck_hook": task_cfg.get("expert_precheck_hook", None),
        "logger": logger,
    }


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "local")).strip().lower()
    if env_backend != "local":
        raise ValueError(
            "AgiBot real env is local-only; remote support has been removed, "
            f"got env.backend={env_backend!r}"
        )
    return AgiBotTaskEnv(**_build_common_kwargs(cfg, logger))
