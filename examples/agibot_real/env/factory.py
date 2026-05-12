"""Local environment factory helpers for AgiBot real training/eval."""
from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from ..config import AgiBotEvalConfig
from ..config import AgiBotRunConfig
from ..config import AgiBotTrainConfig
from .fake_task_env import AgiBotFakeTaskEnv
from .task_env import AgiBotTaskEnv


def _build_common_kwargs(
    cfg: DictConfig | AgiBotRunConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    if isinstance(cfg, (AgiBotTrainConfig, AgiBotEvalConfig)):
        return {
            "task_name": str(cfg.task.name),
            "prompt": str(cfg.task.prompt),
            "arm_layout": str(cfg.env.arm_layout),
            "action_dim": int(cfg.env.action_dim),
            "robot_action_dim": int(cfg.env.robot_action_dim),
            "control_mode": str(cfg.task.control_mode),
            "hz": float(cfg.task.hz),
            "use_smooth_trajectory": bool(cfg.task.use_smooth_trajectory),
            "trajectory_time": cfg.task.trajectory_time,
            "max_episode_steps": cfg.task.max_episode_steps,
            "assets_root": cfg.robot.assets_root,
            "retargeter_urdf_path": cfg.robot.retargeter_urdf_path,
            "retargeter_camera_extrinsic_path": cfg.robot.retargeter_camera_extrinsic_path,
            "controller": asdict(cfg.controller),
            "reset_hook": cfg.task.reset_hook,
            "success_hook": cfg.task.success_hook,
            "logger": logger,
        }

    task_cfg = cfg.get("task", {})
    robot_cfg = cfg.get("robot", {})
    env_cfg = cfg.get("env", {})
    controller_cfg = cfg.get("controller", {})
    return {
        "task_name": str(task_cfg.get("name", "agibot_real_task")),
        "prompt": str(task_cfg.get("prompt", task_cfg.get("name", "agibot_real_task"))),
        "arm_layout": str(env_cfg.get("arm_layout", "dual_arm")),
        "action_dim": int(env_cfg.get("action_dim", 14)),
        "robot_action_dim": int(env_cfg.get("robot_action_dim", 14)),
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
        "logger": logger,
    }


def _resolve_env_backend(cfg: DictConfig | AgiBotRunConfig) -> str:
    if isinstance(cfg, (AgiBotTrainConfig, AgiBotEvalConfig)):
        return str(cfg.env.backend).strip().lower()
    return str(cfg.get("env", {}).get("backend", "local")).strip().lower()


def _resolve_controller_enabled(cfg: DictConfig | AgiBotRunConfig) -> bool:
    if isinstance(cfg, (AgiBotTrainConfig, AgiBotEvalConfig)):
        return bool(cfg.controller.enabled)
    return bool(cfg.get("controller", {}).get("enabled", False))


def create_env(cfg: DictConfig | AgiBotRunConfig, logger: logging.Logger):
    env_backend = _resolve_env_backend(cfg)
    if env_backend not in {"local", "fake"}:
        raise ValueError(
            "AgiBot env backend must be 'local' or 'fake'; remote support has been removed, "
            f"got env.backend={env_backend!r}"
        )
    if not _resolve_controller_enabled(cfg):
        raise ValueError(
            "AgiBot canonical train/eval flow requires controller.enabled=true"
        )
    kwargs = _build_common_kwargs(cfg, logger)
    if env_backend == "fake":
        kwargs["reset_hook"] = None
        kwargs["success_hook"] = None
        return AgiBotFakeTaskEnv(**kwargs)
    return AgiBotTaskEnv(**kwargs)


def _create_env(cfg: DictConfig | AgiBotRunConfig, logger: logging.Logger):
    """Backwards-compatible alias for older call sites."""

    return create_env(cfg, logger)


__all__ = ["create_env"]
