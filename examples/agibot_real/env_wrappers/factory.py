"""Environment factory helpers for AgiBot real training/eval."""
from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from .remote_task_env import RemoteAgiBotTaskEnv
from .task_env import AgiBotTaskEnv


def _resolve_robot_asset_path(
    cfg: DictConfig,
    key: str,
    default_name: str,
) -> str:
    robot_cfg = cfg.get("robot", {})
    explicit = robot_cfg.get(key, None)
    if explicit is not None:
        return str(Path(str(explicit)).expanduser().resolve())
    assets_root = robot_cfg.get("assets_root", None)
    if assets_root is not None:
        return str((Path(str(assets_root)).expanduser() / "G1" / default_name).resolve())
    return str(
        (Path(__file__).resolve().parents[1] / "assets" / "G1" / default_name).resolve()
    )


def _build_common_kwargs(cfg: DictConfig, logger: logging.Logger) -> dict[str, object]:
    task_cfg = cfg.get("task", {})
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
        "retargeter_urdf_path": _resolve_robot_asset_path(cfg, "retargeter_urdf_path", "model.urdf"),
        "retargeter_camera_extrinsic_path": _resolve_robot_asset_path(
            cfg,
            "retargeter_camera_extrinsic_path",
            "head_extrinsic_ours.json",
        ),
        "controller": OmegaConf.to_container(controller_cfg, resolve=True),
        "reset_hook": task_cfg.get("reset_hook", None),
        "success_hook": task_cfg.get("success_hook", None),
        "expert_precheck_hook": task_cfg.get("expert_precheck_hook", None),
        "logger": logger,
    }


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
    common_kwargs = _build_common_kwargs(cfg, logger)
    if env_backend == "local":
        return AgiBotTaskEnv(**common_kwargs)
    if env_backend == "remote":
        remote_cfg = cfg.get("env", {}).get("remote", {})
        return RemoteAgiBotTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 32000)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 180.0)),
            **common_kwargs,
        )
    raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")
