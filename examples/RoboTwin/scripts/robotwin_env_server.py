from __future__ import annotations

"""
RoboTwin 环境 RPC 服务（运行在 robotwin2 conda 环境）。

用法示例：
python scripts/robotwin_env_server.py --host 127.0.0.1 --port 9100
"""

import argparse
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from env_wrappers.task_env import RoboTwinTaskEnv
from env_wrappers.setup import load_task_args, resolve_robo_root, setup_robotwin_pythonpath
from serl_launcher.envs.remote_http import make_pickle_rpc_handler


LOGGER = logging.getLogger("robotwin_env_server")


class _EnvState:
    def __init__(self) -> None:
        self.env: Optional[RoboTwinTaskEnv] = None

    def _meta(self) -> Dict[str, Any]:
        if self.env is None:
            return {
                "current_instruction": None,
                "step_limit": 0,
                "take_action_cnt": 0,
                "last_seed": None,
            }
        try:
            return {
                "current_instruction": self.env.current_instruction,
                "step_limit": int(self.env.step_limit),
                "take_action_cnt": int(self.env.take_action_cnt),
                "last_seed": self.env.last_seed,
            }
        except Exception:  # noqa: BLE001
            return {
                "current_instruction": getattr(self.env, "_current_instruction", None),
                "step_limit": 0,
                "take_action_cnt": 0,
                "last_seed": getattr(self.env, "last_seed", None),
            }

    def create_env(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is not None:
            try:
                self.env.close(clear_cache=False)
            except Exception:  # noqa: BLE001
                pass
            self.env = None

        robo_root_raw = kwargs.pop("robo_root", None)
        robo_root = resolve_robo_root(robo_root_raw)
        setup_robotwin_pythonpath(robo_root)
        # RoboTwin 内部大量资源使用相对路径（如 assets/...），
        # 因此服务端需要切到 RoboTwin 根目录再实例化环境。
        os.chdir(robo_root)
        LOGGER.info("server cwd switched to RoboTwin root: %s", robo_root)

        task_name = str(kwargs["task_name"])
        task_args = kwargs.get("task_args", {})
        if not isinstance(task_args, dict):
            task_args = {}
        task_config = task_args.get("task_config", kwargs.get("task_config", None))
        if task_config is not None:
            task_args = load_task_args(robo_root, task_name, str(task_config))
        kwargs["task_name"] = task_name
        kwargs["task_args"] = task_args

        self.env = RoboTwinTaskEnv(**kwargs)
        return self._meta()

    def get_meta(self) -> Dict[str, Any]:
        return self._meta()

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is not created")
        obs = self.env.reset(**kwargs)
        return {"obs": obs, "meta": self._meta()}

    def expert_precheck(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is not created")
        passed, episode_info = self.env.expert_precheck(**kwargs)
        return {
            "passed": bool(passed),
            "episode_info": episode_info if isinstance(episode_info, dict) else None,
            "meta": self._meta(),
        }

    def step(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is not created")
        action = np.asarray(kwargs["action"], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return {
            "obs": obs,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": dict(info),
            "meta": self._meta(),
        }

    def close(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is not None:
            self.env.close(clear_cache=bool(kwargs.get("clear_cache", False)))
            self.env = None
        return {"closed": True}


STATE = _EnvState()
Handler: type[BaseHTTPRequestHandler] = make_pickle_rpc_handler(
    STATE,
    LOGGER,
    keep_alive=False,
)


def _sapien_render_test() -> None:
    """Run a minimal SAPIEN render test (mirrors RoboTwin's Sapien_TEST).

    If the GPU / Vulkan renderer cannot initialise, we fail fast instead of
    silently returning False on every expert_precheck.
    """
    try:
        import sapien.core as sapien
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        scene_config = sapien.SceneConfig()
        _scene = engine.create_scene(scene_config)
        LOGGER.info("SAPIEN render test PASSED")
    except Exception as exc:
        LOGGER.error("SAPIEN render test FAILED: %s", exc)
        raise SystemExit(
            "Cannot initialise SAPIEN renderer. "
            "Check CUDA_VISIBLE_DEVICES and GPU availability."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--skip-render-test", action="store_true",
        help="Skip the startup SAPIEN render check",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    if not args.skip_render_test:
        _sapien_render_test()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("RoboTwin env server started at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        try:
            STATE.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
