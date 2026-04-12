from __future__ import annotations

"""HTTP RPC server for LIBERO environments (run inside the `libero` conda env)."""

import argparse
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PARENT = REPO_ROOT.parent
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_torch.examples.libero.env_wrappers.task_env import LiberoTaskEnv
from serl_launcher.envs.remote_http import make_pickle_rpc_handler

LOGGER = logging.getLogger("libero_env_server")


class _EnvState:
    def __init__(self) -> None:
        self.env: Optional[LiberoTaskEnv] = None

    def _meta(self) -> Dict[str, Any]:
        if self.env is None:
            return {
                "current_instruction": None,
                "task_description": None,
                "step_limit": 0,
                "take_action_cnt": 0,
                "action_dim": 0,
                "last_seed": None,
                "current_init_state_idx": None,
            }
        return {
            "current_instruction": self.env.current_instruction,
            "task_description": self.env.task_description,
            "step_limit": int(self.env.step_limit),
            "take_action_cnt": int(self.env.take_action_cnt),
            "action_dim": int(self.env.action_dim),
            "last_seed": self.env.last_seed,
            "current_init_state_idx": self.env.current_init_state_idx,
        }

    def create_env(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is not None:
            try:
                self.env.close(clear_cache=False)
            except Exception:  # noqa: BLE001
                pass
            self.env = None
        self.env = LiberoTaskEnv(**kwargs)
        return self._meta()

    def get_meta(self) -> Dict[str, Any]:
        return self._meta()

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is not created")
        obs = self.env.reset(**kwargs)
        return {"obs": obs, "meta": self._meta()}

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

    def step_chunk(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is not created")
        actions = np.asarray(kwargs["actions"], dtype=np.float32)
        result = self.env.step_chunk(actions)
        if not isinstance(result, dict):
            raise RuntimeError("env.step_chunk returned invalid payload")
        payload = dict(result)
        payload["meta"] = self._meta()
        return payload

    def close(self, **kwargs: Any) -> Dict[str, Any]:
        if self.env is not None:
            self.env.close(clear_cache=bool(kwargs.get("clear_cache", False)))
            self.env = None
        return {"closed": True}


STATE = _EnvState()
Handler: type[BaseHTTPRequestHandler] = make_pickle_rpc_handler(
    STATE,
    LOGGER,
    keep_alive=True,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )

    # Keep request handling single-threaded to avoid MuJoCo / OpenGL context issues.
    # The trainer is expected to reuse one persistent connection to reduce per-step RPC churn.
    server = HTTPServer((args.host, args.port), Handler)
    LOGGER.info("LIBERO env server started at http://%s:%s", args.host, args.port)
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
