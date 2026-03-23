from __future__ import annotations

"""HTTP RPC server for LIBERO environments (run inside the `libero` conda env)."""

import argparse
import logging
import pickle
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.env_wrappers.task_env import LiberoTaskEnv

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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _should_keep_alive(self) -> bool:
        return str(self.headers.get("Connection", "")).lower() != "close"

    def _write(self, status: int, payload: Dict[str, Any]) -> None:
        keep_alive = self._should_keep_alive()
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive" if keep_alive else "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = not keep_alive

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/rpc":
            self._write(404, {"ok": False, "error": "not found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            req = pickle.loads(raw)
            method = str(req["method"])
            kwargs = req.get("kwargs", {})
            fn = getattr(STATE, method, None)
            if fn is None:
                raise RuntimeError(f"unknown method: {method}")
            result = fn(**kwargs)
            self._write(200, {"ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("rpc error: %s\n%s", exc, traceback.format_exc())
            self._write(200, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

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
