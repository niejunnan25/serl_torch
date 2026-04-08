"""Optional runtime hooks for AgiBot real-robot tasks."""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional


def resolve_hook(spec: Any) -> Optional[Callable[..., Any]]:
    if spec is None:
        return None
    if callable(spec):
        return spec

    raw = str(spec).strip()
    if not raw:
        return None

    module_name: str
    attr_name: str
    if ":" in raw:
        module_name, attr_name = raw.split(":", 1)
    else:
        module_name, _, attr_name = raw.rpartition(".")
        if not module_name:
            raise ValueError(
                f"Hook spec {raw!r} must use 'module:function' or 'module.function'"
            )

    module: ModuleType = importlib.import_module(module_name)
    hook = getattr(module, attr_name, None)
    if hook is None or (not callable(hook)):
        raise AttributeError(f"Resolved hook {raw!r} is not callable")
    return hook


def call_optional_hook(
    hook: Optional[Callable[..., Any]],
    **kwargs: Any,
) -> Any:
    if hook is None:
        return None
    return hook(**kwargs)


def coerce_precheck_result(result: Any) -> tuple[bool, Optional[dict[str, Any]]]:
    if result is None:
        return True, None
    if isinstance(result, bool):
        return bool(result), None
    if isinstance(result, Mapping):
        passed = bool(result.get("passed", True))
        episode_info = result.get("episode_info", None)
        if episode_info is not None and (not isinstance(episode_info, Mapping)):
            episode_info = None
        return passed, (None if episode_info is None else dict(episode_info))
    raise TypeError(f"Unsupported precheck hook result: {type(result)}")


def coerce_success_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "success": False,
            "info": {},
        }
    if isinstance(result, bool):
        return {
            "reward": 1.0 if bool(result) else 0.0,
            "done": bool(result),
            "truncated": False,
            "success": bool(result),
            "info": {},
        }
    if isinstance(result, Mapping):
        success = bool(result.get("success", result.get("done", False)))
        reward = float(result.get("reward", 1.0 if success else 0.0))
        done = bool(result.get("done", success))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", {})
        if not isinstance(info, Mapping):
            info = {"hook_info": info}
        return {
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "success": success,
            "info": dict(info),
        }
    raise TypeError(f"Unsupported success hook result: {type(result)}")

