"""Helpers for external agentlace actor/learner coordination."""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_agentlace_bootstrap_path(
    *,
    run_dir: Path,
    bootstrap_file: Optional[str],
) -> Path:
    raw_path = Path(str(bootstrap_file or "agentlace_bootstrap.pkl")).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (run_dir / raw_path).resolve()


def save_agentlace_bootstrap(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def wait_for_agentlace_bootstrap(
    path: Path,
    *,
    timeout_sec: float,
    poll_sec: float = 0.5,
) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1e-3, float(timeout_sec))
    last_exc: Optional[BaseException] = None
    while time.monotonic() < deadline:
        if not path.exists():
            time.sleep(max(0.05, float(poll_sec)))
            continue
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"agentlace bootstrap at {path} did not contain a dict payload"
                )
            return payload
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(max(0.05, float(poll_sec)))
    if last_exc is not None:
        raise RuntimeError(
            f"Timed out waiting for readable agentlace bootstrap file: {path}"
        ) from last_exc
    raise RuntimeError(f"Timed out waiting for agentlace bootstrap file: {path}")
