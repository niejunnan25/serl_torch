import glob
import os
from typing import Optional

import torch


def _checkpoint_path(checkpoint_dir: str, step: int):
    return os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")


def _extract_step(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return -1


def latest_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not paths:
        return None
    return max(_extract_step(p) for p in paths)


def save_agent_checkpoint(checkpoint_dir: str, agent, step: int, keep: int = 20):
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "params": agent.state.params,
        "target_params": agent.state.target_params,
        "optimizer": {
            name: opt.state_dict() for name, opt in agent.state.optimizers.items()
        },
    }
    torch.save(payload, _checkpoint_path(checkpoint_dir, step))

    if keep is not None and keep > 0:
        paths = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")),
            key=_extract_step,
        )
        for p in paths[:-keep]:
            try:
                os.remove(p)
            except OSError:
                pass


def load_agent_checkpoint(checkpoint_dir: str, agent, step: Optional[int] = None):
    if os.path.isfile(checkpoint_dir):
        path = checkpoint_dir
    else:
        if step is None:
            step = latest_checkpoint_step(checkpoint_dir)
            if step is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        path = _checkpoint_path(checkpoint_dir, int(step))

    payload = torch.load(path, map_location="cpu")
    if "params" in payload:
        agent.state.params = payload["params"]
    if "target_params" in payload:
        agent.state.target_params = payload["target_params"]

    if "optimizer" in payload:
        for name, state_dict in payload["optimizer"].items():
            if name in agent.state.optimizers:
                agent.state.optimizers[name].load_state_dict(state_dict)

    if "step" in payload:
        agent.state.step = int(payload["step"])

    return agent
