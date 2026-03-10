"""动态指令生成：根据 episode_info 填充模板，对齐 eval_fast.py。"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np


def generate_instruction_from_episode_info(
    task_name: str,
    episode_info: Dict[str, Any],
    instruction_type: str = "seen",
) -> Optional[str]:
    """
    根据 episode_info（由 ``env.play_once()`` 返回）动态生成 instruction。

    与 ``RoboTwin/scripts/eval_fast.py`` 的逻辑一致：

    1. 从 ``episode_info["info"]`` 中提取占位符参数
    2. 调用 ``generate_episode_descriptions`` 填充模板
    3. 从 seen / unseen 列表中随机选取一条
    """
    try:
        from generate_episode_instructions import generate_episode_descriptions  # type: ignore
    except ImportError:
        logging.getLogger(__name__).warning(
            "generate_episode_instructions not importable — falling back to fixed prompt"
        )
        return None

    if not isinstance(episode_info, dict) or "info" not in episode_info:
        return None

    info_params = episode_info["info"]
    if not isinstance(info_params, dict):
        return None

    try:
        results = generate_episode_descriptions(task_name, [info_params], 1000)
        if not results:
            return None
        candidates = results[0].get(instruction_type, [])
        if not candidates:
            # fallback: 尝试另一种 instruction_type
            alt_type = "seen" if instruction_type == "unseen" else "unseen"
            candidates = results[0].get(alt_type, [])
        if candidates:
            return str(np.random.choice(candidates))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("generate instruction failed: %s", exc)
    return None
