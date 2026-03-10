"""RoboTwin 任务环境封装：统一 reset/step/close 接口。"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from env_wrappers.instruction import generate_instruction_from_episode_info
from env_wrappers.setup import instantiate_task


class RoboTwinTaskEnv:
    """
    RoboTwin 任务环境的封装：根据 task_name 实例化底层环境，提供 reset/step/close 接口。

    支持两种 instruction 模式：

    1. **动态生成**（对齐 eval_fast.py）：利用 expert_precheck 返回的 episode_info 动态填充指令模板。
    2. **固定 prompt 回退**：当 episode_info 不可用或指令生成失败时使用固定 prompt。
    """

    def __init__(
        self,
        task_name: str,
        task_args: Dict[str, Any],
        prompt: str,
        max_setup_retries: int = 5,
        instruction_type: str = "seen",
        logger: Optional[logging.Logger] = None,
    ):
        self.task_name = task_name
        self.task_args = task_args
        self.prompt = prompt
        self.max_setup_retries = max_setup_retries
        self.instruction_type = instruction_type
        self.logger = logger or logging.getLogger(__name__)

        self.env = instantiate_task(task_name)
        self.last_seed: Optional[int] = None
        # 当前 episode 使用的 instruction（reset 后更新）
        self._current_instruction: str = prompt

    @property
    def current_instruction(self) -> str:
        """当前 episode 使用的 instruction（可能是动态生成的，也可能是固定 prompt 回退）。"""
        return self._current_instruction

    @property
    def step_limit(self) -> int:
        """单 episode 最大环境步数（setup_demo 之前或 close_env 之后可能为 None，安全返回 0）。"""
        val = getattr(self.env, "step_lim", None)
        return int(val) if val is not None else 0

    @property
    def take_action_cnt(self) -> int:
        """当前已执行的动作步数（setup_demo 之前或 close_env 之后可能为 None，安全返回 0）。"""
        val = getattr(self.env, "take_action_cnt", None)
        return int(val) if val is not None else 0

    def reset(
        self,
        seed: int,
        episode_id: int,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """按 seed 与 episode_id 重试 setup_demo 和 set_instruction，成功则返回初始观测。

        若提供了 episode_info（来自 expert_precheck），则动态生成 instruction（对齐 eval_fast.py）。
        否则回退为固定 prompt。
        """
        # 1) 动态生成 instruction
        instruction = None
        if episode_info is not None:
            instruction = generate_instruction_from_episode_info(
                task_name=self.task_name,
                episode_info=episode_info,
                instruction_type=self.instruction_type,
            )
        if instruction is None:
            instruction = self.prompt

        self._current_instruction = instruction

        # 2) setup_demo + set_instruction（带重试）
        last_err: Optional[Exception] = None
        for attempt in range(self.max_setup_retries):
            real_seed = seed + attempt
            try:
                self.env.setup_demo(
                    now_ep_num=episode_id,
                    seed=real_seed,
                    is_test=True,
                    **self.task_args,
                )
                self.env.set_instruction(instruction=self._current_instruction)
                self.last_seed = real_seed
                self.logger.info(
                    "Episode instruction: %s",
                    self._current_instruction[:120],
                )
                return self.env.get_obs()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self.logger.warning(
                    "reset failed for seed=%s (attempt %s/%s): %s",
                    real_seed,
                    attempt + 1,
                    self.max_setup_retries,
                    exc,
                )
                try:
                    self.env.close_env(clear_cache=False)
                except Exception:  # noqa: BLE001
                    pass

        raise RuntimeError(f"Failed to setup task env after retries: {last_err}")

    def expert_precheck(self, seed: int, episode_id: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """运行专家可行性检查，跳过 bad seed（对齐 RoboTwin eval_fast.py）。

        Returns:
            ``(passed, episode_info)``
            - passed=True 表示 plan_success 和 check_success 都为 True。
            - episode_info 是 ``play_once`` 返回的 dict（用于动态指令生成）。
        """
        try:
            self.env.setup_demo(
                now_ep_num=episode_id,
                seed=seed,
                is_test=True,
                **self.task_args,
            )
            episode_info = self.env.play_once()
            plan_success = bool(getattr(self.env, "plan_success", False))
            check_success = bool(self.env.check_success())
            passed = bool(plan_success and check_success)
            return passed, episode_info if isinstance(episode_info, dict) else None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("expert precheck failed for seed=%s: %s", seed, exc)
            return False, None
        finally:
            try:
                self.env.close_env(clear_cache=False)
            except Exception:  # noqa: BLE001
                pass

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行一步动作，返回 ``(obs, reward, done, truncated, info)``。"""
        self.env.take_action(np.asarray(action, dtype=np.float32))
        obs = self.env.get_obs()
        success = bool(self.env.eval_success)
        done = bool(success or (self.env.take_action_cnt >= self.env.step_lim))
        reward = 1.0 if success else 0.0
        info = {
            "success": success,
            "take_action_cnt": int(self.env.take_action_cnt),
            "step_lim": int(self.env.step_lim),
        }
        return obs, reward, done, False, info

    def close(self, clear_cache: bool = False) -> None:
        """关闭底层环境。"""
        self.env.close_env(clear_cache=clear_cache)
