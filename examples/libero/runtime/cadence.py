from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Protocol


class CadenceUpdater(Protocol):
    def update(
        self,
        *,
        context: str,
        failure_message: str | None = None,
    ) -> bool: ...


@dataclass(slots=True)
class EnvStepCadenceTracker:
    steps_per_update: int
    log_period: int
    next_update_step: int = field(init=False)
    next_log_step: int = field(init=False)

    def __post_init__(self) -> None:
        steps_per_update = int(self.steps_per_update)
        log_period = int(self.log_period)
        if steps_per_update <= 0:
            raise ValueError(
                "steps_per_update must be positive for cadence tracking"
            )
        if log_period <= 0:
            raise ValueError("log_period must be positive for cadence tracking")
        self.steps_per_update = int(steps_per_update)
        self.log_period = int(log_period)
        self.next_update_step = int(steps_per_update)
        self.next_log_step = int(log_period)

    def advance(
        self,
        *,
        env_steps_after_chunk: int,
        trainer_session: CadenceUpdater,
        update_context_prefix: str,
        failure_message: str | None = None,
    ) -> bool:
        target_env_steps = int(env_steps_after_chunk)
        if target_env_steps < 0:
            raise ValueError("env_steps_after_chunk must be non-negative")

        should_log_timer = False
        while int(self.next_update_step) <= int(target_env_steps):
            context = f"{str(update_context_prefix)}_{int(self.next_update_step)}"
            update_until_success = getattr(
                trainer_session,
                "update_until_success",
                None,
            )
            if callable(update_until_success):
                update_until_success(context=context)
            else:
                trainer_session.update(
                    context=context,
                    failure_message=failure_message,
                )
            self.next_update_step += int(self.steps_per_update)

        while int(self.next_log_step) <= int(target_env_steps):
            should_log_timer = True
            self.next_log_step += int(self.log_period)

        return bool(should_log_timer)


__all__ = ["EnvStepCadenceTracker"]
