"""部署层安全运行框架。

本文件把 adapter、policy_io 和 action validation 串起来，形成可测试的
部署循环。默认只做 dry-run，不会把动作写到机械臂。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from piper_smolvla.adapter import PiperSmolVLAAdapter
from piper_smolvla.policy_io import prepare_policy_batch, select_policy_action
from piper_smolvla.schema import ARM_JOINT_DIM, STATE_KEY
from piper_smolvla.validation import validate_action


@dataclass(frozen=True)
class ActionLimitConfig:
    max_delta_arm_rad: float | None = 0.03
    max_delta_wrist_rad: float | None = 0.012
    max_delta_gripper_m: float | None = 0.004


@dataclass(frozen=True)
class DeploymentConfig:
    task: str
    max_steps: int = 1
    send_actions: bool = False
    device: str | None = None
    action_limits: ActionLimitConfig = ActionLimitConfig()


@dataclass(frozen=True)
class DeploymentStep:
    step_index: int
    state: tuple[float, ...]
    raw_action: tuple[float, ...]
    limited_action: tuple[float, ...]
    sent_action: tuple[float, ...] | None


class ActionLimiter:
    """按本项目安全参数做逐步限幅，但不改变 schema 和单位。"""

    def __init__(self, config: ActionLimitConfig = ActionLimitConfig()):
        self.config = config

    def limit(self, current: tuple[float, ...], target: tuple[float, ...]) -> tuple[float, ...]:
        current = validate_action(current)
        target = validate_action(target)
        limited = list(target)

        for index in range(ARM_JOINT_DIM):
            if index < 3:
                max_delta = self.config.max_delta_arm_rad
            else:
                max_delta = self.config.max_delta_wrist_rad
            if max_delta is not None:
                limited[index] = _limit_delta(current[index], target[index], max_delta)

        if self.config.max_delta_gripper_m is not None:
            limited[6] = _limit_delta(current[6], target[6], self.config.max_delta_gripper_m)

        return validate_action(limited)


class DeploymentRunner:
    def __init__(self, *, adapter: PiperSmolVLAAdapter, policy: Any, config: DeploymentConfig):
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.adapter = adapter
        self.policy = policy
        self.config = config
        self.limiter = ActionLimiter(config.action_limits)

    def step(self, step_index: int) -> DeploymentStep:
        observation = self.adapter.read_observation(task=self.config.task)
        state = validate_action(observation[STATE_KEY])
        batch = prepare_policy_batch(observation, device=self.config.device)
        raw_action = select_policy_action(self.policy, batch)
        limited_action = self.limiter.limit(state, raw_action)
        sent_action = self.adapter.send_action(limited_action) if self.config.send_actions else None
        return DeploymentStep(
            step_index=step_index,
            state=state,
            raw_action=raw_action,
            limited_action=limited_action,
            sent_action=sent_action,
        )

    def run(self) -> list[DeploymentStep]:
        return [self.step(index) for index in range(self.config.max_steps)]


def _limit_delta(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target
