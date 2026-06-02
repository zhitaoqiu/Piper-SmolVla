"""limits 限位配置。

本文件保存官方 Piper 关节限位和项目可配置的 gripper 范围。关节单位是
弧度，gripper 单位是米；gripper 最大值不写死，等待本项目实测确认。
"""

from __future__ import annotations

from dataclasses import dataclass

from piper_smolvla.schema import ARM_JOINT_ORDER


@dataclass(frozen=True)
class JointLimit:
    name: str
    lower_rad: float
    upper_rad: float


JOINT_LIMITS_RAD: tuple[JointLimit, ...] = (
    JointLimit("j1", -2.6179, 2.6179),
    JointLimit("j2", 0.0, 3.14),
    JointLimit("j3", -2.967, 0.0),
    JointLimit("j4", -1.745, 1.745),
    JointLimit("j5", -1.22, 1.22),
    JointLimit("j6", -2.09439, 2.09439),
)


@dataclass(frozen=True)
class LimitConfig:
    joint_limits_rad: tuple[JointLimit, ...] = JOINT_LIMITS_RAD
    gripper_min_m: float | None = 0.0
    gripper_max_m: float | None = None
    tolerance: float = 1e-9


DEFAULT_LIMIT_CONFIG = LimitConfig()


def joint_lower_bounds_rad(config: LimitConfig = DEFAULT_LIMIT_CONFIG) -> tuple[float, ...]:
    return tuple(limit.lower_rad for limit in config.joint_limits_rad)


def joint_upper_bounds_rad(config: LimitConfig = DEFAULT_LIMIT_CONFIG) -> tuple[float, ...]:
    return tuple(limit.upper_rad for limit in config.joint_limits_rad)


def validate_limit_config(config: LimitConfig = DEFAULT_LIMIT_CONFIG) -> None:
    if len(config.joint_limits_rad) != len(ARM_JOINT_ORDER):
        raise ValueError("joint_limits_rad must contain exactly six arm joint limits")

    for expected_name, limit in zip(ARM_JOINT_ORDER, config.joint_limits_rad, strict=True):
        if limit.name != expected_name:
            raise ValueError(f"joint limit order mismatch: expected {expected_name}, got {limit.name}")
        if limit.lower_rad > limit.upper_rad:
            raise ValueError(f"{limit.name} lower limit is greater than upper limit")

    if (
        config.gripper_min_m is not None
        and config.gripper_max_m is not None
        and config.gripper_min_m > config.gripper_max_m
    ):
        raise ValueError("gripper_min_m must be less than or equal to gripper_max_m")
