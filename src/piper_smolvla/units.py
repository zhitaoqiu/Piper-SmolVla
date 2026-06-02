"""单位转换工具。

本文件只负责项目单位和 Piper SDK 原始整数单位之间的换算：
关节弧度 <-> SDK 0.001 degree，gripper 米 <-> SDK 0.001 mm。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from piper_smolvla.schema import ACTION_DIM, ARM_JOINT_DIM, STATE_DIM

SDK_JOINT_UNITS_PER_DEGREE = 1000
SDK_GRIPPER_UNITS_PER_METER = 1_000_000


def joint_rad_to_sdk_units(joint_rad: float) -> int:
    """Convert radians to SDK joint units of 0.001 degree."""
    return round(float(joint_rad) * 180.0 / math.pi * SDK_JOINT_UNITS_PER_DEGREE)


def joint_sdk_units_to_rad(joint_sdk_units: int) -> float:
    """Convert SDK joint units of 0.001 degree to radians."""
    return float(joint_sdk_units) / SDK_JOINT_UNITS_PER_DEGREE * math.pi / 180.0


def gripper_m_to_sdk_units(gripper_m: float) -> int:
    """Convert meters to SDK gripper units of 0.001 mm."""
    return round(float(gripper_m) * SDK_GRIPPER_UNITS_PER_METER)


def gripper_sdk_units_to_m(gripper_sdk_units: int) -> float:
    """Convert SDK gripper units of 0.001 mm to meters."""
    return float(gripper_sdk_units) / SDK_GRIPPER_UNITS_PER_METER


def joints_rad_to_sdk_units(joints_rad: Iterable[float]) -> tuple[int, ...]:
    joints = _as_tuple(joints_rad, ARM_JOINT_DIM, "joints_rad")
    return tuple(joint_rad_to_sdk_units(value) for value in joints)


def joints_sdk_units_to_rad(joints_sdk_units: Iterable[int]) -> tuple[float, ...]:
    joints = _as_tuple(joints_sdk_units, ARM_JOINT_DIM, "joints_sdk_units")
    return tuple(joint_sdk_units_to_rad(value) for value in joints)


def state_to_sdk_units(state: Iterable[float]) -> tuple[tuple[int, ...], int]:
    values = _as_tuple(state, STATE_DIM, "state")
    joint_units = joints_rad_to_sdk_units(values[:ARM_JOINT_DIM])
    gripper_units = gripper_m_to_sdk_units(values[-1])
    return joint_units, gripper_units


def sdk_units_to_state(joint_units: Iterable[int], gripper_units: int) -> tuple[float, ...]:
    joints_rad = joints_sdk_units_to_rad(joint_units)
    gripper_m = gripper_sdk_units_to_m(gripper_units)
    return (*joints_rad, gripper_m)


def action_to_sdk_units(action: Iterable[float]) -> tuple[tuple[int, ...], int]:
    values = _as_tuple(action, ACTION_DIM, "action")
    return state_to_sdk_units(values)


def _as_tuple(values: Iterable[float], expected_dim: int, name: str) -> tuple[float, ...]:
    values_tuple = tuple(values)
    if len(values_tuple) != expected_dim:
        raise ValueError(f"{name} must have length {expected_dim}, got {len(values_tuple)}")
    return values_tuple
