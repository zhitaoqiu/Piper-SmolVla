"""状态/动作校验工具。

本文件检查 7D state/action 的 shape、NaN/Inf、数值类型和限位，确保进入
采集、推理、部署流程的数据都符合项目锁定 schema。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from piper_smolvla.limits import DEFAULT_LIMIT_CONFIG, LimitConfig, validate_limit_config
from piper_smolvla.schema import ACTION_DIM, ARM_JOINT_DIM, GRIPPER_NAME, PIPER_JOINT_ORDER, STATE_DIM


class ValidationError(ValueError):
    """Base class for Piper schema validation failures."""


class ShapeValidationError(ValidationError):
    """Raised when a state/action vector has the wrong shape."""


class FiniteValidationError(ValidationError):
    """Raised when a state/action vector contains NaN or Inf."""


class LimitValidationError(ValidationError):
    """Raised when a state/action vector exceeds configured limits."""


def validate_state(
    state: Iterable[float],
    *,
    config: LimitConfig = DEFAULT_LIMIT_CONFIG,
    check_limits: bool = False,
) -> tuple[float, ...]:
    values = _coerce_vector(state, STATE_DIM, "state")
    if check_limits:
        _validate_limits(values, config, "state")
    return values


def validate_action(
    action: Iterable[float],
    *,
    config: LimitConfig = DEFAULT_LIMIT_CONFIG,
    check_limits: bool = False,
) -> tuple[float, ...]:
    values = _coerce_vector(action, ACTION_DIM, "action")
    if check_limits:
        _validate_limits(values, config, "action")
    return values


def _coerce_vector(values: Iterable[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ShapeValidationError(f"{name} must be a numeric vector, not {type(values).__name__}")

    shape = getattr(values, "shape", None)
    if shape is not None and tuple(shape) != (expected_dim,):
        raise ShapeValidationError(f"{name} must have shape ({expected_dim},), got {tuple(shape)}")

    try:
        values_tuple = tuple(values)
    except TypeError as exc:
        raise ShapeValidationError(f"{name} must be iterable") from exc

    if len(values_tuple) != expected_dim:
        raise ShapeValidationError(f"{name} must have length {expected_dim}, got {len(values_tuple)}")

    coerced: list[float] = []
    for index, value in enumerate(values_tuple):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            joint_name = PIPER_JOINT_ORDER[index]
            raise ValidationError(f"{name}[{index}] ({joint_name}) must be numeric") from exc

        if not math.isfinite(numeric_value):
            joint_name = PIPER_JOINT_ORDER[index]
            raise FiniteValidationError(f"{name}[{index}] ({joint_name}) must be finite")

        coerced.append(numeric_value)

    return tuple(coerced)


def _validate_limits(values: tuple[float, ...], config: LimitConfig, name: str) -> None:
    validate_limit_config(config)

    for index, (value, limit) in enumerate(zip(values[:ARM_JOINT_DIM], config.joint_limits_rad, strict=True)):
        lower = limit.lower_rad - config.tolerance
        upper = limit.upper_rad + config.tolerance
        if value < lower or value > upper:
            raise LimitValidationError(
                f"{name}[{index}] ({limit.name})={value} rad exceeds [{limit.lower_rad}, {limit.upper_rad}] rad"
            )

    gripper_value = values[-1]
    if config.gripper_min_m is not None and gripper_value < config.gripper_min_m - config.tolerance:
        raise LimitValidationError(
            f"{name}[6] ({GRIPPER_NAME})={gripper_value} m is below configured min {config.gripper_min_m} m"
        )
    if config.gripper_max_m is not None and gripper_value > config.gripper_max_m + config.tolerance:
        raise LimitValidationError(
            f"{name}[6] ({GRIPPER_NAME})={gripper_value} m exceeds configured max {config.gripper_max_m} m"
        )
