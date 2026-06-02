from dataclasses import replace

import pytest

from piper_smolvla.limits import DEFAULT_LIMIT_CONFIG, JointLimit, LimitConfig, validate_limit_config
from piper_smolvla.validation import (
    FiniteValidationError,
    LimitValidationError,
    ShapeValidationError,
    ValidationError,
    validate_action,
    validate_state,
)

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


def test_valid_state_and_action_pass():
    assert validate_state(VALID_VECTOR) == VALID_VECTOR
    assert validate_action(list(VALID_VECTOR)) == VALID_VECTOR


def test_shape_errors_are_rejected():
    with pytest.raises(ShapeValidationError):
        validate_state([0.0] * 6)

    with pytest.raises(ShapeValidationError):
        validate_action([0.0] * 8)

    with pytest.raises(ShapeValidationError):
        validate_action("0,0,0,0,0,0,0")


def test_shape_attribute_must_be_exact_1d_vector():
    class WrongShape(list):
        shape = (7, 1)

    with pytest.raises(ShapeValidationError):
        validate_state(WrongShape([0.0] * 7))


def test_nan_and_inf_are_rejected():
    nan_vector = list(VALID_VECTOR)
    nan_vector[0] = float("nan")
    with pytest.raises(FiniteValidationError):
        validate_state(nan_vector)

    inf_vector = list(VALID_VECTOR)
    inf_vector[1] = float("inf")
    with pytest.raises(FiniteValidationError):
        validate_action(inf_vector)


def test_non_numeric_values_are_rejected():
    bad_vector = list(VALID_VECTOR)
    bad_vector[2] = object()

    with pytest.raises(ValidationError):
        validate_state(bad_vector)


def test_joint_limit_errors_are_rejected():
    bad_vector = list(VALID_VECTOR)
    bad_vector[0] = 2.7

    with pytest.raises(LimitValidationError):
        validate_action(bad_vector, check_limits=True)


def test_gripper_default_min_is_configurable_and_checked():
    bad_vector = list(VALID_VECTOR)
    bad_vector[6] = -0.001

    with pytest.raises(LimitValidationError):
        validate_action(bad_vector, check_limits=True)

    config_without_gripper_min = replace(DEFAULT_LIMIT_CONFIG, gripper_min_m=None)
    assert validate_action(bad_vector, config=config_without_gripper_min) == tuple(bad_vector)


def test_gripper_max_is_not_default_truth_but_can_be_configured():
    assert DEFAULT_LIMIT_CONFIG.gripper_max_m is None

    config = replace(DEFAULT_LIMIT_CONFIG, gripper_max_m=0.05)
    assert validate_action(VALID_VECTOR, config=config) == VALID_VECTOR

    bad_vector = list(VALID_VECTOR)
    bad_vector[6] = 0.06
    with pytest.raises(LimitValidationError):
        validate_action(bad_vector, config=config, check_limits=True)


def test_limit_check_can_be_disabled_after_shape_and_finite_checks():
    bad_vector = list(VALID_VECTOR)
    bad_vector[0] = 10.0

    assert validate_action(bad_vector, check_limits=False) == tuple(bad_vector)


def test_limit_config_validation_rejects_bad_order_and_ranges():
    bad_order = LimitConfig(
        joint_limits_rad=(
            JointLimit("j2", 0.0, 1.0),
            JointLimit("j1", 0.0, 1.0),
            JointLimit("j3", 0.0, 1.0),
            JointLimit("j4", 0.0, 1.0),
            JointLimit("j5", 0.0, 1.0),
            JointLimit("j6", 0.0, 1.0),
        )
    )
    with pytest.raises(ValueError):
        validate_limit_config(bad_order)

    bad_gripper = replace(DEFAULT_LIMIT_CONFIG, gripper_min_m=0.1, gripper_max_m=0.05)
    with pytest.raises(ValueError):
        validate_limit_config(bad_gripper)

