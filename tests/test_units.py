import math

import pytest

from piper_smolvla import units


def test_joint_rad_sdk_units_round_trip():
    tolerance_rad = math.radians(0.0005) + 1e-12

    for value in [0.0, math.pi / 2, -math.pi / 2, 1.234, -2.09439]:
        sdk_value = units.joint_rad_to_sdk_units(value)
        round_trip = units.joint_sdk_units_to_rad(sdk_value)
        assert round_trip == pytest.approx(value, abs=tolerance_rad)


def test_gripper_m_sdk_units_round_trip():
    for value in [0.0, 0.001, 0.035, 0.05, 0.080123]:
        sdk_value = units.gripper_m_to_sdk_units(value)
        round_trip = units.gripper_sdk_units_to_m(sdk_value)
        assert round_trip == pytest.approx(value, abs=0.5e-6 + 1e-12)


def test_state_and_action_to_sdk_units_preserve_order():
    vector = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)

    joint_units, gripper_units = units.state_to_sdk_units(vector)

    assert joint_units == tuple(units.joint_rad_to_sdk_units(value) for value in vector[:6])
    assert gripper_units == units.gripper_m_to_sdk_units(vector[6])
    assert units.action_to_sdk_units(vector) == (joint_units, gripper_units)


def test_sdk_units_to_state_round_trip():
    vector = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)
    joint_units, gripper_units = units.state_to_sdk_units(vector)
    round_trip = units.sdk_units_to_state(joint_units, gripper_units)

    assert round_trip[:6] == pytest.approx(vector[:6], abs=math.radians(0.0005) + 1e-12)
    assert round_trip[6] == pytest.approx(vector[6], abs=0.5e-6 + 1e-12)


def test_vector_conversion_rejects_wrong_length():
    with pytest.raises(ValueError):
        units.joints_rad_to_sdk_units([0.0] * 5)

    with pytest.raises(ValueError):
        units.state_to_sdk_units([0.0] * 6)

