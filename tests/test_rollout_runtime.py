import csv

import numpy as np
import pytest

from piper_smolvla.rollout_runtime import (
    RolloutActionLimiter,
    RolloutSafetyConfig,
    require_policy_rollout_confirmation,
    write_rollout_csv,
)
from piper_smolvla.schema import PIPER_JOINT_ORDER


def test_deploy_gripper_timing_gate_can_hold_early_close_open():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from deploy_smolvla import apply_gripper_timing_gate

    action, active, count = apply_gripper_timing_gate(
        frame_idx=10,
        current_state=(0.0, 1.8, -0.8, 0.0, 0.0, 0.0, 0.099),
        raw_action=(0.0, 1.8, -0.8, 0.0, 0.0, 0.0, 0.05),
        raw_close_count=0,
        open_until_frame=20,
        close_confirm_frames=1,
        close_j2_min=None,
    )

    assert active
    assert count == 1
    assert np.isclose(action[6], 0.0995)


def test_deploy_gripper_timing_gate_allows_confirmed_close():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from deploy_smolvla import apply_gripper_timing_gate

    action, active, count = apply_gripper_timing_gate(
        frame_idx=30,
        current_state=(0.0, 1.8, -0.8, 0.0, 0.0, 0.0, 0.099),
        raw_action=(0.0, 1.8, -0.8, 0.0, 0.0, 0.0, 0.05),
        raw_close_count=2,
        open_until_frame=20,
        close_confirm_frames=3,
        close_j2_min=1.7,
    )

    assert not active
    assert count == 3
    assert np.isclose(action[6], 0.05)


def test_rollout_action_limiter_caps_arm_wrist_and_gripper():
    limiter = RolloutActionLimiter(
        RolloutSafetyConfig(
            arm_delta_rad=0.1,
            wrist_delta_rad=0.01,
            gripper_delta_m=0.002,
            action_smooth_alpha=0.0,
        )
    )
    state = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)
    target = (1.0, 2.0, -2.0, 1.0, -1.0, 1.0, 0.08)

    limited = limiter.limit(current_state=state, raw_action=target, last_smoothed=None)

    assert np.allclose(limited.sent_target[:3], (0.1, 1.1, -1.1))
    assert np.allclose(limited.sent_target[3:6], (0.51, -0.51, 0.26))
    assert np.isclose(limited.sent_target[6], 0.032)
    assert limited.clamp_joints == [0, 1, 2, 3, 4, 5]
    assert limited.grip_clamped


def test_rollout_action_limiter_freezes_wrist_when_j2_high():
    limiter = RolloutActionLimiter(
        RolloutSafetyConfig(
            arm_delta_rad=0.1,
            wrist_delta_rad=0.05,
            gripper_delta_m=0.002,
            action_smooth_alpha=0.0,
            wrist_freeze_enabled=True,
            wrist_freeze_j2=1.45,
        )
    )
    state = (0.0, 1.5, -1.0, 0.4, -0.4, 0.2, 0.03)
    target = (0.05, 1.55, -1.05, 1.0, -1.0, 1.0, 0.031)

    limited = limiter.limit(current_state=state, raw_action=target, last_smoothed=None)

    assert limited.wrist_frozen
    assert np.allclose(limited.sent_target[3:6], state[3:6])


def test_rollout_action_limiter_accepts_per_joint_delta_profile():
    limiter = RolloutActionLimiter(
        RolloutSafetyConfig(
            arm_delta_rad=0.01,
            wrist_delta_rad=0.01,
            gripper_delta_m=0.002,
            joint_delta_rad=(0.02, 0.07, 0.03, 0.004, 0.025, 0.006),
            action_smooth_alpha=0.0,
        )
    )
    state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
    target = (1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.05)

    limited = limiter.limit(current_state=state, raw_action=target, last_smoothed=None)

    assert np.allclose(limited.sent_target[:6], (0.02, 0.07, -0.03, 0.004, -0.025, 0.006))
    assert np.isclose(limited.sent_target[6], 0.098)
    assert limited.clamp_joints == [0, 1, 2, 3, 4, 5]
    assert limited.grip_clamped


def test_write_rollout_csv_uses_locked_joint_order(tmp_path):
    csv_path = tmp_path / "rollout.csv"
    vector = tuple(float(index) for index in range(7))
    write_rollout_csv(
        csv_path,
        [
            {
                "frame": 0,
                "timestamp": 1.0,
                "state": vector,
                "raw_action": vector,
                "clamped_action": vector,
                "sent_action": vector,
                "sent_delta": vector,
                "wrist_frozen": False,
                "clamp_joints": [1, 2],
                "grip_clamped": True,
                "inf_ms": 10.123,
                "cam_ms": 3.456,
                "ready_count": 0,
                "stagnation_count": 0,
            }
        ],
    )

    rows = list(csv.reader(csv_path.open()))
    assert rows[0][2:9] == [f"obs_{name}" for name in PIPER_JOINT_ORDER]
    assert rows[1][0] == "0"
    assert rows[1][-6] == "1,2"


def test_policy_rollout_confirmation_gate_blocks_without_double_confirm():
    with pytest.raises(PermissionError):
        require_policy_rollout_confirmation(allow_hardware_action=False, confirm_policy_rollout="ROLLOUT")

    with pytest.raises(PermissionError):
        require_policy_rollout_confirmation(allow_hardware_action=True, confirm_policy_rollout="wrong")


def test_policy_rollout_confirmation_gate_allows_explicit_confirm():
    require_policy_rollout_confirmation(allow_hardware_action=True, confirm_policy_rollout="ROLLOUT")
