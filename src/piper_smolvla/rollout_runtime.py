"""SmolVLA 真机 rollout 的安全限幅、日志和运行工具。

这里集中放置不依赖相机 UI 的部署逻辑：动作限幅、policy runtime reset、
CSV/图片保存等。真实动作发送仍只发生在硬件 backend 的 `write_action`。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.schema import PIPER_JOINT_ORDER
from piper_smolvla.validation import validate_action

# ACT-proven safety constants used by the Piper SmolVLA rollout path.
GRIPPER_OPEN_M = 0.0995
PIPER_GRIPPER_MAX_M = 0.101
WRIST_FREEZE_J2 = 1.45
READY_J2 = 1.65
READY_COUNT_MIN = 5
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008
JOINT_LIMIT_STOP_RAD = 3.0
JOINT_CLAMP_RAD = 3.14
ACTION_SMOOTH_ALPHA = 0.5
WRIST_DELTA_RATIO = 0.4
# J2 guard calibrated against LgRb-blue training data distribution:
#   act_j2 min=-0.785  max=1.979  (close @ j2 ≈ 1.91–1.98)
#   ACTION_J2_MAX is set at 2.0 to cover the training envelope with a
#   0.02 rad margin over the observed maximum (1.9785).
ACTION_J2_MIN = -0.1
ACTION_J2_MAX = 2.0
ACTION_J2_SOFT_MARGIN = 0.15
J2_DELTA_WARN_RAD = 0.05
J2_DELTA_STOP_RAD = 0.10
GRIPPER_CLOSE_ONSET = 0.085
POLICY_ROLLOUT_CONFIRMATION = "ROLLOUT"


@dataclass(frozen=True)
class RolloutSafetyConfig:
    arm_delta_rad: float
    wrist_delta_rad: float
    gripper_delta_m: float
    joint_delta_rad: tuple[float, float, float, float, float, float] | None = None
    action_smooth_alpha: float = ACTION_SMOOTH_ALPHA
    wrist_freeze_enabled: bool = True
    wrist_freeze_j2: float = WRIST_FREEZE_J2
    joint_clamp_rad: float = JOINT_CLAMP_RAD
    gripper_max_m: float = PIPER_GRIPPER_MAX_M


@dataclass(frozen=True)
class HardwareActionGate:
    """Typed-confirm gate used before any real hardware action path starts."""

    allow_hardware_action: bool
    confirmation: str
    expected_confirmation: str
    action_name: str = "hardware action"

    def require(self) -> None:
        if not self.allow_hardware_action:
            raise PermissionError(f"--allow-hardware-action is required before {self.action_name}.")
        if self.confirmation != self.expected_confirmation:
            raise PermissionError(
                f"--confirm-policy-rollout must be the literal string {self.expected_confirmation!r}; "
                f"got {self.confirmation!r}."
            )


def require_policy_rollout_confirmation(*, allow_hardware_action: bool, confirm_policy_rollout: str) -> None:
    HardwareActionGate(
        allow_hardware_action=allow_hardware_action,
        confirmation=confirm_policy_rollout,
        expected_confirmation=POLICY_ROLLOUT_CONFIRMATION,
        action_name="policy rollout",
    ).require()


@dataclass(frozen=True)
class LimitedAction:
    raw_target: np.ndarray
    clamped_action: np.ndarray
    sent_target: np.ndarray
    wrist_frozen: bool
    clamp_joints: list[int]
    grip_clamped: bool


class RolloutActionLimiter:
    """Apply deployment-time delta clamps, wrist freeze, EMA, and final validation."""

    def __init__(self, config: RolloutSafetyConfig):
        self.config = config

    def limit(
        self,
        *,
        current_state: tuple[float, ...] | np.ndarray,
        raw_action: tuple[float, ...] | np.ndarray,
        last_smoothed: np.ndarray | None,
    ) -> LimitedAction:
        state_arr = np.asarray(current_state, dtype=np.float32)
        raw_target = np.asarray(raw_action, dtype=np.float32)

        raw_delta = raw_target - state_arr
        if self.config.joint_delta_rad is not None:
            max_delta = np.asarray(self.config.joint_delta_rad, dtype=np.float32)
        else:
            max_delta = np.array(
                [
                    self.config.arm_delta_rad,
                    self.config.arm_delta_rad,
                    self.config.arm_delta_rad,
                    self.config.wrist_delta_rad,
                    self.config.wrist_delta_rad,
                    self.config.wrist_delta_rad,
                ],
                dtype=np.float32,
            )
        for joint in range(6):
            raw_delta[joint] = np.clip(raw_delta[joint], -max_delta[joint], max_delta[joint])
        clamped_action = state_arr + raw_delta

        grip_delta = np.clip(raw_target[6] - state_arr[6], -self.config.gripper_delta_m, self.config.gripper_delta_m)
        clamped_action[6] = state_arr[6] + grip_delta

        wrist_frozen = False
        if self.config.wrist_freeze_enabled and state_arr[1] > self.config.wrist_freeze_j2:
            clamped_action[3:6] = state_arr[3:6]
            wrist_frozen = True

        alpha = self.config.action_smooth_alpha
        if last_smoothed is not None and alpha > 0:
            smoothed_arm = alpha * clamped_action[:6] + (1.0 - alpha) * last_smoothed[:6]
        else:
            smoothed_arm = clamped_action[:6].copy()
        sent_target = np.concatenate([smoothed_arm, [clamped_action[6]]]).astype(np.float32)

        sent_target[:6] = np.clip(sent_target[:6], -self.config.joint_clamp_rad, self.config.joint_clamp_rad)
        sent_target[6] = np.clip(sent_target[6], 0.0, self.config.gripper_max_m)
        sent_target = np.asarray(validate_action(sent_target, check_limits=True), dtype=np.float32)

        clamp_joints = []
        for joint in range(6):
            orig_delta = raw_target[joint] - state_arr[joint]
            sent_delta = sent_target[joint] - state_arr[joint]
            if abs(orig_delta - sent_delta) > 1e-8:
                clamp_joints.append(joint)
        grip_clamped = abs((raw_target[6] - state_arr[6]) - (sent_target[6] - state_arr[6])) > 1e-8

        return LimitedAction(
            raw_target=raw_target,
            clamped_action=clamped_action,
            sent_target=sent_target,
            wrist_frozen=wrist_frozen,
            clamp_joints=clamp_joints,
            grip_clamped=grip_clamped,
        )


def save_image(array: np.ndarray, path: Path) -> None:
    import cv2

    rgb = array.copy()
    if rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def max_abs_diff(a: tuple[float, ...], b: tuple[float, ...] | None) -> float:
    if b is None:
        return float("inf")
    return float(max(abs(a[index] - b[index]) for index in range(6)))


def fmt_vec(vec: Any, precision: int = 4) -> str:
    return "[" + " ".join(f"{value:.{precision}f}" for value in vec) + "]"


def reset_policy_runtime(policy: Any) -> None:
    """Clear LeRobot action queues/processors before a fresh rollout segment."""

    for obj in (policy, getattr(policy, "policy", None), getattr(policy, "preprocessor", None), getattr(policy, "postprocessor", None)):
        reset = getattr(obj, "reset", None)
        if callable(reset):
            reset()


def write_rollout_csv(csv_path: Path, frame_records: list[dict[str, Any]]) -> None:
    with open(csv_path, "w", newline="") as f:
        header = (
            ["frame", "timestamp"]
            + [f"obs_{name}" for name in PIPER_JOINT_ORDER]
            + [f"raw_{name}" for name in PIPER_JOINT_ORDER]
            + [f"clamped_{name}" for name in PIPER_JOINT_ORDER]
            + [f"sent_{name}" for name in PIPER_JOINT_ORDER]
            + [f"delta_{name}" for name in PIPER_JOINT_ORDER]
            + ["wrist_frozen", "clamp_joints", "grip_clamped", "inf_ms", "cam_ms", "ready_count", "stagnation_count"]
        )
        writer = csv.writer(f)
        writer.writerow(header)
        for record in frame_records:
            writer.writerow(
                [record["frame"], record["timestamp"]]
                + list(record["state"])
                + list(record["raw_action"])
                + list(record["clamped_action"])
                + list(record["sent_action"])
                + list(record["sent_delta"])
                + [
                    1 if record["wrist_frozen"] else 0,
                    ",".join(str(joint) for joint in record["clamp_joints"]) if record["clamp_joints"] else "",
                    1 if record["grip_clamped"] else 0,
                    round(record["inf_ms"], 2),
                    round(record["cam_ms"], 2),
                    record["ready_count"],
                    record["stagnation_count"],
                ]
            )
