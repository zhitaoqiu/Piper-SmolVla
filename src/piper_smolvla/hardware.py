"""Piper 硬件后端封装。

本文件通过官方 piper_sdk 连接 Piper 机械臂，提供只读状态、动作写入和
安全启停接口。模块导入不会连接 CAN，也不会发送任何硬件命令。
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from piper_smolvla.limits import DEFAULT_LIMIT_CONFIG, LimitConfig
from piper_smolvla.schema import DEFAULT_CAN_TOPOLOGY_POLICY
from piper_smolvla.units import (
    gripper_m_to_sdk_units,
    gripper_sdk_units_to_m,
    joint_rad_to_sdk_units,
    joint_sdk_units_to_rad,
)
from piper_smolvla.validation import validate_action, validate_state


class PiperHardwareBackend(Protocol):
    @property
    def is_connected(self) -> bool:
        ...

    @property
    def is_enabled(self) -> bool:
        ...

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def enable(self, *, blocking: bool = True) -> bool:
        ...

    def disable(self) -> None:
        ...

    def read_state(self) -> tuple[float, ...]:
        ...

    def write_action(self, action: Sequence[float]) -> tuple[float, ...]:
        ...


@dataclass(frozen=True)
class PiperHardwareConfig:
    can_port: str = "can0"
    gripper_exist: bool = True
    velocity_pct: int = 30
    gripper_effort: int = 1000
    enable_timeout: float = 10.0
    enable_on_connect: bool = False
    disable_on_disconnect: bool = False
    can_topology_policy: str = DEFAULT_CAN_TOPOLOGY_POLICY
    call_master_slave_config: bool = False
    master_slave_config: tuple[int, int, int, int] = (0xFC, 0, 0, 0)
    official_sdk_module: str = "piper_sdk"
    official_sdk_class: str = "C_PiperInterface"
    connect_settle_sec: float = 0.3
    limit_config: LimitConfig = DEFAULT_LIMIT_CONFIG


class OfficialPiperSdkBackend:
    """直接调用官方 piper_sdk 的 Piper 后端。"""

    def __init__(self, config: PiperHardwareConfig):
        self.config = config
        self._piper: Any | None = None
        self._enabled = False

    @property
    def is_connected(self) -> bool:
        return self._piper is not None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def connect(self) -> None:
        if self.is_connected:
            return
        piper_class = self._load_sdk_class()
        piper = self._construct_interface(piper_class)
        piper.ConnectPort()
        time.sleep(max(0.0, self.config.connect_settle_sec))
        if self.config.call_master_slave_config:
            if self.config.can_topology_policy == DEFAULT_CAN_TOPOLOGY_POLICY:
                raise RuntimeError("MasterSlaveConfig requires an explicit non-default CAN topology policy")
            piper.MasterSlaveConfig(*self.config.master_slave_config)
        self._piper = piper
        if self.config.enable_on_connect:
            self.enable(blocking=True)

    def disconnect(self) -> None:
        if self._piper is None:
            return
        if self.config.disable_on_disconnect and self.is_enabled:
            self.disable()
        self._piper = None
        self._enabled = False

    def enable(self, *, blocking: bool = True) -> bool:
        self._require_piper()

        if self.config.gripper_exist:
            self._piper.GripperCtrl(0, 1000, 0x02, 0)
            time.sleep(0.05)
            self._piper.GripperCtrl(0, 1000, 0x01, 0)

        if not blocking:
            self._piper.EnableArm(7)
            self._enabled = True
            return True

        deadline = time.monotonic() + self.config.enable_timeout
        while time.monotonic() < deadline:
            self._piper.EnableArm(7)
            try:
                low = self._piper.GetArmLowSpdInfoMsgs()
                all_on = all(
                    getattr(low, f"motor_{i}").foc_status.driver_enable_status
                    for i in range(1, 7)
                )
                if all_on:
                    self._enabled = True
                    return True
            except Exception:
                pass
            time.sleep(0.2)

        self._enabled = False
        return False

    def disable(self) -> None:
        if self._piper is None:
            return
        if hasattr(self._piper, "DisablePiper"):
            self._piper.DisablePiper()
        else:
            self._piper.DisableArm(7)
        if self.config.gripper_exist:
            self._piper.GripperCtrl(0, 1000, 0x02, 0)
        self._enabled = False

    def is_ok(self) -> bool:
        if self._piper is None:
            return False
        try:
            return bool(self._piper.isOk())
        except Exception:
            return True

    def read_state(self) -> tuple[float, ...]:
        self._assert_ok()
        joint_msg = self._piper.GetArmJointMsgs()
        joint_state = _first_existing_attr(joint_msg, "joint_state", "joint_msgs", default=joint_msg)
        joints = tuple(
            joint_sdk_units_to_rad(_get_number_attr(joint_state, f"joint_{index}"))
            for index in range(1, 7)
        )
        gripper_m = 0.0
        if self.config.gripper_exist:
            gripper_msg = self._piper.GetArmGripperMsgs()
            gripper_state = _first_existing_attr(gripper_msg, "gripper_state", "gripper_msgs", default=gripper_msg)
            gripper_m = gripper_sdk_units_to_m(_get_number_attr(gripper_state, "grippers_angle"))
        return validate_state((*joints, gripper_m), config=self.config.limit_config)

    def write_action(self, action: Sequence[float]) -> tuple[float, ...]:
        self._assert_ok()
        target = validate_action(action, config=self.config.limit_config, check_limits=True)
        _motion_ctrl_joint_mode(self._piper, self.config.velocity_pct)
        raw_joints = tuple(joint_rad_to_sdk_units(value) for value in target[:6])
        self._piper.JointCtrl(*raw_joints)
        if self.config.gripper_exist:
            raw_gripper = max(0, gripper_m_to_sdk_units(target[6]))
            self._piper.GripperCtrl(raw_gripper, self.config.gripper_effort, 0x01, 0)
        return target

    def emergency_stop(self) -> None:
        self._require_piper()
        self._piper.EmergencyStop(0x01)

    def resume_after_emergency_stop(self) -> None:
        self._require_piper()
        self._piper.EmergencyStop(0x02)

    def _assert_ok(self) -> None:
        self._require_piper()
        if not self.is_ok():
            raise IOError(f"CAN heartbeat lost on '{self.config.can_port}'")

    def _load_sdk_class(self) -> type:
        module = importlib.import_module(self.config.official_sdk_module)
        if hasattr(module, self.config.official_sdk_class):
            return getattr(module, self.config.official_sdk_class)
        if hasattr(module, "C_PiperInterface_V2"):
            return getattr(module, "C_PiperInterface_V2")
        raise ImportError(
            f"{self.config.official_sdk_module} has neither {self.config.official_sdk_class} "
            "nor C_PiperInterface_V2"
        )

    def _construct_interface(self, piper_class: type) -> Any:
        attempts = (
            {"can_name": self.config.can_port},
            {"can_name": self.config.can_port, "judge_flag": False},
            {"can_name": self.config.can_port, "can_auto_init": False},
            {},
        )
        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                if kwargs:
                    return piper_class(**kwargs)
                return piper_class(self.config.can_port)
            except TypeError as exc:
                last_exc = exc
        raise TypeError(f"could not construct Piper SDK interface: {last_exc}") from last_exc

    def _require_piper(self) -> None:
        if self._piper is None:
            raise RuntimeError("Piper SDK backend is not connected")


def _motion_ctrl_joint_mode(piper: Any, velocity_pct: int) -> None:
    velocity_pct = max(1, min(int(velocity_pct), 100))
    try:
        piper.MotionCtrl_2(0x01, 0x01, velocity_pct, 0x00)
    except TypeError:
        piper.MotionCtrl_2(0x01, 0x01, velocity_pct)


def _first_existing_attr(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _get_number_attr(value: Any, name: str) -> int | float:
    if not hasattr(value, name):
        raise AttributeError(f"{value!r} has no attribute {name}")
    return getattr(value, name)
