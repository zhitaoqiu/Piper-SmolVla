"""真实硬件数据源。

Piper 状态读取仍放在这里；相机实现已经拆到 ``piper_smolvla.cameras``。
本模块 re-export 旧相机名字，兼容历史脚本和测试。
"""

from __future__ import annotations

from dataclasses import dataclass

from piper_smolvla.cameras import (
    CameraControls,
    RealCameraConfig,
    RealCameraSource,
    RealSenseCamera,
    V4L2Camera,
    _RealSenseCamera,
    _V4L2Camera,
    _configure_sensor_controls,
    _controls_for_role,
    _open_v4l2_capture,
    _read_frame_with_timeout,
    _read_rgb,
    _valid_realsense_serial_or_none,
    is_black_frame,
)
from piper_smolvla.cameras.source import (
    _BAD_EXPLICIT_DEVICES,
    _open_camera,
    _release_camera,
    _resolve_camera_pair,
    _validate_explicit_device,
)
from piper_smolvla.hardware import OfficialPiperSdkBackend, PiperHardwareConfig
from piper_smolvla.validation import validate_state


@dataclass(frozen=True)
class RealPiperStateConfig:
    allow_hardware_readonly: bool = False
    can_port: str = "can0"
    connect_settle_sec: float = 0.3


class RealPiperStateSource:
    def __init__(self, config: RealPiperStateConfig):
        self.config = config
        self._backend: OfficialPiperSdkBackend | None = None

    @property
    def is_connected(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.is_connected

    def connect(self) -> None:
        if not self.config.allow_hardware_readonly:
            raise PermissionError("--allow-hardware-readonly is required before connecting Piper")
        if self._backend is not None:
            return
        self._backend = OfficialPiperSdkBackend(
            PiperHardwareConfig(
                can_port=self.config.can_port,
                enable_on_connect=False,
                disable_on_disconnect=False,
                call_master_slave_config=False,
                connect_settle_sec=self.config.connect_settle_sec,
            )
        )
        self._backend.connect()

    def disconnect(self) -> None:
        if self._backend is not None:
            self._backend.disconnect()
        self._backend = None

    def read_state(self) -> tuple[float, ...]:
        if self._backend is None:
            self.connect()
        return validate_state(self._backend.read_state())


__all__ = [
    "CameraControls",
    "RealCameraConfig",
    "RealCameraSource",
    "RealPiperStateConfig",
    "RealPiperStateSource",
    "RealSenseCamera",
    "V4L2Camera",
    "_BAD_EXPLICIT_DEVICES",
    "_RealSenseCamera",
    "_V4L2Camera",
    "_configure_sensor_controls",
    "_controls_for_role",
    "_open_camera",
    "_open_v4l2_capture",
    "_read_frame_with_timeout",
    "_read_rgb",
    "_release_camera",
    "_resolve_camera_pair",
    "_valid_realsense_serial_or_none",
    "_validate_explicit_device",
    "is_black_frame",
]
