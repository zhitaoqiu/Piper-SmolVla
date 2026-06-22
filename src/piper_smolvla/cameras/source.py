"""Dual-camera source used by Piper/SmolVLA collection and rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.camera_utils import (
    is_explicit,
    is_realsense_device,
    list_video_devices,
    normalize_video_device,
    print_resolved_pair,
    resolve_camera_pair,
)
from piper_smolvla.cameras.config import (
    DEFAULT_BLACK_FRAME_THRESHOLD,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WRIST_CAMERA,
)
from piper_smolvla.cameras.image import read_rgb
from piper_smolvla.cameras.presets import camera_control_defaults
from piper_smolvla.cameras.realsense import RealSenseCamera
from piper_smolvla.cameras.types import CameraControls, merge_camera_controls
from piper_smolvla.cameras.v4l2 import V4L2Camera
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY


@dataclass(frozen=True)
class RealCameraConfig:
    allow_hardware_readonly: bool = False
    global_camera: str = DEFAULT_GLOBAL_CAMERA
    wrist_camera: str = DEFAULT_WRIST_CAMERA
    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    fps: int = DEFAULT_CAMERA_FPS
    black_threshold: float = DEFAULT_BLACK_FRAME_THRESHOLD
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    read_timeout_sec: float = 1.5
    max_consecutive_timeouts: int = 5
    power_line_frequency: int | None = None
    exposure_absolute: int | None = None
    auto_exposure: int | None = None
    gain: float | None = None
    brightness: float | None = None
    auto_white_balance: int | None = None
    white_balance: float | None = None
    gamma: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    sharpness: float | None = None
    wrist_power_line_frequency: int | None = None
    wrist_exposure_absolute: int | None = None
    wrist_auto_exposure: int | None = None
    wrist_gain: float | None = None
    wrist_brightness: float | None = None
    wrist_auto_white_balance: int | None = None
    wrist_white_balance: float | None = None
    wrist_gamma: float | None = None
    wrist_contrast: float | None = None
    wrist_saturation: float | None = None
    wrist_sharpness: float | None = None


class RealCameraSource:
    def __init__(self, config: RealCameraConfig):
        self.config = config
        self._global_cam: Any | None = None
        self._wrist_cam: Any | None = None
        self._resolved_global: str = ""
        self._resolved_wrist: str = ""
        self._assignment_mode: str = "unknown"

    @property
    def resolved_global(self) -> str:
        return self._resolved_global

    @property
    def resolved_wrist(self) -> str:
        return self._resolved_wrist

    @property
    def assignment_mode(self) -> str:
        return self._assignment_mode

    def connect(self) -> None:
        if not self.config.allow_hardware_readonly:
            raise PermissionError("--allow-hardware-readonly is required before opening cameras")
        if self._global_cam is not None and self._wrist_cam is not None:
            return

        self._assignment_mode = (
            "explicit"
            if (is_explicit(self.config.global_camera) and is_explicit(self.config.wrist_camera))
            else "auto"
        )

        global_dev, wrist_dev = _resolve_camera_pair(self.config.global_camera, self.config.wrist_camera)
        _validate_explicit_device(global_dev, self.config)
        _validate_explicit_device(wrist_dev, self.config)

        self._resolved_global = global_dev
        self._resolved_wrist = wrist_dev
        if self._assignment_mode == "auto":
            print_resolved_pair(global_dev, wrist_dev)

        try:
            if self._global_cam is None:
                self._global_cam = _open_camera(global_dev, self.config, role="global")
        except Exception:
            raise

        try:
            if self._wrist_cam is None:
                self._wrist_cam = _open_camera(wrist_dev, self.config, role="wrist")
        except Exception:
            if self._global_cam is not None:
                _release_camera(self._global_cam)
                self._global_cam = None
            raise

    def close(self) -> None:
        for cam_attr in ("_global_cam", "_wrist_cam"):
            cam = getattr(self, cam_attr, None)
            if cam is not None:
                _release_camera(cam)
                setattr(self, cam_attr, None)

    def read_images(self) -> dict[str, np.ndarray]:
        if self._global_cam is None or self._wrist_cam is None:
            self.connect()
        global_rgb = read_rgb(self._global_cam, GLOBAL_IMAGE_KEY, threshold=self.config.black_threshold)
        wrist_rgb = read_rgb(self._wrist_cam, WRIST_IMAGE_KEY, threshold=self.config.black_threshold)
        return {GLOBAL_IMAGE_KEY: global_rgb, WRIST_IMAGE_KEY: wrist_rgb}


_BAD_EXPLICIT_DEVICES: dict[str, str] = {
    "/dev/video2": "known RealSense timeout/bad node",
}


def _validate_explicit_device(device: str, config: RealCameraConfig) -> None:
    if not device.startswith("/dev/video"):
        return
    path = normalize_video_device(device)

    if path in _BAD_EXPLICIT_DEVICES:
        raise RuntimeError(
            f"{path} is a {_BAD_EXPLICIT_DEVICES[path]}. "
            f"Do not use it.\n"
            f"Formal configuration:\n"
            f"  --global-camera {DEFAULT_GLOBAL_CAMERA}\n"
            f"  --wrist-camera {DEFAULT_WRIST_CAMERA}"
        )

    if not Path(path).exists():
        raise FileNotFoundError(f"camera device does not exist: {path}")
    all_video = list_video_devices()
    if path not in all_video:
        raise RuntimeError(f"camera device not in /dev/video*: {path} (found {all_video})")


def _resolve_camera_pair(global_spec: str, wrist_spec: str) -> tuple[str, str]:
    if is_explicit(global_spec) and is_explicit(wrist_spec):
        return normalize_video_device(global_spec), normalize_video_device(wrist_spec)
    return resolve_camera_pair(global_spec, wrist_spec)


def _release_camera(cam: Any) -> None:
    try:
        cam.release()
    except Exception:
        pass


def _open_camera(device: str, config: RealCameraConfig, *, role: str) -> Any:
    controls = merge_camera_controls(
        camera_control_defaults(device, role=role),
        _controls_for_role(config, role),
    )
    if is_realsense_device(device):
        return RealSenseCamera(device, config.width, config.height, config.fps, controls=controls)
    return V4L2Camera(device, config, controls=controls)


def _controls_for_role(config: RealCameraConfig, role: str) -> CameraControls:
    if role != "wrist":
        return CameraControls(
            power_line_frequency=config.power_line_frequency,
            exposure_absolute=config.exposure_absolute,
            auto_exposure=config.auto_exposure,
            gain=config.gain,
            brightness=config.brightness,
            auto_white_balance=config.auto_white_balance,
            white_balance=config.white_balance,
            gamma=config.gamma,
            contrast=config.contrast,
            saturation=config.saturation,
            sharpness=config.sharpness,
        )
    return CameraControls(
        power_line_frequency=(
            config.wrist_power_line_frequency
            if config.wrist_power_line_frequency is not None
            else config.power_line_frequency
        ),
        exposure_absolute=(
            config.wrist_exposure_absolute
            if config.wrist_exposure_absolute is not None
            else config.exposure_absolute
        ),
        auto_exposure=(
            config.wrist_auto_exposure
            if config.wrist_auto_exposure is not None
            else config.auto_exposure
        ),
        gain=config.wrist_gain if config.wrist_gain is not None else config.gain,
        brightness=config.wrist_brightness if config.wrist_brightness is not None else config.brightness,
        auto_white_balance=(
            config.wrist_auto_white_balance
            if config.wrist_auto_white_balance is not None
            else config.auto_white_balance
        ),
        white_balance=(
            config.wrist_white_balance
            if config.wrist_white_balance is not None
            else config.white_balance
        ),
        gamma=config.wrist_gamma if config.wrist_gamma is not None else config.gamma,
        contrast=config.wrist_contrast if config.wrist_contrast is not None else config.contrast,
        saturation=config.wrist_saturation if config.wrist_saturation is not None else config.saturation,
        sharpness=config.wrist_sharpness if config.wrist_sharpness is not None else config.sharpness,
    )
