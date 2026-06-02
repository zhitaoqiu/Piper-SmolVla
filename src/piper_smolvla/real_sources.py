"""真实硬件数据源。

提供 Piper 状态读取（通过 piper_sdk）和相机图像采集。
全局 5MP USB Camera 使用 V4L2；腕部 RealSense D435i 使用 pyrealsense2，
避免 RealSense 走 V4L2 时的坏帧/条纹问题。所有硬件操作都需要显式
allow_hardware_readonly=True。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from piper_smolvla.camera_utils import (
    is_explicit,
    is_realsense_device,
    list_video_devices,
    normalize_video_device,
    print_resolved_pair,
    probe_readable_v4l2_devices,
    realsense_fps_candidates,
    realsense_serial_from_video_device,
    resolve_camera_pair,
    stop_pipeline_quietly,
    video_device_group,
    video_device_name,
    video_index,
)
from piper_smolvla.hardware import OfficialPiperSdkBackend, PiperHardwareConfig
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY
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


@dataclass(frozen=True)
class RealCameraConfig:
    allow_hardware_readonly: bool = False
    global_camera: str = "auto"
    wrist_camera: str = "auto"
    width: int = 640
    height: int = 480
    fps: int = 30
    black_threshold: float = 5.0
    warmup_frames: int = 5
    power_line_frequency: int | None = None
    exposure_absolute: int | None = None
    auto_exposure: int | None = None


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

        self._assignment_mode = "explicit" if (is_explicit(self.config.global_camera) and is_explicit(self.config.wrist_camera)) else "auto"

        global_dev, wrist_dev = _resolve_camera_pair(self.config.global_camera, self.config.wrist_camera)
        _validate_explicit_device(global_dev, self.config)
        _validate_explicit_device(wrist_dev, self.config)

        self._resolved_global = global_dev
        self._resolved_wrist = wrist_dev
        if self._assignment_mode == "auto":
            print_resolved_pair(global_dev, wrist_dev)

        # Open global first; if wrist fails, release global.
        try:
            if self._global_cam is None:
                self._global_cam = _open_camera(global_dev, self.config)
        except Exception:
            raise

        try:
            if self._wrist_cam is None:
                self._wrist_cam = _open_camera(wrist_dev, self.config)
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
        global_rgb = _read_rgb(self._global_cam, GLOBAL_IMAGE_KEY, threshold=self.config.black_threshold)
        wrist_rgb = _read_rgb(self._wrist_cam, WRIST_IMAGE_KEY, threshold=self.config.black_threshold)
        return {GLOBAL_IMAGE_KEY: global_rgb, WRIST_IMAGE_KEY: wrist_rgb}


def _validate_explicit_device(device: str, config: RealCameraConfig) -> None:
    if not device.startswith("/dev/video"):
        return
    path = normalize_video_device(device)
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(f"camera device does not exist: {path}")
    all_video = list_video_devices()
    if path not in all_video:
        raise RuntimeError(f"camera device not in /dev/video*: {path} (found {all_video})")


def _resolve_camera_pair(global_spec: str, wrist_spec: str) -> tuple[str, str]:
    if is_explicit(global_spec) and is_explicit(wrist_spec):
        return normalize_video_device(global_spec), normalize_video_device(wrist_spec)
    devices = probe_readable_v4l2_devices() or list_video_devices()
    if not devices:
        raise RuntimeError("no /dev/video* device found; pass explicit --global-camera and --wrist-camera")
    return resolve_camera_pair(global_spec, wrist_spec, devices=devices)


def _release_camera(cam: Any) -> None:
    try:
        cam.release()
    except Exception:
        pass


def _open_camera(device: str, config: RealCameraConfig) -> Any:
    if is_realsense_device(device):
        return _RealSenseCamera(device, config.width, config.height, config.fps)
    return _V4L2Camera(device, config)


class _V4L2Camera:
    color_order = "bgr"

    def __init__(self, device: str, config: RealCameraConfig):
        import cv2

        device = normalize_video_device(device)
        index = video_index(device) if device.startswith("/dev/video") else device
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open V4L2 camera: {device}")

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.fps)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._configure_controls(config)

        for _ in range(max(1, config.warmup_frames)):
            self._cap.read()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.release()
            raise RuntimeError(f"V4L2 camera {device}: failed to read frame after warmup")
        if float(np.asarray(frame, dtype=np.uint8).mean()) < config.black_threshold:
            self.release()
            raise RuntimeError(
                f"V4L2 camera {device}: frame appears black "
                f"(mean={float(np.asarray(frame, dtype=np.uint8).mean()):.1f} < {config.black_threshold})"
            )

    def _configure_controls(self, config: RealCameraConfig) -> None:
        for name, prop_id, value in (
            ("power_line_frequency", getattr(__import__("cv2"), "CAP_PROP_POWERLINE_FREQUENCY", None), config.power_line_frequency),
            ("exposure_time_absolute", getattr(__import__("cv2"), "CAP_PROP_EXPOSURE", None), config.exposure_absolute),
            ("auto_exposure", getattr(__import__("cv2"), "CAP_PROP_AUTO_EXPOSURE", None), config.auto_exposure),
        ):
            if value is None or prop_id is None:
                continue
            try:
                ok = self._cap.set(prop_id, value)
                actual = self._cap.get(prop_id)
                if abs(actual - value) > 0.5 and ok:
                    print(f"  [V4L2] {name}: set {value}, got {actual} (driver may have clamped)")
            except Exception:
                print(f"  [V4L2] {name}: not supported by this device")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._cap.read()

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


class _RealSenseCamera:
    color_order = "bgr"

    def __init__(self, device: str, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense wrist camera") from exc

        self._rs = rs
        self._pipe = None
        self.device = device
        self.serial = _valid_realsense_serial_or_none(rs, realsense_serial_from_video_device(device))
        profile = None
        last_error: RuntimeError | None = None
        for rate in realsense_fps_candidates(fps):
            pipe = rs.pipeline()
            try:
                cfg = rs.config()
                if self.serial:
                    cfg.enable_device(self.serial)
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, rate)
                profile = pipe.start(cfg)
                self._pipe = pipe
                break
            except RuntimeError as exc:
                last_error = exc
                stop_pipeline_quietly(pipe)
                continue
        if profile is None or self._pipe is None:
            raise RuntimeError(f"failed to start RealSense color stream at {width}x{height}: {last_error}")

        self._align = rs.align(rs.stream.color)
        _configure_sensor_controls(profile.get_device(), rs)
        _warmup_deadline = time.monotonic() + 2.0
        while time.monotonic() < _warmup_deadline:
            try:
                self._pipe.wait_for_frames(timeout_ms=500)
            except RuntimeError:
                pass

        s = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = s.width()
        self.height = s.height()
        self.fps = s.fps()

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frames = self._pipe.wait_for_frames(timeout_ms=2000)
            aligned = self._align.process(frames)
            color = aligned.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.get_data())
        except RuntimeError:
            return False, None

    def release(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None


def _configure_sensor_controls(dev: Any, rs: Any) -> None:
    try:
        depth_sensor = dev.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, rng.min)
    except Exception:
        pass

    try:
        rgb_sensor = dev.first_color_sensor()
        if rgb_sensor.supports(rs.option.power_line_frequency):
            rgb_sensor.set_option(rs.option.power_line_frequency, 1)
        if rgb_sensor.supports(rs.option.enable_auto_exposure):
            rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
    except Exception:
        pass


def _valid_realsense_serial_or_none(rs: Any, candidate: str | None) -> str | None:
    if not candidate:
        return None
    try:
        ctx = rs.context()
        for dev in ctx.devices:
            if dev.get_info(rs.camera_info.serial_number) == candidate:
                return candidate
    except Exception:
        pass
    return None


def _read_rgb(camera: Any, key: str, *, threshold: float = 5.0) -> np.ndarray:
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError(f"{key}: failed to read frame")
    if frame.ndim == 2:
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    color_order = getattr(camera, "color_order", "bgr").lower()
    if frame.shape[-1] == 4:
        import cv2
        if color_order == "bgra":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif color_order == "rgba":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            raise RuntimeError(f"{key}: unsupported 4-channel color order {color_order!r}")
    elif frame.shape[-1] == 3:
        if color_order == "bgr":
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_order == "rgb":
            pass
        else:
            raise RuntimeError(f"{key}: unsupported 3-channel color order {color_order!r}")
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.size == 0:
        raise RuntimeError(f"{key}: empty frame")
    if float(arr.mean()) < threshold:
        raise RuntimeError(f"{key}: appears black (mean={arr.mean():.1f} < {threshold})")
    return arr


def is_black_frame(image: Any, *, threshold: float = 5.0) -> bool:
    arr = np.asarray(image)
    if arr.size == 0:
        return True
    return float(arr.mean()) < threshold
