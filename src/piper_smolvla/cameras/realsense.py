"""Intel RealSense camera driver using pyrealsense2."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from piper_smolvla.camera_utils import (
    realsense_fps_candidates,
    realsense_serial_from_video_device,
    stop_pipeline_quietly,
)
from piper_smolvla.cameras.types import CameraControls


class RealSenseCamera:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        *,
        controls: CameraControls | None = None,
    ):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense wrist camera") from exc

        self._rs = rs
        self._pipe = None
        self.device = device
        self.controls = controls or CameraControls()
        self.serial = valid_realsense_serial_or_none(rs, realsense_serial_from_video_device(device))
        # D405 synthesizes color from stereo pair; its native output is RGB.
        _is_d405 = self.serial is not None and _device_name_by_serial(rs, self.serial).startswith("Intel RealSense D405")
        _formats = (rs.format.rgb8, rs.format.bgr8) if _is_d405 else (rs.format.bgr8, rs.format.rgb8)
        profile = None
        last_error: RuntimeError | None = None
        selected_format = None
        # Retry up to 3 times with backoff — V4L2 probe can leave the device busy.
        for attempt in range(3):
            if attempt > 0:
                time.sleep(0.3 * attempt)
            for rate in realsense_fps_candidates(fps):
                for fmt in _formats:
                    pipe = rs.pipeline()
                    try:
                        cfg = rs.config()
                        if self.serial:
                            cfg.enable_device(self.serial)
                        cfg.enable_stream(rs.stream.color, width, height, fmt, rate)
                        candidate_profile = pipe.start(cfg)
                        if not _wait_for_color_frame(pipe, timeout_sec=2.0):
                            last_error = RuntimeError(
                                f"started RealSense color stream at {width}x{height}@{rate} {fmt}, "
                                "but no color frame arrived"
                            )
                            stop_pipeline_quietly(pipe)
                            continue
                        profile = candidate_profile
                        self._pipe = pipe
                        selected_format = fmt
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        stop_pipeline_quietly(pipe)
                        continue
                if profile is not None:
                    break
            if profile is not None and self._pipe is not None:
                break
        if profile is None or self._pipe is None:
            raise RuntimeError(f"failed to start RealSense color stream at {width}x{height}: {last_error}")

        self.color_order = "rgb" if selected_format == rs.format.rgb8 else "bgr"
        configure_sensor_controls(profile.get_device(), rs, controls=self.controls)
        _warmup_deadline = time.monotonic() + 2.0
        while time.monotonic() < _warmup_deadline:
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=500)
                if frames.get_color_frame():
                    break
            except RuntimeError:
                pass

        s = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = s.width()
        self.height = s.height()
        self.fps = s.fps()
        self._consecutive_failures = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frames = self._pipe.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                self._consecutive_failures += 1
                if self._consecutive_failures <= 3 or self._consecutive_failures % 10 == 0:
                    print(f"  [RealSense] no color frame from {self.device} (consecutive={self._consecutive_failures})")
                return False, None
            self._consecutive_failures = 0
            return True, self._apply_postprocess(np.asanyarray(color.get_data()))
        except RuntimeError as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures <= 3 or self._consecutive_failures % 10 == 0:
                print(f"  [RealSense] read failed from {self.device}: {exc} (consecutive={self._consecutive_failures})")
            return False, None

    def _apply_postprocess(self, frame: np.ndarray) -> np.ndarray:
        gains = (self.controls.post_red_gain, self.controls.post_green_gain, self.controls.post_blue_gain)
        gamma = self.controls.post_gamma
        auto_wb = bool(self.controls.post_auto_white_balance)
        if all(value is None for value in gains) and gamma is None and not auto_wb:
            return frame

        arr = np.asarray(frame, dtype=np.float32)
        if auto_wb:
            arr = _apply_neutral_gray_white_balance(arr, color_order=self.color_order)

        red_gain = 1.0 if self.controls.post_red_gain is None else float(self.controls.post_red_gain)
        green_gain = 1.0 if self.controls.post_green_gain is None else float(self.controls.post_green_gain)
        blue_gain = 1.0 if self.controls.post_blue_gain is None else float(self.controls.post_blue_gain)
        if self.color_order == "bgr":
            arr[..., 0] *= blue_gain
            arr[..., 1] *= green_gain
            arr[..., 2] *= red_gain
        else:
            arr[..., 0] *= red_gain
            arr[..., 1] *= green_gain
            arr[..., 2] *= blue_gain
        arr = np.clip(arr, 0, 255)
        if gamma is not None and float(gamma) > 0:
            arr = ((arr / 255.0) ** float(gamma)) * 255.0
        return np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)

    def release(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None


def configure_sensor_controls(dev: Any, rs: Any, *, controls: CameraControls | None = None) -> None:
    controls = controls or CameraControls()
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
        rgb_sensor = _color_control_sensor(dev)
        power_line = 1 if controls.power_line_frequency is None else controls.power_line_frequency
        _set_supported_option(rgb_sensor, rs, "power_line_frequency", power_line)
        auto_exposure = 1 if controls.auto_exposure is None else controls.auto_exposure
        _set_supported_option(rgb_sensor, rs, "enable_auto_exposure", auto_exposure)
        _set_supported_option(rgb_sensor, rs, "exposure", controls.exposure_absolute)
        _set_supported_option(rgb_sensor, rs, "gain", controls.gain)
        _set_supported_option(rgb_sensor, rs, "brightness", controls.brightness)
        _set_supported_option(rgb_sensor, rs, "enable_auto_white_balance", controls.auto_white_balance)
        _set_supported_option(rgb_sensor, rs, "white_balance", controls.white_balance)
        _set_supported_option(rgb_sensor, rs, "gamma", controls.gamma)
        _set_supported_option(rgb_sensor, rs, "contrast", controls.contrast)
        _set_supported_option(rgb_sensor, rs, "saturation", controls.saturation)
        _set_supported_option(rgb_sensor, rs, "sharpness", controls.sharpness)
    except Exception:
        pass


def _color_control_sensor(dev: Any) -> Any:
    try:
        return dev.first_color_sensor()
    except Exception:
        return dev.first_depth_sensor()


def _set_supported_option(sensor: Any, rs: Any, option_name: str, value: float | int | None) -> bool:
    if value is None:
        return False
    option = getattr(rs.option, option_name, None)
    if option is None:
        return False
    try:
        if not sensor.supports(option):
            return False
        sensor.set_option(option, float(value))
        return True
    except Exception:
        return False


def _apply_neutral_gray_white_balance(arr: np.ndarray, *, color_order: str) -> np.ndarray:
    """Correct color cast using neutral, non-saturated pixels from the scene."""

    if arr.ndim != 3 or arr.shape[-1] != 3:
        return arr
    channels = _rgb_channels(arr, color_order=color_order)
    red, green, blue = channels
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    luminance = (red + green + blue) / 3.0
    chroma = maximum - minimum
    saturation = chroma / np.maximum(maximum, 1.0)

    neutral_mask = (luminance > 35.0) & (luminance < 235.0) & (saturation < 0.28)
    if int(np.count_nonzero(neutral_mask)) < 1000:
        neutral_mask = (luminance > 35.0) & (luminance < 245.0)
    if int(np.count_nonzero(neutral_mask)) < 1000:
        return arr

    means = np.asarray(
        [
            float(red[neutral_mask].mean()),
            float(green[neutral_mask].mean()),
            float(blue[neutral_mask].mean()),
        ],
        dtype=np.float32,
    )
    if np.any(means < 1.0):
        return arr

    target = float(means.mean())
    gains = np.clip(target / means, 0.65, 1.55)
    if color_order == "bgr":
        arr[..., 2] *= gains[0]
        arr[..., 1] *= gains[1]
        arr[..., 0] *= gains[2]
    else:
        arr[..., 0] *= gains[0]
        arr[..., 1] *= gains[1]
        arr[..., 2] *= gains[2]
    return np.clip(arr, 0, 255)


def _rgb_channels(arr: np.ndarray, *, color_order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if color_order == "bgr":
        return arr[..., 2], arr[..., 1], arr[..., 0]
    return arr[..., 0], arr[..., 1], arr[..., 2]


def _wait_for_color_frame(pipe: Any, *, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            frames = pipe.wait_for_frames(timeout_ms=500)
        except RuntimeError:
            continue
        if frames.get_color_frame():
            return True
    return False


def _device_name_by_serial(rs: Any, serial: str) -> str:
    try:
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return str(dev.get_info(rs.camera_info.name)).strip()
    except Exception:
        pass
    return ""


def valid_realsense_serial_or_none(rs: Any, candidate: str | None) -> str | None:
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


_RealSenseCamera = RealSenseCamera
_configure_sensor_controls = configure_sensor_controls
_valid_realsense_serial_or_none = valid_realsense_serial_or_none
