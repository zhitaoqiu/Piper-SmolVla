import importlib
import sys

import numpy as np
import pytest

from piper_smolvla.cameras import (
    CameraControls,
    DEFAULT_CAMERA_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
    RealSenseCamera,
    _controls_for_role,
    _read_rgb,
    is_black_frame,
)
from piper_smolvla.real_sources import (
    RealCameraConfig as LegacyRealCameraConfig,
    RealCameraSource as LegacyRealCameraSource,
    RealPiperStateConfig,
    RealPiperStateSource,
)


def test_real_piper_requires_explicit_allow_before_import(monkeypatch):
    calls = []

    def fake_import(name, *args, **kwargs):
        calls.append(name)
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    source = RealPiperStateSource(RealPiperStateConfig(allow_hardware_readonly=False))

    with pytest.raises(PermissionError):
        source.connect()

    assert not any("piper" in call.lower() for call in calls)


def test_real_camera_requires_explicit_allow_before_opening():
    source = RealCameraSource(RealCameraConfig(allow_hardware_readonly=False, global_camera="/dev/video6"))

    with pytest.raises(PermissionError):
        source.connect()


def test_real_sources_reexports_camera_api_for_compatibility():
    assert LegacyRealCameraConfig is RealCameraConfig
    assert LegacyRealCameraSource is RealCameraSource


def test_real_camera_config_defaults_match_current_dual_realsense_setup():
    config = RealCameraConfig()

    assert config.global_camera == DEFAULT_GLOBAL_CAMERA == "realsense:243222074879"
    assert config.wrist_camera == DEFAULT_WRIST_CAMERA == "realsense:260322275595"
    assert config.fps == DEFAULT_CAMERA_FPS == 30
    assert config.width == 640
    assert config.height == 480


def test_black_frame_detection():
    assert is_black_frame([[[0, 0, 0]]])
    assert not is_black_frame([[[255, 255, 255]]])


def test_wrist_camera_controls_override_common_controls():
    config = RealCameraConfig(
        power_line_frequency=1,
        exposure_absolute=20,
        auto_exposure=1,
        gain=2.0,
        brightness=3.0,
        wrist_power_line_frequency=2,
        wrist_exposure_absolute=40,
        wrist_auto_exposure=0,
        wrist_gain=5.0,
        wrist_brightness=6.0,
    )

    assert _controls_for_role(config, "global") == CameraControls(
        power_line_frequency=1,
        exposure_absolute=20,
        auto_exposure=1,
        gain=2.0,
        brightness=3.0,
    )
    assert _controls_for_role(config, "wrist") == CameraControls(
        power_line_frequency=2,
        exposure_absolute=40,
        auto_exposure=0,
        gain=5.0,
        brightness=6.0,
    )


def test_wrist_camera_controls_fall_back_to_common_controls():
    config = RealCameraConfig(
        power_line_frequency=1,
        exposure_absolute=20,
        auto_exposure=1,
        gain=2.0,
        brightness=3.0,
    )

    assert _controls_for_role(config, "wrist") == CameraControls(
        power_line_frequency=1,
        exposure_absolute=20,
        auto_exposure=1,
        gain=2.0,
        brightness=3.0,
    )


class FakeCamera:
    def __init__(self, frame, color_order):
        self.frame = frame
        self.color_order = color_order

    def read(self):
        return True, self.frame


def test_read_rgb_respects_camera_color_order():
    bgr_camera = FakeCamera(np.asarray([[[0, 0, 255]]], dtype=np.uint8), "bgr")
    rgb_camera = FakeCamera(np.asarray([[[255, 0, 0]]], dtype=np.uint8), "rgb")

    assert _read_rgb(bgr_camera, "image", threshold=0).tolist() == [[[255, 0, 0]]]
    assert _read_rgb(rgb_camera, "image", threshold=0).tolist() == [[[255, 0, 0]]]


def test_realsense_postprocess_respects_bgr_channel_order():
    cam = object.__new__(RealSenseCamera)
    cam.color_order = "bgr"
    cam.controls = CameraControls(
        post_red_gain=0.5,
        post_green_gain=1.0,
        post_blue_gain=2.0,
    )
    frame = np.asarray([[[10, 20, 100]]], dtype=np.uint8)

    assert cam._apply_postprocess(frame).tolist() == [[[20, 20, 50]]]


def test_realsense_postprocess_auto_white_balances_neutral_pixels():
    cam = object.__new__(RealSenseCamera)
    cam.color_order = "bgr"
    cam.controls = CameraControls(post_auto_white_balance=1)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[..., 0] = 60
    frame[..., 1] = 80
    frame[..., 2] = 120

    corrected = cam._apply_postprocess(frame)
    b, g, r = [float(corrected[..., idx].mean()) for idx in range(3)]

    assert abs(r - g) < 2.0
    assert abs(b - g) < 2.0
