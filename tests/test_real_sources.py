import importlib
import sys

import numpy as np
import pytest

from piper_smolvla.real_sources import (
    RealCameraConfig,
    RealCameraSource,
    RealPiperStateConfig,
    RealPiperStateSource,
    _read_rgb,
    is_black_frame,
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


def test_black_frame_detection():
    assert is_black_frame([[[0, 0, 0]]])
    assert not is_black_frame([[[255, 255, 255]]])


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
