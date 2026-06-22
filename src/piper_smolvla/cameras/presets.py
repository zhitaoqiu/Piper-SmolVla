"""Baked-in camera control presets for the current Piper setup."""

from __future__ import annotations

from piper_smolvla.camera_utils import is_realsense_device, video_device_name
from piper_smolvla.cameras.types import CameraControls


WRIST_REALSENSE_DEFAULTS = CameraControls(
    power_line_frequency=1,  # 50 Hz mains in CN lab lighting.
    auto_exposure=1,
    auto_white_balance=1,
)


D405_WRIST_DEFAULTS = CameraControls(
    power_line_frequency=1,
    auto_exposure=1,
    auto_white_balance=1,
    brightness=0.0,
    post_auto_white_balance=1,
    post_gamma=0.95,
)


def camera_control_defaults(device: str, *, role: str) -> CameraControls:
    """Return lab defaults that should not have to be typed on every command."""

    if role != "wrist" or not is_realsense_device(device):
        return CameraControls()
    name = video_device_name(device).lower()
    if "d405" in name:
        return D405_WRIST_DEFAULTS
    return WRIST_REALSENSE_DEFAULTS
