"""Camera package for Piper/SmolVLA hardware sources."""

from __future__ import annotations

from piper_smolvla.cameras.config import (
    DEFAULT_BLACK_FRAME_THRESHOLD,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_DATASET_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_ROLLOUT_RATE_HZ,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WRIST_CAMERA,
    camera_defaults_summary,
)
from piper_smolvla.cameras.image import _read_rgb, is_black_frame, read_rgb
from piper_smolvla.cameras.realsense import (
    RealSenseCamera,
    _RealSenseCamera,
    _configure_sensor_controls,
    _valid_realsense_serial_or_none,
    configure_sensor_controls,
    valid_realsense_serial_or_none,
)
from piper_smolvla.cameras.source import RealCameraConfig, RealCameraSource, _controls_for_role
from piper_smolvla.cameras.types import CameraControls, CameraReader
from piper_smolvla.cameras.v4l2 import (
    V4L2Camera,
    _V4L2Camera,
    _open_v4l2_capture,
    _read_frame_with_timeout,
    open_v4l2_capture,
    read_frame_with_timeout,
)

__all__ = [
    "CameraControls",
    "CameraReader",
    "DEFAULT_BLACK_FRAME_THRESHOLD",
    "DEFAULT_CAMERA_FPS",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_DATASET_FPS",
    "DEFAULT_GLOBAL_CAMERA",
    "DEFAULT_ROLLOUT_RATE_HZ",
    "DEFAULT_WARMUP_FRAMES",
    "DEFAULT_WRIST_CAMERA",
    "RealCameraConfig",
    "RealCameraSource",
    "RealSenseCamera",
    "V4L2Camera",
    "camera_defaults_summary",
    "configure_sensor_controls",
    "is_black_frame",
    "open_v4l2_capture",
    "read_frame_with_timeout",
    "read_rgb",
    "valid_realsense_serial_or_none",
    "_RealSenseCamera",
    "_V4L2Camera",
    "_configure_sensor_controls",
    "_controls_for_role",
    "_open_v4l2_capture",
    "_read_frame_with_timeout",
    "_read_rgb",
    "_valid_realsense_serial_or_none",
]
