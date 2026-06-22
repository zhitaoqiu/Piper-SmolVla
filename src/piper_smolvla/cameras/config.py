"""Shared camera defaults for collection, preview, and deployment."""

from __future__ import annotations

# Current two-RealSense Piper setup used for the 170-episode SmolVLA dataset.
DEFAULT_GLOBAL_CAMERA = "realsense:243222074879"  # Intel RealSense D435I
DEFAULT_WRIST_CAMERA = "realsense:260322275595"  # Intel RealSense D405

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
DEFAULT_DATASET_FPS = 20
DEFAULT_ROLLOUT_RATE_HZ = 20.0

DEFAULT_BLACK_FRAME_THRESHOLD = 5.0
DEFAULT_WARMUP_FRAMES = 5


def camera_defaults_summary() -> dict[str, object]:
    return {
        "global_camera": DEFAULT_GLOBAL_CAMERA,
        "wrist_camera": DEFAULT_WRIST_CAMERA,
        "camera_width": DEFAULT_CAMERA_WIDTH,
        "camera_height": DEFAULT_CAMERA_HEIGHT,
        "camera_fps": DEFAULT_CAMERA_FPS,
        "dataset_fps": DEFAULT_DATASET_FPS,
        "rollout_rate_hz": DEFAULT_ROLLOUT_RATE_HZ,
        "black_frame_threshold": DEFAULT_BLACK_FRAME_THRESHOLD,
        "warmup_frames": DEFAULT_WARMUP_FRAMES,
    }
