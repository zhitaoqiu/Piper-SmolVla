"""Common camera types shared by concrete camera drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraControls:
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
    post_red_gain: float | None = None
    post_green_gain: float | None = None
    post_blue_gain: float | None = None
    post_gamma: float | None = None
    post_auto_white_balance: int | None = None


def merge_camera_controls(defaults: CameraControls, overrides: CameraControls) -> CameraControls:
    """Use override values when provided, otherwise keep baked-in defaults."""

    return CameraControls(
        power_line_frequency=(
            overrides.power_line_frequency
            if overrides.power_line_frequency is not None
            else defaults.power_line_frequency
        ),
        exposure_absolute=(
            overrides.exposure_absolute
            if overrides.exposure_absolute is not None
            else defaults.exposure_absolute
        ),
        auto_exposure=overrides.auto_exposure if overrides.auto_exposure is not None else defaults.auto_exposure,
        gain=overrides.gain if overrides.gain is not None else defaults.gain,
        brightness=overrides.brightness if overrides.brightness is not None else defaults.brightness,
        auto_white_balance=(
            overrides.auto_white_balance
            if overrides.auto_white_balance is not None
            else defaults.auto_white_balance
        ),
        white_balance=overrides.white_balance if overrides.white_balance is not None else defaults.white_balance,
        gamma=overrides.gamma if overrides.gamma is not None else defaults.gamma,
        contrast=overrides.contrast if overrides.contrast is not None else defaults.contrast,
        saturation=overrides.saturation if overrides.saturation is not None else defaults.saturation,
        sharpness=overrides.sharpness if overrides.sharpness is not None else defaults.sharpness,
        post_red_gain=overrides.post_red_gain if overrides.post_red_gain is not None else defaults.post_red_gain,
        post_green_gain=(
            overrides.post_green_gain if overrides.post_green_gain is not None else defaults.post_green_gain
        ),
        post_blue_gain=(
            overrides.post_blue_gain if overrides.post_blue_gain is not None else defaults.post_blue_gain
        ),
        post_gamma=overrides.post_gamma if overrides.post_gamma is not None else defaults.post_gamma,
        post_auto_white_balance=(
            overrides.post_auto_white_balance
            if overrides.post_auto_white_balance is not None
            else defaults.post_auto_white_balance
        ),
    )


class CameraReader(Protocol):
    color_order: str
    width: int
    height: int
    fps: float

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return one frame in the driver's native color order."""

    def release(self) -> None:
        """Release the camera handle."""
