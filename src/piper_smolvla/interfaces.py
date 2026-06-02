"""外部依赖协议。

本文件只定义 StateSource、ImageSource、ActionSink 这类最小接口，让转接层
可以先脱离真实 Piper SDK 和具体相机库进行测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class StateSource(Protocol):
    def read_state(self) -> Sequence[float]:
        """Return [j1, j2, j3, j4, j5, j6, gripper] in radians/meters."""


class ActionSink(Protocol):
    def write_action(self, action: Sequence[float]) -> Sequence[float]:
        """Write a validated action and return the action that was accepted."""


class ImageSource(Protocol):
    def read_images(self) -> Mapping[str, Any]:
        """Return images keyed by canonical LeRobot image keys."""
