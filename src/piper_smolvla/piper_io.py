"""PiperIO 标准输入输出封装。

本文件把 PiperHardwareBackend 和 ImageSource 包成上层统一接口：读取时输出
SmolVLA 标准 observation，写入时只接受项目锁定的 7D action。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from piper_smolvla.adapter import PiperSmolVLAAdapter
from piper_smolvla.config import PiperSmolVLAAdapterConfig
from piper_smolvla.hardware import PiperHardwareBackend
from piper_smolvla.interfaces import ImageSource


@dataclass(frozen=True)
class PiperIOConfig:
    task: str
    allow_action_writes: bool = False
    require_task: bool = True


class PiperIO:
    def __init__(self, *, hardware: PiperHardwareBackend, image_source: ImageSource, config: PiperIOConfig):
        self.hardware = hardware
        self.image_source = image_source
        self.config = config
        self._adapter = PiperSmolVLAAdapter(
            state_source=hardware,
            image_source=image_source,
            action_sink=hardware,
            config=PiperSmolVLAAdapterConfig(
                require_task=config.require_task,
                allow_action_sink=config.allow_action_writes,
            ),
        )

    @property
    def is_connected(self) -> bool:
        return self.hardware.is_connected

    @property
    def is_enabled(self) -> bool:
        return self.hardware.is_enabled

    def connect(self) -> None:
        self.hardware.connect()

    def disconnect(self) -> None:
        self.hardware.disconnect()

    def enable(self, *, blocking: bool = True) -> bool:
        return self.hardware.enable(blocking=blocking)

    def disable(self) -> None:
        self.hardware.disable()

    def get_observation(self, *, task: str | None = None) -> dict[str, Any]:
        return self._adapter.read_observation(task=task or self.config.task)

    def send_action(self, action: Sequence[float]) -> tuple[float, ...]:
        return self._adapter.send_action(action)

    def validate_action(self, action: Sequence[float]) -> tuple[float, ...]:
        return self._adapter.validate_policy_action(action)

