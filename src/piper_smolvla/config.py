"""转接层配置。

本文件集中定义 Piper + SmolVLA adapter 的安全默认配置。默认不改变 CAN
拓扑、不调用 MasterSlaveConfig，也不允许把动作写入硬件 sink。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from piper_smolvla.limits import DEFAULT_LIMIT_CONFIG, LimitConfig
from piper_smolvla.schema import (
    DEFAULT_CALL_MASTER_SLAVE_CONFIG,
    DEFAULT_CAN_TOPOLOGY_POLICY,
    IMAGE_KEYS,
)


@dataclass(frozen=True)
class ImageFeatureConfig:
    key: str
    shape: tuple[int, int, int] | None = None
    dtype: str = "uint8"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PiperSmolVLAAdapterConfig:
    """High-level adapter defaults.

    Defaults are intentionally conservative: do not change CAN topology and do
    not allow action writes through a sink unless a caller opts in explicitly.
    """

    limit_config: LimitConfig = DEFAULT_LIMIT_CONFIG
    image_keys: tuple[str, ...] = IMAGE_KEYS
    require_task: bool = True
    can_port: str = "can0"
    can_topology_policy: str = DEFAULT_CAN_TOPOLOGY_POLICY
    call_master_slave_config: bool = DEFAULT_CALL_MASTER_SLAVE_CONFIG
    allow_action_sink: bool = False


def validate_adapter_config(config: PiperSmolVLAAdapterConfig) -> None:
    if not config.image_keys:
        raise ValueError("image_keys must not be empty")
    if config.call_master_slave_config and config.can_topology_policy == DEFAULT_CAN_TOPOLOGY_POLICY:
        raise ValueError("call_master_slave_config requires an explicit non-default CAN topology policy")
