"""Piper + SmolVLA 核心转接器。

本文件把状态源、图像源和动作 sink 组合成标准 SmolVLA observation/action
接口。这里不导入 piper_sdk，不连接 CAN，默认也不允许发送动作。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from piper_smolvla.config import PiperSmolVLAAdapterConfig, validate_adapter_config
from piper_smolvla.interfaces import ActionSink, ImageSource, StateSource
from piper_smolvla.schema import STATE_KEY
from piper_smolvla.validation import validate_action, validate_state


class AdapterError(RuntimeError):
    """Base adapter framework error."""


class MissingImageError(AdapterError):
    """Raised when a required canonical image key is missing."""


class MissingTaskError(AdapterError):
    """Raised when task text is required but missing."""


class ActionSinkDisabledError(AdapterError):
    """Raised when a caller tries to write actions before explicitly enabling a sink."""


@dataclass
class StaticImageSource:
    images: Mapping[str, Any]

    def read_images(self) -> Mapping[str, Any]:
        return dict(self.images)


@dataclass
class DryRunPiperIO:
    """In-memory state/action endpoint for tests and schema dry-runs."""

    state: Sequence[float]

    def read_state(self) -> Sequence[float]:
        return tuple(self.state)

    def write_action(self, action: Sequence[float]) -> Sequence[float]:
        self.state = tuple(action)
        return tuple(self.state)


class PiperSmolVLAAdapter:
    """Canonical schema adapter for Piper observations/actions."""

    def __init__(
        self,
        *,
        state_source: StateSource,
        image_source: ImageSource | None = None,
        action_sink: ActionSink | None = None,
        config: PiperSmolVLAAdapterConfig | None = None,
    ):
        self.config = config or PiperSmolVLAAdapterConfig()
        validate_adapter_config(self.config)
        self._state_source = state_source
        self._image_source = image_source
        self._action_sink = action_sink

    def read_observation(self, *, task: str | None = None) -> dict[str, Any]:
        state = validate_state(self._state_source.read_state(), config=self.config.limit_config)
        images = self._read_required_images()
        observation: dict[str, Any] = {STATE_KEY: state, **images}
        if self.config.require_task:
            if task is None or not task.strip():
                raise MissingTaskError("task is required for SmolVLA observations")
            observation["task"] = task
        elif task is not None:
            observation["task"] = task
        return observation

    def validate_policy_action(self, action: Sequence[float]) -> tuple[float, ...]:
        return validate_action(action, config=self.config.limit_config, check_limits=True)

    def prepare_policy_batch(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if STATE_KEY not in observation:
            raise KeyError(f"missing {STATE_KEY}")
        state = validate_state(observation[STATE_KEY], config=self.config.limit_config)
        batch = dict(observation)
        batch[STATE_KEY] = state
        for key in self.config.image_keys:
            if key not in batch:
                raise MissingImageError(f"missing required image key: {key}")
        if self.config.require_task and not str(batch.get("task", "")).strip():
            raise MissingTaskError("task is required for SmolVLA policy batches")
        return batch

    def send_action(self, action: Sequence[float]) -> tuple[float, ...]:
        validated = self.validate_policy_action(action)
        if not self.config.allow_action_sink:
            raise ActionSinkDisabledError("action sink is disabled by default")
        if self._action_sink is None:
            raise ActionSinkDisabledError("no action sink is configured")
        sent = self._action_sink.write_action(validated)
        return validate_action(sent, config=self.config.limit_config, check_limits=True)

    def _read_required_images(self) -> dict[str, Any]:
        if self._image_source is None:
            raise MissingImageError("no image source is configured")
        images = dict(self._image_source.read_images())
        missing = [key for key in self.config.image_keys if key not in images]
        if missing:
            raise MissingImageError(f"missing required image keys: {missing}")
        return {key: images[key] for key in self.config.image_keys}
