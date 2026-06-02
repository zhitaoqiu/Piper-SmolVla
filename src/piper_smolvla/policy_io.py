"""推理输入输出工具。

本文件负责把标准 observation frame 转成策略可用的 batch，并把策略输出
还原成项目锁定的 7D action。这里不加载具体模型，也不执行硬件动作。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np

from piper_smolvla.schema import ACTION_DIM, ACTION_KEY, GLOBAL_IMAGE_KEY, IMAGE_KEYS, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state


def prepare_policy_batch(
    observation: Mapping[str, Any],
    *,
    device: str | None = None,
    normalize_images: bool = True,
) -> dict[str, Any]:
    import torch

    state = torch.as_tensor(validate_state(observation[STATE_KEY]), dtype=torch.float32).unsqueeze(0)
    if device is not None:
        state = state.to(device)

    batch: dict[str, Any] = {STATE_KEY: state}
    for key in IMAGE_KEYS:
        if key not in observation:
            raise KeyError(f"missing image key: {key}")
        image = image_to_chw_float32(observation[key], normalize=normalize_images)
        tensor = torch.from_numpy(image).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        batch[key] = tensor

    if "task" in observation:
        batch["task"] = [str(observation["task"])]
    return batch


def image_to_chw_float32(image: Any, *, normalize: bool = True) -> np.ndarray:
    arr = to_numpy(image)
    if arr.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {arr.shape}")
    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[-1] == 3:
        chw = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"image must have 3 RGB channels, got shape {arr.shape}")

    chw = chw.astype(np.float32, copy=False)
    if normalize and float(np.max(chw)) > 1.0:
        chw = chw / 255.0
    return chw


def extract_action(
    policy_output: Any,
    *,
    validate_limits: bool = True,
    require_finite: bool = True,
) -> tuple[float, ...]:
    """从常见 policy 输出格式中提取项目锁定的 7D action。

    `validate_limits=False` 用于离线诊断：允许脚本统计越界 warning，而不是
    在第一帧轻微越界时直接失败。
    """

    if isinstance(policy_output, Mapping):
        for key in (ACTION_KEY, "action", "actions"):
            if key in policy_output:
                return extract_action(
                    policy_output[key],
                    validate_limits=validate_limits,
                    require_finite=require_finite,
                )
        raise KeyError(f"policy output dict missing action key; available keys={list(policy_output)}")

    arr = to_numpy(policy_output)
    if arr.ndim == 0:
        raise ValueError("policy output must contain a 7D action vector")
    while arr.ndim > 1:
        arr = arr[0]
    if arr.shape[0] < ACTION_DIM:
        raise ValueError(f"policy output must contain at least {ACTION_DIM} values, got {arr.shape}")
    if arr.shape[0] > ACTION_DIM:
        arr = arr[:ACTION_DIM]
    values = tuple(float(value) for value in arr.tolist())
    if require_finite:
        bad = [value for value in values if not math.isfinite(value)]
        if bad:
            raise ValueError(f"policy output contains NaN/Inf values: {bad}")
    if validate_limits:
        return validate_action(values)
    return values


def select_policy_action(policy: Any, batch: Mapping[str, Any]) -> tuple[float, ...]:
    return select_policy_action_with_options(policy, batch)


def select_policy_action_with_options(
    policy: Any,
    batch: Mapping[str, Any],
    *,
    validate_limits: bool = True,
    require_finite: bool = True,
) -> tuple[float, ...]:
    if hasattr(policy, "select_action"):
        output = policy.select_action(dict(batch))
    elif hasattr(policy, "predict_action_chunk"):
        output = policy.predict_action_chunk(dict(batch))
    elif callable(policy):
        output = policy(dict(batch))
    else:
        raise TypeError("policy must be callable or expose select_action/predict_action_chunk")
    return extract_action(output, validate_limits=validate_limits, require_finite=require_finite)


@dataclass
class ProcessedPolicy:
    """LeRobot policy wrapper that always runs preprocessor and postprocessor."""

    policy: Any
    preprocessor: Any
    postprocessor: Any

    def select_action(self, batch: Mapping[str, Any]) -> Any:
        import torch

        with torch.inference_mode():
            processed = self.preprocessor(dict(batch))
            action = self.policy.select_action(processed)
            return self.postprocessor(action)


def load_lerobot_policy(
    checkpoint: str | Path,
    *,
    ds_meta: Any = None,
    policy_type: str = "smolvla",
    device: str | None = None,
) -> ProcessedPolicy:
    """Load a LeRobot policy together with its saved pre/post processors."""

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        import torch
        from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing LeRobot policy factory: {exc}") from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_policy_config(policy_type, pretrained_path=str(path), device=resolved_device, push_to_hub=False)
    policy = make_policy(cfg, ds_meta=ds_meta)
    if hasattr(policy, "eval"):
        policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": resolved_device}},
    )
    return ProcessedPolicy(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)


class HoldCurrentPolicy:
    """测试用策略：把当前 state 原样作为 action 输出。"""

    def __call__(self, batch: Mapping[str, Any]) -> Any:
        return batch[STATE_KEY]


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
