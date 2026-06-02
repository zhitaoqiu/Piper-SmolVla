"""特征字段映射工具。

本文件负责在项目 canonical 7D 向量和 j1.pos...gripper.pos
平铺字段之间互转，同时生成 LeRobot/SmolVLA 需要的 feature spec。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from piper_smolvla.schema import (
    ACTION_KEY,
    GLOBAL_IMAGE_KEY,
    IMAGE_KEYS,
    ACTION_DIM,
    PIPER_JOINT_ORDER,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)
from piper_smolvla.validation import validate_action, validate_state

MOTOR_POS_KEYS: tuple[str, ...] = tuple(f"{name}.pos" for name in PIPER_JOINT_ORDER)
MOTOR_BARE_KEYS: tuple[str, ...] = PIPER_JOINT_ORDER


def vector_to_motor_pos_dict(vector: Sequence[float], *, kind: str = "state") -> dict[str, float]:
    values = validate_action(vector) if kind == "action" else validate_state(vector)
    return {key: float(value) for key, value in zip(MOTOR_POS_KEYS, values, strict=True)}


def motor_pos_dict_to_vector(values: Mapping[str, Any], *, kind: str = "action") -> tuple[float, ...]:
    if all(key in values for key in MOTOR_POS_KEYS):
        vector = tuple(float(values[key]) for key in MOTOR_POS_KEYS)
    elif all(key in values for key in MOTOR_BARE_KEYS):
        vector = tuple(float(values[key]) for key in MOTOR_BARE_KEYS)
    else:
        missing_pos = [key for key in MOTOR_POS_KEYS if key not in values]
        missing_bare = [key for key in MOTOR_BARE_KEYS if key not in values]
        raise KeyError(f"missing motor keys: pos={missing_pos}, bare={missing_bare}")

    return validate_action(vector) if kind == "action" else validate_state(vector)


def build_lerobot_feature_spec(
    *,
    image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    image_shapes = image_shapes or {}
    features: dict[str, dict[str, Any]] = {
        STATE_KEY: {"dtype": "float32", "shape": (STATE_DIM,), "names": list(PIPER_JOINT_ORDER)},
        ACTION_KEY: {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(PIPER_JOINT_ORDER)},
    }
    for key in IMAGE_KEYS:
        features[key] = {"dtype": "video", "shape": image_shapes.get(key)}
    return features


def required_observation_keys() -> tuple[str, ...]:
    return (STATE_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY)
