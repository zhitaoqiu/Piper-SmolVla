"""采集层工具。

本文件负责把 Piper 状态、双相机图像和动作整理成 LeRobot/SmolVLA
标准 frame。这里不直接连接相机或机械臂，只提供可复用的采集流程骨架。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from piper_smolvla.adapter import PiperSmolVLAAdapter
from piper_smolvla.features import build_lerobot_feature_spec
import numpy as np

from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, IMAGE_KEYS, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state


ActionSource = Callable[[Mapping[str, Any]], Sequence[float]]


@dataclass(frozen=True)
class CollectionConfig:
    fps: int = 10
    task: str = "Piper SmolVLA collection"
    image_shape_chw: tuple[int, int, int] = (3, 480, 640)
    image_shapes_chw: Mapping[str, tuple[int, int, int]] | None = None
    image_keys: tuple[str, ...] = IMAGE_KEYS
    use_videos: bool = True


@dataclass
class EpisodeBuffer:
    """内存里的 episode 缓冲区，方便 dry-run 和单元测试。"""

    frames: list[dict[str, Any]] = field(default_factory=list)

    def add_frame(self, observation: Mapping[str, Any], action: Sequence[float]) -> dict[str, Any]:
        frame = make_lerobot_frame(observation, action)
        self.frames.append(frame)
        return frame

    def clear(self) -> None:
        self.frames.clear()

    def __len__(self) -> int:
        return len(self.frames)


def build_collection_features(config: CollectionConfig = CollectionConfig()) -> dict[str, dict[str, Any]]:
    image_shapes = dict(config.image_shapes_chw or {key: config.image_shape_chw for key in config.image_keys})
    return build_lerobot_feature_spec(image_shapes=image_shapes)


def make_lerobot_frame(observation: Mapping[str, Any], action: Sequence[float]) -> dict[str, Any]:
    if STATE_KEY not in observation:
        raise KeyError(f"missing {STATE_KEY}")
    if "task" not in observation or not str(observation["task"]).strip():
        raise ValueError("task is required for collection frames")

    state = vector_to_float32_array(validate_state(observation[STATE_KEY]))
    action_array = vector_to_float32_array(validate_action(action))

    missing_images = [key for key in IMAGE_KEYS if key not in observation]
    if missing_images:
        raise KeyError(f"missing image keys for collection frame: {missing_images}")

    return {
        STATE_KEY: state,
        ACTION_KEY: action_array,
        GLOBAL_IMAGE_KEY: image_to_chw_uint8(observation[GLOBAL_IMAGE_KEY]),
        WRIST_IMAGE_KEY: image_to_chw_uint8(observation[WRIST_IMAGE_KEY]),
        "task": str(observation["task"]),
    }


def make_readonly_transition_frame(
    *,
    previous_state: Sequence[float],
    current_state: Sequence[float],
    previous_images: Mapping[str, Any],
    task: str,
) -> dict[str, Any]:
    """按本项目 read-only mirror demonstration 语义构造 frame。

    observation.state = previous qpos, observation.images = previous images,
    action = current qpos。state 和 images 都来自上一个时间步，
    与 action（当前 qpos）形成 temporal pair，避免 state/image 错位。
    """

    observation = {
        STATE_KEY: validate_state(previous_state),
        GLOBAL_IMAGE_KEY: previous_images[GLOBAL_IMAGE_KEY],
        WRIST_IMAGE_KEY: previous_images[WRIST_IMAGE_KEY],
        "task": task,
    }
    return make_lerobot_frame(observation, validate_action(current_state))


def image_to_chw_uint8(image: Any) -> np.ndarray:
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {arr.shape}")
    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[-1] == 3:
        chw = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"image must have 3 channels, got shape {arr.shape}")
    if np.issubdtype(chw.dtype, np.floating):
        if float(np.nanmax(chw)) <= 1.0:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = chw.astype(np.uint8)
    return np.ascontiguousarray(chw)


def vector_to_float32_array(vector: Sequence[float]) -> np.ndarray:
    return np.asarray(tuple(vector), dtype=np.float32)


def current_state_as_action(observation: Mapping[str, Any]) -> tuple[float, ...]:
    return validate_state(observation[STATE_KEY])


def collect_dry_run_episode(
    adapter: PiperSmolVLAAdapter,
    *,
    num_frames: int,
    task: str,
    action_source: ActionSource = current_state_as_action,
) -> EpisodeBuffer:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    buffer = EpisodeBuffer()
    for _ in range(num_frames):
        observation = adapter.read_observation(task=task)
        action = action_source(observation)
        buffer.add_frame(observation, action)
    return buffer


def create_lerobot_dataset(
    *,
    root: str | Path,
    repo_id: str,
    config: CollectionConfig = CollectionConfig(),
) -> Any:
    """创建 LeRobotDataset。

    这个函数只封装数据集创建，不负责相机/机械臂连接。调用方必须传入已经
    采集并校验过的 frame。
    """

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        features=build_collection_features(config),
        root=Path(root),
        use_videos=config.use_videos,
    )


def write_episode(dataset: Any, frames: Sequence[Mapping[str, Any]]) -> int:
    if not frames:
        raise ValueError("cannot write an empty episode")
    for frame in frames:
        dataset.add_frame(dict(frame))
    dataset.save_episode()
    return len(frames)
