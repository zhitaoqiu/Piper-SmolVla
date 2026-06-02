"""数据集兼容性检查。

本文件用于检查 LeRobot 数据集是否能被本项目读成标准 SmolVLA 输入：
7D state、7D action、global_rgb、wrist_rgb 和 task。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from piper_smolvla.schema import (
    ACTION_KEY,
    ACTION_DIM,
    GLOBAL_IMAGE_KEY,
    PIPER_JOINT_ORDER,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)
from piper_smolvla.validation import validate_action, validate_state

REQUIRED_IMAGE_KEYS: tuple[str, ...] = (GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY)


@dataclass(frozen=True)
class StandardInputFrame:
    state: tuple[float, ...]
    action: tuple[float, ...]
    global_rgb: Any
    wrist_rgb: Any
    task: str


@dataclass
class DatasetCompatibilityResult:
    dataset: str
    total_episodes: int | None = None
    total_frames: int | None = None
    checked_state_action_frames: int = 0
    decoded_image_frames: int = 0
    tasks: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def standardize_frame(frame: Mapping[str, Any], *, task_fallback: str | None = None) -> StandardInputFrame:
    missing = [key for key in (STATE_KEY, ACTION_KEY, *REQUIRED_IMAGE_KEYS) if key not in frame]
    if missing:
        raise KeyError(f"frame missing required SmolVLA keys: {missing}")

    task = str(frame.get("task") or task_fallback or "").strip()
    if not task:
        raise ValueError("frame task is required")

    state = validate_state(frame[STATE_KEY])
    action = validate_action(frame[ACTION_KEY])
    _validate_rgb_image(frame[GLOBAL_IMAGE_KEY], GLOBAL_IMAGE_KEY)
    _validate_rgb_image(frame[WRIST_IMAGE_KEY], WRIST_IMAGE_KEY)

    return StandardInputFrame(
        state=state,
        action=action,
        global_rgb=frame[GLOBAL_IMAGE_KEY],
        wrist_rgb=frame[WRIST_IMAGE_KEY],
        task=task,
    )


def check_metadata(meta: Any, *, expected_episodes: int | None = None) -> list[str]:
    errors: list[str] = []
    features = _meta_attr(meta, "features", default={}) or {}
    camera_keys = tuple(_meta_attr(meta, "camera_keys", default=()) or ())
    total_episodes = _meta_attr(meta, "total_episodes", default=None)

    for key, dim in ((STATE_KEY, STATE_DIM), (ACTION_KEY, ACTION_DIM)):
        feature = features.get(key)
        if feature is None:
            errors.append(f"metadata missing feature {key}")
            continue
        shape = _feature_shape(feature)
        if shape != (dim,):
            errors.append(f"{key} shape must be ({dim},), got {shape}")
        names = _feature_names(feature)
        if names is not None and tuple(names) != PIPER_JOINT_ORDER:
            errors.append(f"{key} names must be {PIPER_JOINT_ORDER}, got {tuple(names)}")

    for key in REQUIRED_IMAGE_KEYS:
        if key not in features:
            errors.append(f"metadata missing image feature {key}")
        if camera_keys and key not in camera_keys:
            errors.append(f"metadata camera_keys missing {key}")

    if expected_episodes is not None and total_episodes is not None and int(total_episodes) != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, got {total_episodes}")

    return errors


def check_lerobot_dataset(
    dataset: Any,
    *,
    name: str = "",
    expected_episodes: int | None = None,
    image_frame_indices: Iterable[int] | None = None,
) -> DatasetCompatibilityResult:
    result = DatasetCompatibilityResult(dataset=name or repr(dataset))

    result.total_episodes = _dataset_int_attr(dataset, "num_episodes")
    result.total_frames = _dataset_int_attr(dataset, "num_frames") or len(dataset)
    result.errors.extend(check_metadata(dataset.meta, expected_episodes=expected_episodes))

    table = getattr(dataset, "hf_dataset", None)
    if table is None:
        frame_indices = range(len(dataset))
        for index in frame_indices:
            try:
                standard = standardize_frame(dataset[index])
                result.tasks.add(standard.task)
                result.checked_state_action_frames += 1
                result.decoded_image_frames += 1
            except Exception as exc:  # noqa: BLE001 - collect all compatibility failures
                result.errors.append(f"frame {index}: {type(exc).__name__}: {exc}")
        return result

    for index in range(len(table)):
        row = table[index]
        try:
            validate_state(row[STATE_KEY])
            validate_action(row[ACTION_KEY])
            result.checked_state_action_frames += 1
        except Exception as exc:  # noqa: BLE001 - collect all compatibility failures
            result.errors.append(f"state/action frame {index}: {type(exc).__name__}: {exc}")

    if image_frame_indices is None:
        image_frame_indices = _default_image_indices(len(dataset))

    for index in image_frame_indices:
        try:
            standard = standardize_frame(dataset[int(index)])
            result.tasks.add(standard.task)
            result.decoded_image_frames += 1
        except Exception as exc:  # noqa: BLE001 - collect all compatibility failures
            result.errors.append(f"decoded frame {index}: {type(exc).__name__}: {exc}")

    return result


def _validate_rgb_image(image: Any, key: str) -> None:
    shape = _shape_tuple(image)
    if shape is None:
        raise ValueError(f"{key} image has no shape")
    if len(shape) != 3:
        raise ValueError(f"{key} image must be 3D, got shape {shape}")
    channels_first = shape[0] == 3
    channels_last = shape[-1] == 3
    if not (channels_first or channels_last):
        raise ValueError(f"{key} image must have 3 RGB channels, got shape {shape}")


def _default_image_indices(length: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    return tuple(sorted({0, length // 2, length - 1}))


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _feature_shape(feature: Any) -> tuple[int, ...] | None:
    if isinstance(feature, Mapping):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _feature_names(feature: Any) -> Sequence[str] | None:
    if isinstance(feature, Mapping):
        return feature.get("names")
    return getattr(feature, "names", None)


def _meta_attr(meta: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(name, default)
    return getattr(meta, name, default)


def _dataset_int_attr(dataset: Any, name: str) -> int | None:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    return int(value)


@dataclass
class TaskDistribution:
    """按 episode 统计的 task 分布。"""

    task_counts: dict[str, int] = field(default_factory=dict)
    episode_count: int = 0
    total_frames: int = 0

    def add_episode_task(self, task: str, frame_count: int = 1) -> None:
        self.task_counts[task] = self.task_counts.get(task, 0) + 1
        self.episode_count += 1
        self.total_frames += frame_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_counts": dict(self.task_counts),
            "episode_count": self.episode_count,
            "total_frames": self.total_frames,
        }


def compute_task_distribution(dataset: Any) -> TaskDistribution:
    """按 episode 统计数据集的 task 分布。

    不依赖 check_lerobot_dataset().tasks（那是帧级别的集合，不是 episode 级别）。
    """

    dist = TaskDistribution()
    table = getattr(dataset, "hf_dataset", None)
    num_episodes = _dataset_int_attr(dataset, "num_episodes") or 1

    if table is not None:
        for ep_idx in range(num_episodes):
            try:
                ep_data = table[ep_idx]
                task = _extract_episode_task(ep_data)
                frame_count = len(ep_data[STATE_KEY]) if STATE_KEY in ep_data else 0
                dist.add_episode_task(task, frame_count)
            except Exception:
                dist.add_episode_task("<error>", 0)
    else:
        for ep_idx in range(min(num_episodes, len(dataset))):
            try:
                frame = dataset[ep_idx] if ep_idx < len(dataset) else dataset[0]
                task = str(frame.get("task", "<unknown>"))
                dist.add_episode_task(task, 1)
            except Exception:
                dist.add_episode_task("<error>", 0)

    return dist


def _extract_episode_task(ep_data: Mapping[str, Any]) -> str:
    tasks = ep_data.get("task")
    if tasks is None:
        return "<unknown>"
    if isinstance(tasks, (list, tuple, np.ndarray)):
        values = [str(t) for t in tasks if str(t).strip()]
        return values[0] if values else "<unknown>"
    return str(tasks)
