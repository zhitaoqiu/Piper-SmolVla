#!/usr/bin/env python3
"""Read-only SmolVLA / X-VLA smoke test for one Piper LeRobot dataset."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.policy_io import prepare_policy_batch
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, IMAGE_KEYS, STATE_KEY, WRIST_IMAGE_KEY


BLUE_TASK = "Pick up the blue object and put it into the box."
GREEN_TASK = "Pick up the green object and put it into the box."
DEFAULT_HF_DATASETS_CACHE = "/tmp/piper_smolvla_hf_cache/datasets"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one LeRobot dataset through current SmolVLA and X-VLA loading paths."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    configure_cache(args.hf_datasets_cache)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = args.repo_id or f"piper/{dataset_root.name}"
    dataset = LeRobotDataset(repo_id, root=str(dataset_root), tolerance_s=0.5)
    print(f"dataset_root={dataset_root}")
    print(f"repo_id={repo_id}")
    print(f"episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.fps}")
    print("hardware_access=NO")
    print("training=NO")
    print("model_download=NO")

    selections = select_blue_and_green(dataset)
    errors: list[str] = []
    decoded_items: dict[str, dict[str, Any]] = {}
    for label, (episode_index, frame_index) in selections.items():
        item = dataset[frame_index]
        decoded_items[label] = item
        print_decoded_item(label, episode_index=episode_index, frame_index=frame_index, item=item)
        errors.extend(validate_raw_item(label, item))

    errors.extend(check_smolvla_path(decoded_items))
    errors.extend(check_xvla_path(dataset_root, decoded_items))

    print("\nsummary:")
    if errors:
        print(f"errors={len(errors)}")
        for error in errors:
            print(f"  ERROR: {error}")
        print("SMOLVLA_XVLA_SMOKE_READY=NO")
        return 1
    print("errors=0")
    print("SMOLVLA_XVLA_SMOKE_READY=YES")
    return 0


def configure_cache(path: str) -> None:
    cache = Path(path).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache)


def select_blue_and_green(dataset: Any) -> dict[str, tuple[int, int]]:
    selected: dict[str, tuple[int, int]] = {}
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        tasks = [str(task) for task in ep.get("tasks", [])]
        if len(set(tasks)) != 1:
            continue
        task = tasks[0]
        frame_index = int(ep["dataset_from_index"])
        if task == BLUE_TASK and "blue" not in selected:
            selected["blue"] = (ep_idx, frame_index)
        if task == GREEN_TASK and "green" not in selected:
            selected["green"] = (ep_idx, frame_index)
        if set(selected) == {"blue", "green"}:
            return selected
    missing = sorted({"blue", "green"} - set(selected))
    raise SystemExit(f"dataset must contain at least one blue and one green episode; missing={missing}")


def print_decoded_item(label: str, *, episode_index: int, frame_index: int, item: dict[str, Any]) -> None:
    print(f"\n{label}_sample:")
    print(f"  episode_index={episode_index} frame_index={frame_index}")
    print(f"  task={item.get('task')!r}")
    print(f"  task_index={to_scalar(item.get('task_index'))}")
    print(f"  image_keys={list(IMAGE_KEYS)}")
    for key in (STATE_KEY, ACTION_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY):
        value = item.get(key)
        print(f"  {key}: shape={shape_of(value)} dtype={dtype_of(value)}")


def validate_raw_item(label: str, item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (STATE_KEY, ACTION_KEY):
        shape = shape_of(item.get(key))
        if shape != (7,):
            errors.append(f"{label}: raw {key} must be 7D, got {shape}")
    for key in IMAGE_KEYS:
        shape = shape_of(item.get(key))
        if len(shape) != 3 or 3 not in (shape[0], shape[-1]):
            errors.append(f"{label}: {key} must be RGB CHW/HWC, got {shape}")
    if "observation.images.image" in item or "observation.images.image2" in item:
        errors.append(f"{label}: raw item already contains X-VLA renamed image keys")
    if not isinstance(item.get("task"), str) or not item.get("task", "").strip():
        errors.append(f"{label}: decoded task string missing")
    return errors


def check_smolvla_path(items: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    print("\nsmolvla_path:")
    for label, item in items.items():
        try:
            batch = prepare_policy_batch(item, normalize_images=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SmolVLA prepare_policy_batch failed for {label}: {type(exc).__name__}: {exc}")
            continue
        print(f"  {label}: batch_task={batch.get('task')!r}")
        print(f"  {label}: {STATE_KEY} shape={shape_of(batch[STATE_KEY])}")
        print(f"  {label}: {GLOBAL_IMAGE_KEY} shape={shape_of(batch[GLOBAL_IMAGE_KEY])}")
        print(f"  {label}: {WRIST_IMAGE_KEY} shape={shape_of(batch[WRIST_IMAGE_KEY])}")
        if shape_of(batch[STATE_KEY]) != (1, 7):
            errors.append(f"SmolVLA {label}: state batch must be (1, 7)")
    return errors


def check_xvla_path(dataset_root: Path, items: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    xvla = import_script("train_xvla", Path(__file__).resolve().parent / "train_xvla.py")
    mapping = getattr(xvla, "PIPER_TO_XVLA_RENAME_MAP", {})
    expected = {
        GLOBAL_IMAGE_KEY: "observation.images.image",
        WRIST_IMAGE_KEY: "observation.images.image2",
    }
    print("\nxvla_path:")
    print(f"  rename_map={json.dumps(mapping, sort_keys=True)}")
    if mapping != expected:
        errors.append(f"X-VLA rename map mismatch: {mapping}")

    args = xvla.parse_args(
        [
            "--dataset",
            str(dataset_root),
            "--output",
            "/tmp/piper_xvla_smoke_output_not_created",
            "--skip-dataset-check",
        ]
    )
    cmd = xvla.build_command(args)
    joined = " ".join(cmd)
    print(f"  command_contains_max_action_dim_20={'--policy.max_action_dim=20' in cmd}")
    print(f"  command_contains_rename_map={'--rename_map=' in joined}")
    if "--policy.max_action_dim=20" not in cmd:
        errors.append("X-VLA command must keep padding in processor via --policy.max_action_dim=20")
    if "--rename_map=" not in joined:
        errors.append("X-VLA command missing --rename_map")

    for label, item in items.items():
        renamed = apply_rename_map(item, mapping)
        print(f"  {label}: raw_action_shape={shape_of(item[ACTION_KEY])} renamed_image_keys_present={all(v in renamed for v in mapping.values())}")
        if shape_of(item[ACTION_KEY]) != (7,):
            errors.append(f"X-VLA {label}: raw action must remain 7D before processor padding")
        for dst in mapping.values():
            if dst not in renamed:
                errors.append(f"X-VLA {label}: renamed batch missing {dst}")
    print("  xvla_padding_location=policy_processor")
    print("  raw_dataset_state_action_dim=7")
    return errors


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_rename_map(item: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        out[mapping.get(key, key)] = value
    return out


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def shape_of(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(dim) for dim in getattr(value, "shape", to_numpy(value).shape))


def dtype_of(value: Any) -> str:
    if value is None:
        return "missing"
    return str(getattr(value, "dtype", to_numpy(value).dtype))


def to_scalar(value: Any) -> Any:
    if value is None:
        return None
    arr = to_numpy(value)
    return arr.reshape(-1)[0].item() if arr.size else None


if __name__ == "__main__":
    raise SystemExit(main())
