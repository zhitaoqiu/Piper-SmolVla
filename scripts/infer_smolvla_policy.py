#!/usr/bin/env python3
"""离线 SmolVLA 推理入口。

从现有 dataset 取 episode，运行 checkpoint 或 HoldCurrentPolicy，输出 7D
action CSV 和 gripper 趋势统计。不会连接真机。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

from piper_smolvla.limits import DEFAULT_LIMIT_CONFIG
from piper_smolvla.policy_io import (
    HoldCurrentPolicy,
    load_lerobot_policy,
    prepare_policy_batch,
    select_policy_action_with_options,
)
from piper_smolvla.schema import (
    ACTION_DIM,
    DEFAULT_TASK_INSTRUCTION,
    GLOBAL_IMAGE_KEY,
    PIPER_JOINT_ORDER,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline SmolVLA inference on a dataset episode.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--task", default=DEFAULT_TASK_INSTRUCTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset:
        print("No --dataset supplied; running import-only smoke.")
        print("NO HARDWARE ACCESS")
        return 0

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dataset)
    dataset = LeRobotDataset(f"piper/{root.name}", root=str(root), tolerance_s=0.5)
    policy = load_policy(args.checkpoint, ds_meta=dataset.meta)
    rows = []
    gripper = []
    limit_warnings = 0
    count = 0
    for index in episode_indices(dataset, args.episode):
        if count >= args.max_frames:
            break
        frame = dict(dataset[index])
        frame.setdefault("task", args.task)
        batch = prepare_policy_batch(
            {
                STATE_KEY: frame[STATE_KEY],
                GLOBAL_IMAGE_KEY: frame[GLOBAL_IMAGE_KEY],
                WRIST_IMAGE_KEY: frame[WRIST_IMAGE_KEY],
                "task": frame.get("task", args.task),
            }
        )
        action = select_offline_action(policy, batch)
        limit_warnings += count_limit_warnings(action)
        rows.append({"frame_index": int(frame["frame_index"]), **{f"a{i}": action[i] for i in range(7)}})
        gripper.append(float(action[6]))
        count += 1

    if not rows:
        raise RuntimeError(f"episode {args.episode} produced no rows")
    output = Path(args.output_csv) if args.output_csv else Path("outputs") / "infer" / "smolvla_actions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("offline_inference_ok=True")
    print(f"frames={len(rows)}")
    print(f"output_csv={output}")
    print(f"predicted_action_dim=7")
    print(f"gripper_min={min(gripper):.6f}")
    print(f"gripper_max={max(gripper):.6f}")
    print(f"gripper_first={gripper[0]:.6f}")
    print(f"gripper_last={gripper[-1]:.6f}")
    print(f"gripper_open_close_release_trend={classify_gripper_trend(gripper)}")
    print(f"limit_warnings={limit_warnings}")
    print("NO HARDWARE ACCESS")
    return 0


def load_policy(checkpoint: str, *, ds_meta=None):
    if not checkpoint:
        print("checkpoint=none; using HoldCurrentPolicy")
        return HoldCurrentPolicy()
    try:
        return load_lerobot_policy(checkpoint, ds_meta=ds_meta)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"failed to load LeRobot policy with processors: {exc}") from exc


def episode_indices(dataset, episode: int):
    table = dataset.hf_dataset
    for index in range(len(table)):
        if int(table[index]["episode_index"]) == episode:
            yield index


def select_offline_action(policy: Any, batch: dict[str, Any]) -> tuple[float, ...]:
    return select_policy_action_with_options(policy, batch, validate_limits=False, require_finite=True)


def count_limit_warnings(action: tuple[float, ...]) -> int:
    warnings = 0
    config = DEFAULT_LIMIT_CONFIG
    for index, (value, limit) in enumerate(zip(action[:6], config.joint_limits_rad, strict=True)):
        if value < limit.lower_rad - config.tolerance or value > limit.upper_rad + config.tolerance:
            warnings += 1
            print(
                f"LIMIT_WARNING frame_action[{index}] {PIPER_JOINT_ORDER[index]}="
                f"{value:.6f} outside [{limit.lower_rad}, {limit.upper_rad}]"
            )
    gripper = action[6]
    if config.gripper_min_m is not None and gripper < config.gripper_min_m - config.tolerance:
        warnings += 1
        print(f"LIMIT_WARNING frame_action[6] gripper={gripper:.6f} below {config.gripper_min_m}")
    if config.gripper_max_m is not None and gripper > config.gripper_max_m + config.tolerance:
        warnings += 1
        print(f"LIMIT_WARNING frame_action[6] gripper={gripper:.6f} above {config.gripper_max_m}")
    return warnings


def classify_gripper_trend(values: list[float]) -> str:
    if len(values) < 3:
        return "too_short"
    first = values[0]
    min_value = min(values)
    last = values[-1]
    closed = min_value < first - 0.01
    released = last > min_value + 0.01
    if closed and released:
        return "open_close_release"
    if closed:
        return "open_close"
    return "no_clear_close"


if __name__ == "__main__":
    raise SystemExit(main())
