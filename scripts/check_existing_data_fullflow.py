#!/usr/bin/env python3
"""现成 LeRobot 数据到 SmolVLA 标准输入的 fullflow 检查。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from piper_smolvla.dataset_compat import check_lerobot_dataset, standardize_frame
from piper_smolvla.policy_io import HoldCurrentPolicy, prepare_policy_batch, select_policy_action
from piper_smolvla.schema import ACTION_KEY, DEFAULT_TASK_INSTRUCTION, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check existing dual Piper dataset fullflow.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--expect-wrist", action="store_true")
    parser.add_argument("--checkpoint", default="", help="Optional local SmolVLA checkpoint for one-batch forward.")
    parser.add_argument("--try-model-forward", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from lerobot.datasets import LeRobotDataset

    root = Path(args.dataset)
    repo_id = args.repo_id or f"piper/{root.name}"
    dataset = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)
    result = check_lerobot_dataset(dataset, name=str(root), expected_episodes=64 if "64" in root.name else None)
    if args.expect_wrist and any("wrist_rgb" in error for error in result.errors):
        pass
    if not result.ok:
        for error in result.errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    frame = dict(dataset[0])
    frame.setdefault("task", DEFAULT_TASK_INSTRUCTION)
    standard = standardize_frame(frame, task_fallback=DEFAULT_TASK_INSTRUCTION)
    batch = prepare_policy_batch(
        {
            STATE_KEY: standard.state,
            ACTION_KEY: standard.action,
            GLOBAL_IMAGE_KEY: standard.global_rgb,
            WRIST_IMAGE_KEY: standard.wrist_rgb,
            "task": standard.task,
        }
    )
    action = select_policy_action(HoldCurrentPolicy(), batch)

    print("dataset_fullflow_ok=True")
    print(f"episodes={result.total_episodes}")
    print(f"frames={result.total_frames}")
    print(f"state_dim={len(standard.state)}")
    print(f"action_dim={len(standard.action)}")
    print(f"batch_state_shape={tuple(batch[STATE_KEY].shape)}")
    print(f"batch_global_shape={tuple(batch[GLOBAL_IMAGE_KEY].shape)}")
    print(f"batch_wrist_shape={tuple(batch[WRIST_IMAGE_KEY].shape)}")
    print(f"predicted_action_dim={len(action)}")
    print(f"tasks={sorted(result.tasks) if result.tasks else [standard.task]}")

    maybe_forward(args, batch)
    return 0


def maybe_forward(args: argparse.Namespace, batch: dict) -> None:
    if not args.try_model_forward and not args.checkpoint:
        print("smolvla_forward=skipped (no --checkpoint and no --try-model-forward; no model download attempted)")
        return
    if not args.checkpoint:
        print("smolvla_forward=skipped (provide a local --checkpoint; no model download attempted)")
        return
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"smolvla_forward=skipped (checkpoint not found: {checkpoint})")
        return
    try:
        from lerobot.policies.factory import make_policy
    except Exception as exc:  # noqa: BLE001
        print(f"smolvla_forward=skipped (missing LeRobot policy factory: {exc})")
        return
    try:
        policy = make_policy(pretrained_path=str(checkpoint))
        action = select_policy_action(policy, batch)
        print(f"smolvla_forward=ok action_dim={len(action)}")
    except Exception as exc:  # noqa: BLE001
        print(f"smolvla_forward=failed ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
