#!/usr/bin/env python3
"""SmolVLA 部署脚本骨架。

默认只跑 dry-run，不写动作。传入 --send-dry-run-actions 也只会更新内存里的
DryRunPiperIO，仍然不会连接 CAN 或控制真实 Piper。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource
from piper_smolvla.config import PiperSmolVLAAdapterConfig
from piper_smolvla.deployment import DeploymentConfig, DeploymentRunner
from piper_smolvla.policy_io import HoldCurrentPolicy, load_lerobot_policy
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run Piper SmolVLA deployment loop.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--task", default="Piper SmolVLA dry-run deployment")
    parser.add_argument("--state", default="0,1,-1,0,0,0,0.01")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default="", help="Optional LeRobot dataset root used for policy metadata.")
    parser.add_argument("--send-dry-run-actions", action="store_true")
    return parser.parse_args()


def parse_state(text: str) -> tuple[float, ...]:
    return validate_state(float(part.strip()) for part in text.split(","))


def main() -> int:
    args = parse_args()
    state = parse_state(args.state)
    image = np.zeros((3, 480, 640), dtype=np.uint8)
    io = DryRunPiperIO(state)
    adapter = PiperSmolVLAAdapter(
        state_source=io,
        image_source=StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image}),
        action_sink=io,
        config=PiperSmolVLAAdapterConfig(allow_action_sink=args.send_dry_run_actions),
    )
    policy = build_policy(args)
    runner = DeploymentRunner(
        adapter=adapter,
        policy=policy,
        config=DeploymentConfig(
            task=args.task,
            max_steps=args.steps,
            send_actions=args.send_dry_run_actions,
        ),
    )

    steps = runner.run()
    print("deployment dry-run ok")
    print(f"steps={len(steps)}")
    print(f"final_state={list(io.read_state())}")
    print(f"sent_actions={sum(step.sent_action is not None for step in steps)}")
    print(f"policy={'checkpoint' if args.checkpoint else 'hold_current'}")
    print("NO HARDWARE ACCESS")
    return 0


def build_policy(args: argparse.Namespace):
    if not args.checkpoint:
        return HoldCurrentPolicy()

    ds_meta = None
    if args.dataset:
        from lerobot.datasets import LeRobotDataset

        root = args.dataset
        name = root.rstrip("/").split("/")[-1] or "dataset"
        ds_meta = LeRobotDataset(f"piper/{name}", root=root, tolerance_s=0.5).meta
    return load_lerobot_policy(args.checkpoint, ds_meta=ds_meta)


if __name__ == "__main__":
    raise SystemExit(main())
