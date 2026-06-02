#!/usr/bin/env python3
"""SmolVLA full training 入口。

默认只检查数据和打印训练计划；只有传 --start-training 才会尝试调用本地
LeRobot 训练入口，不会连接真机。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from train_smolvla_utils import run_training_entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or start SmolVLA full training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_training_entrypoint(parse_args(), mode="full"))
