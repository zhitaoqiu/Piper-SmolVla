#!/usr/bin/env python3
"""
X-VLA training launcher for Piper LeRobot datasets.

This script launches the official `lerobot-train` CLI.  The XVLA policy is loaded
via `--policy.path=lerobot/xvla-base` so that the pretrained config / weights are
picked up correctly by the LeRobot training stack.

Environment:
  conda activate lerobot_q

Dry run:
  python scripts/train_xvla.py --dataset datasets/<name> --output outputs/<name>

Smoke train:
  python scripts/train_xvla.py --dataset datasets/<name> --output outputs/<name> \
    --start-training --allow-model-download
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


PIPER_TO_XVLA_RENAME_MAP = {
    "observation.images.global_rgb": "observation.images.image",
    "observation.images.wrist_rgb": "observation.images.image2",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X-VLA training launcher")
    parser.add_argument("--dataset", required=True, help="Path to a local LeRobot dataset")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    # --lr is accepted but NOT forwarded to lerobot-train yet;
    # the first smoke run uses the official XVLA default optimizer settings.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained-path", default="lerobot/xvla-base")
    parser.add_argument("--policy-repo-id", default="piper/xvla-piper")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-image-views", type=int, default=3)
    parser.add_argument("--empty-cameras", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--n-action-steps", type=int, default=30)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--skip-dataset-check", action="store_true")
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args(argv)


def validate_dataset(dataset_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        f"piper/{dataset_path.name}",
        root=str(dataset_path),
        tolerance_s=0.5,
    )
    meta = dataset.meta
    print(f"  episodes={meta.total_episodes}")
    print(f"  frames={meta.total_frames}")
    for name, ft in meta.features.items():
        print(f"  feature: {name}  dtype={ft.get('dtype', '?')}  shape={ft.get('shape', '?')}")

    required = (
        "observation.state",
        "action",
        "observation.images.global_rgb",
        "observation.images.wrist_rgb",
    )
    missing = [key for key in required if key not in meta.features]
    if missing:
        raise SystemExit(f"Dataset is missing required Piper features: {missing}")

    state_shape = tuple(meta.features["observation.state"].get("shape", ()))
    action_shape = tuple(meta.features["action"].get("shape", ()))
    if state_shape[-1:] != (7,) or action_shape[-1:] != (7,):
        raise SystemExit(f"Expected 7D Piper state/action, got state={state_shape}, action={action_shape}")

    tasks = getattr(meta, "tasks", None)
    if tasks is None or len(tasks) == 0:
        raise SystemExit("Dataset has no official LeRobot task table.")
    print("  tasks:")
    for task_text, row in tasks.iterrows():
        print(f"    {int(row['task_index'])}: {task_text}")

    episodes = getattr(meta, "episodes", None)
    if episodes is None or len(episodes) == 0:
        raise SystemExit("Dataset has no official LeRobot episode metadata.")

    first_item = dataset[0]
    for key in ("observation.state", "action", *PIPER_TO_XVLA_RENAME_MAP.keys()):
        if key not in first_item:
            raise SystemExit(f"Official LeRobotDataset decoded sample is missing {key}")
        shape = tuple(getattr(first_item[key], "shape", ()))
        print(f"  decoded sample: {key} shape={shape}")
    task = first_item.get("task")
    if not isinstance(task, str) or not task.strip():
        raise SystemExit("Official LeRobotDataset decoded sample is missing task string.")
    print(f"  decoded sample task={task!r}")


def build_command(args: argparse.Namespace) -> list[str]:
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    save_freq = args.save_freq if args.save_freq is not None else args.steps
    rename_map = json.dumps(PIPER_TO_XVLA_RENAME_MAP, separators=(",", ":"))

    return [
        "lerobot-train",
        f"--dataset.repo_id=piper/{dataset_path.name}",
        f"--dataset.root={dataset_path}",
        f"--output_dir={output_path}",
        f"--job_name={output_path.name}",
        "--policy.type=xvla",
        f"--policy.path={args.pretrained_path}",
        f"--policy.repo_id={args.policy_repo_id}",
        f"--policy.dtype={args.dtype}",
        f"--policy.device={args.device}",
        "--policy.action_mode=auto",
        "--policy.max_action_dim=20",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.num_image_views={args.num_image_views}",
        f"--policy.empty_cameras={args.empty_cameras}",
        "--policy.freeze_vision_encoder=false",
        "--policy.freeze_language_encoder=false",
        "--policy.train_policy_transformer=true",
        "--policy.train_soft_prompts=true",
        "--policy.push_to_hub=false",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        f"--log_freq={args.log_freq}",
        "--wandb.enable=false",
        f"--rename_map={rename_map}",
    ]


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def resolve_lerobot_train() -> str:
    env_cli = os.environ.get("LEROBOT_TRAIN_CLI")
    if env_cli:
        return env_cli

    cli = shutil.which("lerobot-train")
    if cli:
        return cli

    server_cli = Path("/home/huatecserver/miniconda3/envs/lerobot_q/bin/lerobot-train")
    if server_cli.exists():
        return str(server_cli)

    raise SystemExit(
        "lerobot-train not found. Activate `conda activate lerobot_q` or set LEROBOT_TRAIN_CLI."
    )


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if not args.allow_model_download:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            env[key] = ""
        env["NO_PROXY"] = "*"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    return env


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if args.start_training and output_path.exists():
        raise SystemExit(
            f"Output directory already exists: {output_path}. "
            "Use a new --output or move the old run first."
        )

    print(f"Dataset:    {dataset_path}")
    print(f"Output:     {output_path}")
    print(f"Steps:      {args.steps}")
    print(f"Batch:      {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Pretrained: {args.pretrained_path}")
    print("Hardware:   NO")

    if not args.skip_dataset_check:
        print("\nDataset features:")
        validate_dataset(dataset_path)
    else:
        print("\nDataset check skipped")

    cmd = build_command(args)
    print("\ntraining_command=" + command_to_shell(cmd))

    if not args.start_training:
        print("\nDRY RUN - pass --start-training to launch")
        return 0

    train_cli = resolve_lerobot_train()
    cmd[0] = train_cli
    print("\nLaunching: " + command_to_shell(cmd))
    return subprocess.run(cmd, check=False, env=build_env(args)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
