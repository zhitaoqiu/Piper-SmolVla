"""训练脚本共享逻辑。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

from piper_smolvla.dataset_compat import check_lerobot_dataset


def run_training_entrypoint(args: Namespace, *, mode: str) -> int:
    dataset = Path(args.dataset)
    output = Path(args.output)

    if output.exists():
        raise SystemExit(
            f"output directory already exists: {output}. "
            "Remove it or choose a different --output to avoid overwriting existing data."
        )
    if output.resolve() == dataset.resolve():
        raise SystemExit("output must not be the same directory as dataset")

    check_dataset(dataset)
    print(f"training_mode={mode}")
    print(f"dataset={dataset}")
    print(f"output={output}")
    print(f"steps={args.steps}")
    print(f"batch_size={args.batch_size}")
    if hasattr(args, "episodes"):
        print(f"episodes={args.episodes}")

    if not args.start_training:
        print("training_not_started=True (pass --start-training to launch)")
        print("NO HARDWARE ACCESS")
        return 0

    if args.model_id.startswith("lerobot/") and not args.allow_model_download:
        raise SystemExit("--allow-model-download is required before using a remote SmolVLA model id")

    train_cli = shutil.which("lerobot-train")
    if train_cli is None:
        raise SystemExit("missing lerobot-train; install/use a LeRobot checkout with the training CLI")

    cmd = [
        train_cli,
        "--dataset.repo_id",
        f"piper/{dataset.name}",
        "--dataset.root",
        str(dataset),
        "--policy.type",
        "smolvla",
        "--policy.pretrained_path",
        args.model_id,
        "--output_dir",
        str(output),
        "--steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        "0",
        "--policy.device",
        "cuda",
        "--policy.push_to_hub",
        "false",
        "--wandb.enable",
        "false",
        "--save_checkpoint",
        "true",
        "--save_freq",
        str(args.steps),
        "--log_freq",
        "10",
    ]
    if hasattr(args, "episodes"):
        if args.episodes <= 0:
            raise SystemExit("--episodes must be positive")
        episodes = ",".join(str(index) for index in range(args.episodes))
        cmd.extend(["--dataset.episodes", f"[{episodes}]"])
    print("launching_training_command=" + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=False).returncode


def check_dataset(root: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(f"piper/{root.name}", root=str(root), tolerance_s=0.5)
    result = check_lerobot_dataset(dataset, name=str(root))
    if not result.ok:
        for error in result.errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"dataset_ok=True episodes={result.total_episodes} frames={result.total_frames}")
