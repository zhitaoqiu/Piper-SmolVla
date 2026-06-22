#!/usr/bin/env python3
"""Standalone SmolVLA training script for Piper arm datasets.

Self-contained — no piper_smolvla/piper_xvla package imports needed.
Dry-run by default; pass --start-training to launch.

Examples:
  # Dry-run (check dataset, print command, no training)
  python scripts/q_smolvla_train.py \\
      --dataset data/single_cube_line4pos_40_clean \\
      --output outputs/smolvla_single_cube

  # Smoke test
  python scripts/q_smolvla_train.py \\
      --dataset data/single_cube_line4pos_40_clean \\
      --output outputs/smolvla_smoke \\
      --steps 500 --episodes 4 --start-training

  # Full training
  python scripts/q_smolvla_train.py \\
      --dataset data/single_cube_line4pos_40_clean \\
      --output outputs/smolvla_full \\
      --steps 20000 --batch-size 4 --start-training
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — Piper → SmolVLA mapping
# ---------------------------------------------------------------------------

# Dataset camera keys → SmolVLA policy keys
RENAME_MAP = {
    "observation.images.global_rgb": "observation.image",
    "observation.images.wrist_rgb": "observation.image2",
}

# SmolVLA model expects 3 image views; Piper has 2
EMPTY_CAMERAS = 1

# SmolVLA pretrained dims are 6; Piper is 7.  Pad both to 32 so the
# extra dimension is learned during training.
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32

# SmolVLA defaults (matching lerobot/smolvla_base)
CHUNK_SIZE = 50
N_ACTION_STEPS = 50
RESIZE_WITH_PADDING = (512, 512)

# Local HF cache for offline model loading
DEFAULT_CACHE_HOME = os.path.expanduser("~/.cache/piper_smolvla_hf")
DEFAULT_MODEL_ID = "lerobot/smolvla_base"


# ---------------------------------------------------------------------------
# Dataset validation (inline — no piper_smolvla import)
# ---------------------------------------------------------------------------

def check_dataset(root: Path) -> dict[str, Any]:
    """Validate a LeRobot v3 dataset and return summary info."""
    from lerobot.datasets import LeRobotDataset

    repo_id = f"piper/{root.name}"
    ds = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)

    meta = ds.meta
    features = meta.features if hasattr(meta, "features") else meta.get("features", {})

    errors: list[str] = []

    # Required keys
    for key in ("observation.images.global_rgb", "observation.images.wrist_rgb",
                "observation.state", "action"):
        if key not in features:
            errors.append(f"missing feature: {key}")

    # State / action dims
    state_feat = features.get("observation.state", {})
    action_feat = features.get("action", {})
    state_shape = tuple(state_feat.get("shape", [])) if isinstance(state_feat, dict) else getattr(state_feat, "shape", [])
    action_shape = tuple(action_feat.get("shape", [])) if isinstance(action_feat, dict) else getattr(action_feat, "shape", [])

    info = {
        "repo_id": repo_id,
        "total_episodes": int(getattr(meta, "total_episodes", getattr(ds, "num_episodes", 0)) or 0),
        "total_frames": int(getattr(ds, "num_frames", len(ds)) or 0),
        "state_dim": state_shape[0] if len(state_shape) == 1 else len(state_shape),
        "action_dim": action_shape[0] if len(action_shape) == 1 else len(action_shape),
        "errors": errors,
    }

    # Check tasks
    try:
        table = getattr(ds, "hf_dataset", None)
        if table is not None:
            tasks = set()
            for ep_idx in range(info["total_episodes"]):
                ep = table[ep_idx]
                t = ep.get("task")
                if t is not None:
                    tasks.add(str(t[0] if isinstance(t, (list, tuple)) else t))
            info["tasks"] = sorted(tasks)
    except Exception:
        info["tasks"] = []

    return info


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(
    *,
    dataset: Path,
    output: Path,
    model_id: str,
    steps: int,
    batch_size: int,
    save_freq: int,
    log_freq: int,
    episodes: list[int] | None = None,
    max_state_dim: int = MAX_STATE_DIM,
    max_action_dim: int = MAX_ACTION_DIM,
    chunk_size: int = CHUNK_SIZE,
    n_action_steps: int = N_ACTION_STEPS,
    resize: tuple[int, int] = RESIZE_WITH_PADDING,
    empty_cameras: int = EMPTY_CAMERAS,
    rename_map: dict[str, str] | None = None,
    freeze_vision_encoder: bool = True,
    train_expert_only: bool = True,
    gradient_checkpointing: bool = False,
) -> list[str]:
    """Build the `lerobot-train` command as a list of strings."""

    if rename_map is None:
        rename_map = dict(RENAME_MAP)

    resize_h, resize_w = resize

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id=piper/{dataset.name}",
        f"--dataset.root={dataset}",
        f"--output_dir={output}",
        f"--job_name={output.name}",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={model_id}",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}",
        f"--policy.resize_imgs_with_padding=[{resize_h},{resize_w}]",
        f"--policy.max_state_dim={max_state_dim}",
        f"--policy.max_action_dim={max_action_dim}",
        f"--policy.empty_cameras={empty_cameras}",
        f"--policy.freeze_vision_encoder={_bool(freeze_vision_encoder)}",
        f"--policy.train_expert_only={_bool(train_expert_only)}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=0",
        "--persistent_workers=false",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--wandb.enable=false",
    ]

    if gradient_checkpointing:
        cmd.append("--policy.gradient_checkpointing=true")

    if episodes is not None:
        cmd.append(f"--dataset.episodes=[{','.join(map(str, episodes))}]")

    return cmd


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone SmolVLA training for Piper arm (dry-run by default)."
    )
    # Required
    p.add_argument("--dataset", required=True, help="Path to LeRobot v3 dataset directory")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")

    # Training hyperparams
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--save-freq", type=int, default=None, help="Default: same as --steps")
    p.add_argument("--log-freq", type=int, default=10)

    # SmolVLA model
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="Pretrained model path or HF repo id")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--n-action-steps", type=int, default=N_ACTION_STEPS)
    p.add_argument("--max-state-dim", type=int, default=MAX_STATE_DIM)
    p.add_argument("--max-action-dim", type=int, default=MAX_ACTION_DIM)
    p.add_argument("--empty-cameras", type=int, default=EMPTY_CAMERAS)
    p.add_argument("--no-freeze-vision-encoder", dest="freeze_vision_encoder",
                   action="store_false", default=True)
    p.add_argument("--no-train-expert-only", dest="train_expert_only",
                   action="store_false", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)

    # Smoke / subset
    p.add_argument("--episodes", type=int, default=0,
                   help="Limit to first N episodes (0 = all)")

    # Execution control
    p.add_argument("--skip-dataset-check", action="store_true")
    p.add_argument("--start-training", action="store_true")
    p.add_argument("--allow-model-download", action="store_true")
    p.add_argument("--offline-cache-home", default=DEFAULT_CACHE_HOME,
                   help="HF cache directory for offline models")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()

    # ---- Guardrails ----
    if not dataset.is_dir():
        raise SystemExit(f"dataset directory not found: {dataset}")

    if output.exists():
        if args.start_training:
            raise SystemExit(
                f"output directory already exists: {output}. "
                "Remove it or choose a different --output."
            )
        print(f"WARNING: output directory already exists: {output}")

    if output.resolve() == dataset.resolve():
        raise SystemExit("--output must not be the same directory as --dataset")

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    save_freq = args.save_freq or args.steps
    episodes = list(range(args.episodes)) if args.episodes > 0 else None

    # ---- Dataset check ----
    if not args.skip_dataset_check:
        try:
            info = check_dataset(dataset)
            for err in info["errors"]:
                print(f"DATASET ERROR: {err}")
            if info["errors"]:
                if args.start_training:
                    raise SystemExit(1)
            else:
                print(
                    f"dataset_ok=True "
                    f"episodes={info['total_episodes']} "
                    f"frames={info['total_frames']} "
                    f"state_dim={info['state_dim']} "
                    f"action_dim={info['action_dim']} "
                    f"tasks={info.get('tasks', [])}"
                )
        except Exception as exc:
            print(f"dataset_check_failed={type(exc).__name__}: {exc}")
            if args.start_training:
                raise SystemExit(1) from exc
    else:
        print("dataset_check_skipped=True")

    # ---- Build & print command ----
    cmd = build_command(
        dataset=dataset,
        output=output,
        model_id=args.model_id,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=save_freq,
        log_freq=args.log_freq,
        episodes=episodes,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        empty_cameras=args.empty_cameras,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    print(f"training_command={command_to_shell(cmd)}")
    print("no_hardware_access=True")

    if not args.start_training:
        print("training_not_started=True (pass --start-training to launch)")
        return 0

    # ---- Model download guard ----
    if args.model_id.startswith("lerobot/") and not args.allow_model_download:
        raise SystemExit(
            "--allow-model-download is required before using a remote SmolVLA model id.\n"
            "Use --model-id with a local path, or pass --allow-model-download."
        )

    # ---- Launch training ----
    train_cli = shutil.which(cmd[0])
    if train_cli is None:
        raise SystemExit(
            "missing lerobot-train; activate the piper_smolvla conda environment"
        )

    env = os.environ.copy()
    env["HF_HOME"] = args.offline_cache_home
    if not args.allow_model_download:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        env.setdefault(key, "")
    env.setdefault("NO_PROXY", "*")

    cmd = [train_cli, *cmd[1:]]
    print(f"launching={' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=False, env=env).returncode


def _bool(v: bool) -> str:
    return "true" if v else "false"


if __name__ == "__main__":
    raise SystemExit(main())
