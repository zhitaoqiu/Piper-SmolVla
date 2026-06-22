#!/usr/bin/env python3
"""Shadow test: 真机只读 + 离线策略推理，不发送任何动作。

用法:
  # 左绿右蓝场景，测 green prompt
  bash scripts/shadow_test.sh LgRb green

  # 左绿右蓝场景，测 blue prompt
  bash scripts/shadow_test.sh LgRb blue
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_smolvla.adapter import PiperSmolVLAAdapter
from piper_smolvla.policy_io import load_lerobot_policy, prepare_policy_batch, select_policy_action_with_options
from piper_smolvla.cameras import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DATASET_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
)
from piper_smolvla.real_sources import RealPiperStateConfig, RealPiperStateSource
from piper_smolvla.schema import (
    ACTION_DIM,
    GLOBAL_IMAGE_KEY,
    PIPER_JOINT_ORDER,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow test: read-only observation + offline policy inference.")
    p.add_argument("--checkpoint", required=True, help="Path to trained policy checkpoint")
    p.add_argument("--task", required=True, help="Task prompt, e.g. 'Pick up the green object and put it into the box.'")
    p.add_argument("--can-port", default="can0")
    p.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    p.add_argument("--duration-sec", type=float, default=15.0)
    p.add_argument("--output-dir", default="outputs/shadow_test")
    p.add_argument("--fps", type=int, default=DEFAULT_DATASET_FPS, help="Observation + inference frequency")
    p.add_argument("--allow-hardware-readonly", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_hardware_readonly:
        raise SystemExit("ERROR: --allow-hardware-readonly is required before connecting CAN or cameras.")

    print("=" * 60)
    print("SHADOW TEST — NO MOTION COMMANDS WILL BE SENT")
    print(f"task: {args.task}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"duration: {args.duration_sec}s  fps: {args.fps}")
    print("=" * 60)

    # ── load policy ────────────────────────────────────────────────────
    print("\nloading policy...")
    policy = load_lerobot_policy(args.checkpoint)
    print(f"policy loaded: {type(policy).__name__}")

    # ── connect hardware (read-only) ───────────────────────────────────
    print("\nconnecting hardware (read-only)...")
    state_source = RealPiperStateSource(
        RealPiperStateConfig(allow_hardware_readonly=args.allow_hardware_readonly, can_port=args.can_port)
    )
    camera_source = RealCameraSource(
        RealCameraConfig(
            allow_hardware_readonly=args.allow_hardware_readonly,
            global_camera=args.global_camera,
            wrist_camera=args.wrist_camera,
            fps=args.camera_fps,
        )
    )
    adapter = PiperSmolVLAAdapter(state_source=state_source, image_source=camera_source)

    # ── output setup ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"shadow_{ts}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    header = ["timestamp", "frame"] + [f"pred_a{j}" for j in range(ACTION_DIM)] + [f"obs_j{j}" for j in range(7)]
    writer.writerow(header)

    pred_actions: list[np.ndarray] = []
    obs_states: list[np.ndarray] = []

    interval = 1.0 / max(1, args.fps)
    deadline = time.monotonic() + args.duration_sec
    frame_idx = 0

    print(f"\nstarting shadow loop ({args.duration_sec}s)...")
    print("Press Ctrl+C to stop early\n")

    try:
        while time.monotonic() < deadline:
            loop_start = time.monotonic()

            obs = adapter.read_observation(task=args.task)
            current_state = np.asarray(obs[STATE_KEY])

            batch = prepare_policy_batch(obs)
            pred_action = select_policy_action_with_options(policy, batch)

            pred_actions.append(pred_action)
            obs_states.append(current_state)

            row = [time.time(), frame_idx] + list(pred_action) + list(current_state)
            writer.writerow(row)

            if frame_idx % args.fps == 0:
                grip = pred_action[6]
                tag = "CLOSE" if grip < 0.07 else ("OPEN" if grip > 0.09 else "HOLD")
                print(f"  frame {frame_idx:4d}  pred_grip={grip:.4f} ({tag})  "
                      f"obs_grip={current_state[6]:.4f}  j1={pred_action[0]:.3f}  j2={pred_action[1]:.3f}")

            frame_idx += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        csv_f.close()
        camera_source.close()
        state_source.disconnect()
        print("hardware released")

    # ── summary ─────────────────────────────────────────────────────────
    if pred_actions:
        pred = np.array(pred_actions)
        grip = pred[:, 6]
        close_count = int(np.sum(grip < 0.07))
        open_count = int(np.sum(grip > 0.09))
        print(f"\nframes: {len(pred)}")
        print(f"pred gripper: min={grip.min():.4f} max={grip.max():.4f} mean={grip.mean():.4f}")
        print(f"close frames (grip<0.07): {close_count}  open (grip>0.09): {open_count}")
        print(f"no NaN: {not np.any(np.isnan(pred))}  no Inf: {not np.any(np.isinf(pred))}")
        print(f"\nper-joint range:")
        for j, name in enumerate(PIPER_JOINT_ORDER):
            print(f"  {name}: {pred[:, j].min():.4f} ~ {pred[:, j].max():.4f}")
        print(f"\nCSV saved: {csv_path}")

    print("\nNO MOTION COMMAND SENT")
    print("SHADOW TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
