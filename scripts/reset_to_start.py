#!/usr/bin/env python3
"""逐步把 Piper 臂移到采集起始位置。

使用小步长逐步逼近 VERIFIED_START_QPOS，每步受 delta limit 约束。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import time

import numpy as np

from piper_smolvla.hardware import OfficialPiperSdkBackend, PiperHardwareConfig
from piper_smolvla.schema import PIPER_JOINT_ORDER, VERIFIED_START_QPOS
from piper_smolvla.validation import validate_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move Piper to verified start position step-by-step.")
    p.add_argument("--can-port", default="can0")
    p.add_argument("--velocity-pct", type=int, default=20)
    p.add_argument("--step-rad", type=float, default=0.03, help="Max joint delta per step (rad)")
    p.add_argument("--step-gripper-m", type=float, default=0.005, help="Max gripper delta per step (m)")
    p.add_argument("--tolerance-rad", type=float, default=0.02, help="Joint convergence tolerance (rad)")
    p.add_argument("--tolerance-gripper-m", type=float, default=0.005, help="Gripper convergence tolerance (m)")
    p.add_argument("--max-steps", type=int, default=300, help="Hard stop after this many command steps")
    p.add_argument(
        "--include-gripper",
        action="store_true",
        help="Also move gripper toward VERIFIED_START_QPOS. By default the current gripper position is held.",
    )
    p.add_argument("--allow-hardware-action", action="store_true")
    p.add_argument("--confirm-reset", default="", help="Must be the literal string RESET_TO_START")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_hardware_action:
        raise SystemExit("ERROR: --allow-hardware-action is required before connecting CAN or sending commands.")
    if args.confirm_reset != "RESET_TO_START":
        raise SystemExit("ERROR: --confirm-reset must be the literal string RESET_TO_START.")
    if args.max_steps <= 0:
        raise SystemExit("ERROR: --max-steps must be positive.")
    if args.step_rad <= 0 or args.step_gripper_m <= 0:
        raise SystemExit("ERROR: step limits must be positive.")

    target = np.array(VERIFIED_START_QPOS, dtype=np.float64)

    print("=" * 60)
    print("RESET TO START POSITION — REAL HARDWARE ACTION")
    print(f"target: {[round(v, 4) for v in target]}")
    print(f"step:   {args.step_rad} rad / {args.step_gripper_m} m")
    print(f"max steps: {args.max_steps}")
    print(f"include gripper: {args.include_gripper}")
    print("=" * 60)

    hw_cfg = PiperHardwareConfig(
        can_port=args.can_port,
        enable_on_connect=False,
        disable_on_disconnect=False,
        call_master_slave_config=False,
        velocity_pct=args.velocity_pct,
    )
    backend = OfficialPiperSdkBackend(hw_cfg)
    state = np.zeros(7, dtype=np.float64)
    step = 0

    try:
        backend.connect()
        print(f"CAN connected: {args.can_port}")
        state = np.array(backend.read_state(), dtype=np.float64)
        if not args.include_gripper:
            target[6] = state[6]
        print(f"current: {[round(v, 4) for v in state]}\n")

        while True:
            # ── joint errors ──────────────────────────────────────────────
            errors = target - state
            joint_done = all(abs(errors[i]) < args.tolerance_rad for i in range(6))
            gripper_done = abs(errors[6]) < args.tolerance_gripper_m

            if joint_done and gripper_done:
                print(f"\nconverged at step {step}")
                print(f"final: {[round(v, 4) for v in state]}")
                break
            if step >= args.max_steps:
                raise RuntimeError(f"max steps reached before convergence: {args.max_steps}")

            # ── compute next target ───────────────────────────────────────
            next_action = state.copy()
            for i in range(6):
                delta = errors[i]
                clamped = max(-args.step_rad, min(args.step_rad, delta))
                next_action[i] = state[i] + clamped
            if args.include_gripper:
                grip_delta = errors[6]
                next_action[6] = state[6] + max(-args.step_gripper_m, min(args.step_gripper_m, grip_delta))
            else:
                next_action[6] = state[6]

            next_action = np.array(validate_action(next_action, check_limits=True), dtype=np.float64)

            # ── send ──────────────────────────────────────────────────────
            sent = backend.write_action(next_action)

            # ── print ─────────────────────────────────────────────────────
            err_str = " ".join(f"{errors[i]:+.4f}" for i in range(6))
            act_str = " ".join(f"{sent[i]:.4f}" for i in range(6))
            status = " ".join(
                "✓" if abs(errors[i]) < args.tolerance_rad else "·" for i in range(6)
            )
            print(f"  step {step:3d}  err: [{err_str}] grip_err={errors[6]:+.4f}  "
                  f"sent: [{act_str}] grip={sent[6]:.4f}  [{status}]")

            state = np.array(backend.read_state(), dtype=np.float64)
            step += 1

            time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\nstopped at step {step}")
        print(f"last state: {[round(v, 4) for v in state]}")
    finally:
        backend.disconnect()
        print("hardware released (NOT disabled)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
