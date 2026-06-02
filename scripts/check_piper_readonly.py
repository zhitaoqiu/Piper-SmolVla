#!/usr/bin/env python3
"""真实 Piper + 双摄只读检查。

只有显式 --allow-hardware-readonly 才会连接 CAN/相机；脚本不发送任何动作。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from piper_smolvla.adapter import PiperSmolVLAAdapter
from piper_smolvla.real_sources import RealCameraConfig, RealCameraSource, RealPiperStateConfig, RealPiperStateSource
from piper_smolvla.schema import DEFAULT_TASK_INSTRUCTION, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Piper + camera observation check.")
    parser.add_argument("--allow-hardware-readonly", action="store_true")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--global-camera", required=True)
    parser.add_argument("--wrist-camera", default="auto")
    parser.add_argument("--duration-sec", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_hardware_readonly:
        raise SystemExit("--allow-hardware-readonly is required for real Piper/camera access")

    state_source = RealPiperStateSource(
        RealPiperStateConfig(
            allow_hardware_readonly=True,
            can_port=args.can_port,
        )
    )
    camera_source = RealCameraSource(
        RealCameraConfig(
            allow_hardware_readonly=True,
            global_camera=args.global_camera,
            wrist_camera=args.wrist_camera,
        )
    )
    adapter = PiperSmolVLAAdapter(state_source=state_source, image_source=camera_source)

    deadline = time.monotonic() + max(0.1, args.duration_sec)
    observation = None
    samples = 0
    try:
        while time.monotonic() < deadline:
            observation = adapter.read_observation(task=DEFAULT_TASK_INSTRUCTION)
            samples += 1
            time.sleep(0.1)
    except Exception as exc:
        print(f"READONLY_CHECK_FAILED: {type(exc).__name__}: {exc}")
        print_failure_hint(exc, args)
        print("NO MOTION COMMAND SENT")
        return 1
    finally:
        camera_source.close()
        state_source.disconnect()

    if observation is None:
        raise RuntimeError("no observation was read")

    qpos = observation[STATE_KEY]
    global_img = observation[GLOBAL_IMAGE_KEY]
    wrist_img = observation[WRIST_IMAGE_KEY]
    qpos_arr = np.asarray(qpos)
    global_arr = np.asarray(global_img)
    wrist_arr = np.asarray(wrist_img)
    print(f"samples={samples}")
    print(f"qpos={list(qpos)}")
    print(f"gripper_m={qpos[6]:.6f}")
    print(f"observation_keys={sorted(observation.keys())}")
    print(f"global_image_shape={tuple(global_arr.shape)}")
    print(f"wrist_image_shape={tuple(wrist_arr.shape)}")
    print(f"global_image_mean={float(global_arr.mean()):.2f}  min={global_arr.min()}  max={global_arr.max()}")
    print(f"wrist_image_mean={float(wrist_arr.mean()):.2f}  min={wrist_arr.min()}  max={wrist_arr.max()}")
    all_finite = bool(np.isfinite(qpos_arr).all())
    print(f"no_nan_inf={all_finite}")
    print("NO MOTION COMMAND SENT")
    return 0


def print_failure_hint(exc: Exception, args: argparse.Namespace) -> None:
    message = str(exc).lower()
    if "heartbeat lost" in message or "sendcanmessage" in message:
        print("hint=CAN link is up at Linux level, but Piper heartbeat/feedback was not received.")
        print(f"can_port={args.can_port}")
        print("check_1=ip -details -statistics link show can0")
        print("check_2=timeout 2 candump -L can0")
        print("check_3=confirm Piper power, emergency stop, CAN-H/CAN-L wiring, adapter, and bitrate 1000000")
        print("check_4=confirm no other process owns or reconfigures the same CAN adapter")
        print("note=script did not call MasterSlaveConfig, reset, enable, or send_action")
    if "no /dev/video" in message:
        print("hint=no camera device found. Check USB connection.")
        print("fix_1=pass --global-camera /dev/video0 --wrist-camera /dev/video2 explicitly")
        print("fix_2=or check that cameras are plugged in and recognized by the kernel (ls /dev/video*)")
    if "cannot open camera" in message or "cannot open v4l2 camera" in message:
        print("hint=camera device exists but could not be opened.")
        print("fix=check that no other process is using the camera (fuser /dev/video*)")
    if "failed to read frame" in message or "failed to read first frame" in message:
        print("hint=camera opened but could not deliver a valid frame.")
        print("fix=try a different /dev/video* node for this camera, or check USB cable/power.")
    if "appears black" in message:
        print("hint=camera delivers frames but they are nearly black (mean < threshold).")
        print("fix=check lens cap, lighting, or try --global-camera / --wrist-camera to skip bad nodes.")


if __name__ == "__main__":
    raise SystemExit(main())
