#!/usr/bin/env python3
"""采集 readiness 检查。

连续读取 state/images，组装 SmolVLA frame，默认不写正式数据集、不发动作。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np

from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource
from piper_smolvla.collection import (
    CollectionConfig,
    EpisodeBuffer,
    create_lerobot_dataset,
    make_readonly_transition_frame,
    write_episode,
)
from piper_smolvla.cameras import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
)
from piper_smolvla.real_sources import RealPiperStateConfig, RealPiperStateSource
from piper_smolvla.schema import DEFAULT_TASK_INSTRUCTION, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Piper SmolVLA collection readiness.")
    parser.add_argument("--allow-hardware-readonly", action="store_true")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--wrist-auto-exposure", type=int, default=None,
                        help="Wrist camera auto exposure: 1=on, 0=off.")
    parser.add_argument("--wrist-exposure", type=int, default=None,
                        help="Wrist camera manual exposure value.")
    parser.add_argument("--wrist-gain", type=float, default=None,
                        help="Wrist camera gain/ISO; higher is brighter and noisier.")
    parser.add_argument("--wrist-brightness", type=float, default=None,
                        help="Wrist camera brightness offset.")
    parser.add_argument("--wrist-power-line", type=int, default=None,
                        help="Wrist camera power line frequency: 1=50Hz, 2=60Hz.")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-smoke-output", default="")
    parser.add_argument("--task", default=DEFAULT_TASK_INSTRUCTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter, cleanup = build_adapter(args)
    buffer = EpisodeBuffer()
    deadline = time.monotonic() + max(0.1, args.duration_sec)
    previous = None

    try:
        while time.monotonic() < deadline:
            current = adapter.read_observation(task=args.task)
            if previous is not None:
                frame = make_readonly_transition_frame(
                    previous_state=previous[STATE_KEY],
                    current_state=current[STATE_KEY],
                    previous_images={GLOBAL_IMAGE_KEY: previous[GLOBAL_IMAGE_KEY], WRIST_IMAGE_KEY: previous[WRIST_IMAGE_KEY]},
                    task=args.task,
                )
                buffer.frames.append(frame)
            previous = current
            time.sleep(0.1)
    finally:
        cleanup()

    if len(buffer) == 0:
        raise RuntimeError("readiness buffer did not grow")
    first = buffer.frames[0]
    print(f"frames_buffered={len(buffer)}")
    print(f"frame_keys={sorted(first)}")
    print(f"global_shape={tuple(np.asarray(first[GLOBAL_IMAGE_KEY]).shape)}")
    print(f"wrist_shape={tuple(np.asarray(first[WRIST_IMAGE_KEY]).shape)}")
    print("NO MOTION COMMAND SENT")

    if args.write_smoke_output:
        dataset = create_lerobot_dataset(
            root=Path(args.write_smoke_output),
            repo_id="piper/smolvla_collection_smoke",
            config=CollectionConfig(task=args.task),
        )
        count = write_episode(dataset, buffer.frames)
        print(f"wrote_smoke_frames={count}")
    else:
        print("write_smoke_output=skipped")

    if hasattr(adapter, "image_source") and hasattr(adapter.image_source, "resolved_global"):
        g = adapter.image_source.resolved_global
        w = adapter.image_source.resolved_wrist
        mode = adapter.image_source.assignment_mode
    elif args.allow_hardware_readonly:
        g = args.global_camera.strip()
        w = args.wrist_camera.strip()
        mode = "explicit" if (g.lower() not in ("", "auto") and w.lower() not in ("", "auto")) else "unknown"
        g = g if g.lower() not in ("", "auto") else "/dev/videoX"
        w = w if w.lower() not in ("", "auto") else "/dev/videoY"
    else:
        g, w, mode = "/dev/videoX", "/dev/videoY", "dry-run"
    print(f"camera_assignment_mode={mode}")
    print(f"global_camera_device={g}")
    print(f"wrist_camera_device={w}")
    print("next_one_demo_command:")
    print(
        "python scripts/collect_smolvla_dataset.py "
        "--allow-hardware-readonly --can-port can0 "
        f"--global-camera {g} --wrist-camera {w} "
        "--output data/smolvla_cube_dual_one_demo "
        f"--task {args.task!r} --episodes 1 --operator-demo --require-keyboard-start-stop"
    )
    return 0


def build_adapter(args: argparse.Namespace):
    if args.allow_hardware_readonly:
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
                fps=args.camera_fps,
                wrist_auto_exposure=args.wrist_auto_exposure,
                wrist_exposure_absolute=args.wrist_exposure,
                wrist_gain=args.wrist_gain,
                wrist_brightness=args.wrist_brightness,
                wrist_power_line_frequency=args.wrist_power_line,
            )
        )
        return (
            PiperSmolVLAAdapter(state_source=state_source, image_source=camera_source),
            lambda: (camera_source.close(), state_source.disconnect()),
        )
    if not args.dry_run:
        raise SystemExit("--allow-hardware-readonly is required unless --dry-run is used")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    io = DryRunPiperIO((0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.01))
    camera_source = StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image})
    return PiperSmolVLAAdapter(state_source=io, image_source=camera_source), lambda: None


if __name__ == "__main__":
    raise SystemExit(main())
