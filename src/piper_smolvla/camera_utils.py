"""相机设备发现和分配工具。

本文件只做 /dev/video* 枚举、设备名/USB group 读取、V4L2 可打开性探测，
以及 global/wrist 自动分配。它不会打开长期相机流，也不会连接 Piper。
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── device enumeration ───────────────────────────────────────────────────────

def list_video_devices() -> list[str]:
    return sorted((str(path) for path in Path("/dev").glob("video*") if path.exists()), key=video_sort_key)


def video_sort_key(path: str | Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)$", Path(path).name)
    return (0, int(match.group(1))) if match else (1, Path(path).name)


def normalize_video_device(spec: str) -> str:
    value = spec.strip()
    if value.isdigit():
        return f"/dev/video{value}"
    return value


def video_index(device: str) -> int:
    normalized = normalize_video_device(device)
    return int(Path(normalized).name.replace("video", ""))


# ── device metadata ──────────────────────────────────────────────────────────

def video_device_name(device: str | Path) -> str:
    serial = parse_realsense_spec(device)
    if serial:
        for info in list_realsense_physical_devices():
            if info.serial == serial:
                return f"{info.name} serial={serial}"
        return f"Intel RealSense serial={serial}"
    dev = Path(normalize_video_device(str(device)))
    try:
        return (Path("/sys/class/video4linux") / dev.name / "name").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def video_device_group(device: str | Path) -> str:
    serial = parse_realsense_spec(device)
    if serial:
        for info in list_realsense_physical_devices():
            if info.serial == serial and info.groups:
                return ",".join(info.groups)
        return realsense_spec(serial)
    dev = Path(normalize_video_device(str(device)))
    try:
        resolved = Path(os.path.realpath(Path("/sys/class/video4linux") / dev.name / "device"))
    except OSError:
        return str(dev)
    for part in reversed(resolved.parts):
        if "-" in part and ":" not in part and part[0].isdigit():
            return part
    return str(resolved)


def is_realsense_device(device: str | Path) -> bool:
    if parse_realsense_spec(device):
        return True
    return "realsense" in video_device_name(device).lower()


# ── per-process bad-device cache ─────────────────────────────────────────────

_BAD_DEVICE_CACHE: set[str] = set()


def _invalidated_bad_device_cache() -> None:
    _BAD_DEVICE_CACHE.clear()


def _is_known_bad(device: str) -> bool:
    return normalize_video_device(device) in _BAD_DEVICE_CACHE


def _mark_bad(device: str) -> None:
    _BAD_DEVICE_CACHE.add(normalize_video_device(device))


# Sysfs helper: check device name for known non-RGB patterns.
# NOTE: Do NOT skip "depth" — RealSense RGB color nodes share "Depth" in their V4L2
# device name.  The thread-based probe timeout handles genuinely bad nodes.
_SKIP_NAME_PATTERNS = re.compile(r"metadata|capture[_ ]?\d+|v4l-subdev|gyro|accel|ir\b", re.IGNORECASE)


def _looks_like_rgb_source(device: str) -> bool:
    """Quick sysfs check to skip obvious non-RGB nodes before opening."""
    name = video_device_name(device)
    if bool(_SKIP_NAME_PATTERNS.search(name)):
        return False
    return True


# ── probe ────────────────────────────────────────────────────────────────────

@dataclass
class DeviceProbeInfo:
    device: str
    shape: tuple[int, int, int] | None = None
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    black: bool = False
    name: str = ""
    group: str = ""
    realsense: bool = False
    status: str = "unprobed"  # unprobed | ok | no_frame | timeout | black | skipped


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    serial: str
    name: str = "Intel RealSense"
    usb_type: str = ""
    video_nodes: tuple[str, ...] = field(default_factory=tuple)
    groups: tuple[str, ...] = field(default_factory=tuple)

    @property
    def spec(self) -> str:
        return realsense_spec(self.serial)


# Per-probe timeout in seconds — a single bad node should never stall for 30 s.
_PROBE_TIMEOUT_SEC = 2.5


def probe_readable_v4l2_devices(
    *,
    black_threshold: float = 5.0,
    warmup_frames: int = 3,
    verbose: bool = True,
    skip_realsense: bool = True,
) -> list[str]:
    """返回可被 OpenCV/V4L2 打开、能正常读帧且非黑帧的 /dev/video*。

    每个设备最多等待 _PROBE_TIMEOUT_SEC 秒；超时/无帧设备会被本进程缓存
    并直接跳过。
    """

    try:
        import cv2
        import os as _os
        cv2.setLogLevel(0)
        _os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    except Exception:
        return list_video_devices()

    working: list[str] = []
    for device in list_video_devices():
        if skip_realsense and is_realsense_device(device):
            if verbose:
                print(f"  {device}  SKIP_REALSENSE  name={video_device_name(device)!r}  group={video_device_group(device)}")
            continue
        if _is_known_bad(device):
            if verbose:
                print(f"  {device}  SKIP (cached bad)")
            continue
        info = _probe_single_device(device, cv2, black_threshold=black_threshold, warmup_frames=warmup_frames)
        if verbose:
            _print_probe(info)
        if info.shape is not None and not info.black and info.status == "ok":
            working.append(device)
        else:
            _mark_bad(device)
    if working:
        time.sleep(0.1)
    return working


def probe_readable_v4l2_devices_detailed(
    *,
    black_threshold: float = 5.0,
    warmup_frames: int = 3,
    verbose: bool = True,
    skip_realsense: bool = True,
) -> list[DeviceProbeInfo]:
    """同 probe_readable_v4l2_devices，但返回完整 DeviceProbeInfo 列表。"""

    try:
        import cv2
    except Exception:
        return [DeviceProbeInfo(device=d, name=video_device_name(d), group=video_device_group(d),
                                realsense=is_realsense_device(d))
                for d in list_video_devices()]

    results: list[DeviceProbeInfo] = []
    for device in list_video_devices():
        if skip_realsense and is_realsense_device(device):
            info = DeviceProbeInfo(
                device=device,
                status="skipped_realsense",
                name=video_device_name(device),
                group=video_device_group(device),
                realsense=True,
            )
            if verbose:
                _print_probe(info)
            results.append(info)
            continue
        if _is_known_bad(device):
            info = DeviceProbeInfo(
                device=device, status="skipped",
                name=video_device_name(device), group=video_device_group(device),
                realsense=is_realsense_device(device),
            )
            if verbose:
                _print_probe(info)
            results.append(info)
            continue

        info = _probe_single_device(device, cv2, black_threshold=black_threshold, warmup_frames=warmup_frames)
        if verbose:
            _print_probe(info)
        if info.shape is not None and not info.black and info.status == "ok":
            pass
        else:
            _mark_bad(device)
        results.append(info)
    if results:
        time.sleep(0.1)
    return results


def _probe_single_device(
    device: str,
    cv2: Any,
    *,
    black_threshold: float,
    warmup_frames: int,
) -> DeviceProbeInfo:
    name = video_device_name(device)
    group = video_device_group(device)
    is_rs = is_realsense_device(device)

    if not _looks_like_rgb_source(device):
        info = DeviceProbeInfo(device=device, status="skipped", name=name, group=group, realsense=is_rs)
        return info

    result_container: dict[str, DeviceProbeInfo] = {}

    def _do_probe() -> None:
        info = DeviceProbeInfo(device=device, name=name, group=group, realsense=is_rs)
        cap = cv2.VideoCapture(video_index(device), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                info.status = "no_frame"
                result_container["info"] = info
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            for _ in range(max(1, warmup_frames)):
                cap.read()
                if result_container.get("_cancelled"):
                    info.status = "timeout"
                    result_container["info"] = info
                    return
            ret, frame = cap.read()
            if not ret or frame is None:
                info.status = "no_frame"
                result_container["info"] = info
                return
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.size == 0:
                info.status = "no_frame"
                result_container["info"] = info
                return
            info.shape = arr.shape
            info.mean = float(arr.mean())
            info.min = float(arr.min())
            info.max = float(arr.max())
            info.black = info.mean < black_threshold
            info.status = "black" if info.black else "ok"
        finally:
            cap.release()
        result_container["info"] = info

    t = threading.Thread(target=_do_probe, daemon=True)
    t.start()
    t.join(timeout=_PROBE_TIMEOUT_SEC)
    if t.is_alive():
        result_container["_cancelled"] = True
        # Don't join — thread is stuck in a blocking read; daemon=True so it won't keep process alive.
        return DeviceProbeInfo(device=device, status="timeout", name=name, group=group, realsense=is_rs)

    return result_container.get("info", DeviceProbeInfo(device=device, status="no_frame", name=name, group=group, realsense=is_rs))


def _print_probe(info: DeviceProbeInfo) -> None:
    name_str = f"  name={info.name!r}" if info.name else ""
    group_str = f"  group={info.group}" if info.group else ""
    rs_str = "  REALSENSE" if info.realsense else ""
    extras = f"{name_str}{group_str}{rs_str}"

    if info.status in ("timeout", "skipped", "skipped_realsense", "no_frame"):
        print(f"  {info.device}  {info.status.upper()}{extras}")
        return
    if info.shape is None:
        print(f"  {info.device}  NO_FRAME{extras}")
        return
    tag = "BLACK" if info.black else "OK"
    print(
        f"  {info.device}  {tag}  "
        f"shape={info.shape}  mean={info.mean:.1f}  "
        f"min={info.min:.0f}  max={info.max:.0f}{extras}"
    )


# ── camera pair resolution ───────────────────────────────────────────────────

_RESOLVED_CACHE: dict[tuple[str, str], tuple[str, str]] = {}


def resolve_camera_pair(
    global_spec: str,
    wrist_spec: str,
    *,
    devices: list[str] | None = None,
    allow_same_group: bool = False,
    verbose: bool = True,
) -> tuple[str, str]:
    """将 global/wrist spec 解析为具体 /dev/video* 路径。

    如果 spec 已经是显式路径（非 auto），直接使用。只有当任一 spec 为 auto
    时才探测设备池。默认禁止同一设备或同一 USB group 分配给两路相机。
    """

    # Fast path: both are explicit devices — no probing needed.
    if _is_explicit(global_spec) and _is_explicit(wrist_spec):
        global_dev = normalize_video_device(global_spec)
        wrist_dev = normalize_video_device(wrist_spec)
    else:
        cache_key = (global_spec.strip().lower(), wrist_spec.strip().lower(), str(devices or []), str(allow_same_group))
        if cache_key in _RESOLVED_CACHE:
            return _RESOLVED_CACHE[cache_key]

        candidates = devices if devices is not None else auto_camera_candidates(verbose=verbose)
        if not candidates:
            candidates = list_video_devices()
        if not candidates:
            raise RuntimeError("no /dev/video* device found; pass explicit --global-camera and --wrist-camera")

        global_dev = resolve_one_camera(global_spec, candidates, consumed_paths=set(), consumed_groups=set(), role="global")
        global_group = video_device_group(global_dev)
        wrist_dev = resolve_one_camera(
            wrist_spec,
            candidates,
            consumed_paths={global_dev},
            consumed_groups=set() if allow_same_group else {global_group},
            role="wrist",
        )
        _RESOLVED_CACHE[cache_key] = (global_dev, wrist_dev)

    # Sanity checks.
    if global_dev == wrist_dev:
        raise RuntimeError(
            f"global and wrist cameras resolved to the same device: {global_dev}. "
            "Pass explicit --global-camera and --wrist-camera pointing to different devices."
        )

    global_group = video_device_group(global_dev)
    wrist_group = video_device_group(wrist_dev)
    if not allow_same_group and global_group == wrist_group:
        raise RuntimeError(
            f"global ({global_dev}) and wrist ({wrist_dev}) are from the same USB group "
            f"'{global_group}'. Pass --allow-same-group to override, "
            "or use explicit --global-camera / --wrist-camera from different USB buses."
        )

    return global_dev, wrist_dev


def resolve_one_camera(
    spec: str,
    devices: list[str],
    consumed_paths: set[str],
    consumed_groups: set[str],
    *,
    role: str,
) -> str:
    raw = spec.strip()
    value = raw.lower()
    if value not in ("", "auto"):
        return normalize_video_device(raw)

    ranked = sorted(devices, key=lambda device: camera_auto_rank(device, role))
    for device in ranked:
        if device not in consumed_paths and video_device_group(device) not in consumed_groups:
            return device
    raise RuntimeError(
        f"not enough /dev/video* devices for two cameras (found {len(devices)}). "
        "Pass explicit --global-camera and --wrist-camera."
    )


def camera_auto_rank(device: str, role: str) -> tuple[int, int]:
    name = video_device_name(device).lower()
    number = video_index(device) if _is_video_node(device) else 0
    if role == "global":
        if "5mp" in name or "usb camera" in name:
            return (0, number)
        if "realsense" in name:
            return (5, number)
    if role == "wrist":
        if "realsense" in name:
            return (0, number)
        if "5mp" in name or "usb camera" in name:
            return (5, number)
    return (2, number)


def print_resolved_pair(global_dev: str, wrist_dev: str) -> None:
    """打印已 resolve 的相机对信息。"""
    print(f"\nresolved camera pair:")
    for role, dev in [("global", global_dev), ("wrist", wrist_dev)]:
        name = video_device_name(dev)
        group = video_device_group(dev)
        rs = " (RealSense)" if is_realsense_device(dev) else ""
        print(f"  {role}: {dev}  name={name!r}  group={group}{rs}")


def is_explicit(spec: str) -> bool:
    return _is_explicit(spec)


def _is_explicit(spec: str) -> bool:
    value = spec.strip().lower()
    return value not in ("", "auto")


def auto_camera_candidates(*, verbose: bool = True) -> list[str]:
    """Return physical camera candidates without V4L2-probing RealSense nodes."""

    candidates: list[str] = []
    candidates.extend(probe_readable_v4l2_devices(verbose=verbose, skip_realsense=True))
    for info in list_realsense_physical_devices():
        candidates.append(info.spec)
    return candidates


# ── RealSense helpers ────────────────────────────────────────────────────────

_REALSENSE_SPEC_PREFIX = "realsense:"


def realsense_spec(serial: str) -> str:
    return f"{_REALSENSE_SPEC_PREFIX}{serial.strip()}"


def parse_realsense_spec(spec: str | Path) -> str | None:
    value = str(spec).strip()
    if value.lower().startswith(_REALSENSE_SPEC_PREFIX):
        return value.split(":", 1)[1].strip() or None
    return None


def _is_video_node(value: str | Path) -> bool:
    raw = str(value).strip()
    return raw.startswith("/dev/video") or raw.isdigit()


def list_realsense_physical_devices() -> list[RealSenseDeviceInfo]:
    """List physical RealSense devices via librealsense, with sysfs video nodes."""

    try:
        import pyrealsense2 as rs
    except Exception:
        return []

    try:
        devices = list(rs.context().query_devices())
    except Exception:
        return []

    nodes_by_serial: dict[str, list[str]] = {}
    groups_by_serial: dict[str, set[str]] = {}
    for node in list_video_devices():
        if not is_realsense_device(node):
            continue
        serial = realsense_serial_from_video_device(node)
        if not serial:
            continue
        nodes_by_serial.setdefault(serial, []).append(node)
        groups_by_serial.setdefault(serial, set()).add(video_device_group(node))

    infos: list[RealSenseDeviceInfo] = []
    for dev in devices:
        serial = _get_rs_info(dev, rs.camera_info.serial_number)
        if not serial:
            continue
        infos.append(
            RealSenseDeviceInfo(
                serial=serial,
                name=_get_rs_info(dev, rs.camera_info.name) or "Intel RealSense",
                usb_type=_get_rs_info(dev, rs.camera_info.usb_type_descriptor) or "",
                video_nodes=tuple(sorted(nodes_by_serial.get(serial, []), key=video_sort_key)),
                groups=tuple(sorted(groups_by_serial.get(serial, set()))),
            )
        )
    return infos


def _get_rs_info(dev: Any, key: Any) -> str:
    try:
        return str(dev.get_info(key)).strip()
    except Exception:
        return ""

def realsense_fps_candidates(fps: int) -> tuple[int, ...]:
    ordered: list[int] = []
    for rate in (fps, 30, 15, 60, 90):
        if rate not in ordered:
            ordered.append(rate)
    return tuple(ordered)


def realsense_serial_from_video_device(device: str) -> str | None:
    """尽量从 /dev/videoN 对应的 USB sysfs 路径找到 RealSense serial。"""

    value = device.strip()
    if not value or value.lower() == "auto":
        return None
    explicit_serial = parse_realsense_spec(value)
    if explicit_serial:
        return explicit_serial
    if not value.startswith("/dev/video") and not value.isdigit():
        return value

    dev = Path(normalize_video_device(value))
    try:
        node = Path(os.path.realpath(Path("/sys/class/video4linux") / dev.name / "device"))
    except OSError:
        return None

    for parent in (node, *node.parents):
        serial = _read_nonempty(parent / "serial")
        if serial:
            return serial
        if parent.name == "sys":
            break
    group = video_device_group(dev)
    return _read_nonempty(Path("/sys/bus/usb/devices") / group / "serial")


def _read_nonempty(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def stop_pipeline_quietly(pipe: Any) -> None:
    try:
        pipe.stop()
    except Exception:
        pass
