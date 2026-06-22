from piper_smolvla import camera_utils


def test_resolve_camera_pair_prefers_5mp_global_and_realsense_wrist(monkeypatch):
    devices = ["/dev/video2", "/dev/video4", "/dev/video6"]
    names = {
        "/dev/video2": "Intel(R) RealSense(TM) Depth Camera",
        "/dev/video4": "Intel(R) RealSense(TM) Depth Camera",
        "/dev/video6": "5MP USB Camera: 5MP USB Camera",
    }
    groups = {
        "/dev/video2": "2-5",
        "/dev/video4": "2-5",
        "/dev/video6": "1-12",
    }

    monkeypatch.setattr(camera_utils, "video_device_name", lambda device: names[str(device)])
    monkeypatch.setattr(camera_utils, "video_device_group", lambda device: groups[str(device)])

    global_dev, wrist_dev = camera_utils.resolve_camera_pair("auto", "auto", devices=devices)

    assert global_dev == "/dev/video6"
    assert wrist_dev == "/dev/video2"


def test_realsense_fps_candidates_prioritize_requested_then_supported_rates():
    assert camera_utils.realsense_fps_candidates(25) == (25, 30, 15, 60, 90)
    assert camera_utils.realsense_fps_candidates(30) == (30, 15, 60, 90)


def test_explicit_numeric_camera_spec_normalizes_to_dev_video():
    assert camera_utils.normalize_video_device("6") == "/dev/video6"
    assert camera_utils.normalize_video_device("/dev/video2") == "/dev/video2"


def test_explicit_serial_camera_spec_preserves_case():
    device = camera_utils.resolve_one_camera(
        "RealSenseABC123",
        ["/dev/video2"],
        consumed_paths=set(),
        consumed_groups=set(),
        role="wrist",
    )

    assert device == "RealSenseABC123"


def test_realsense_spec_is_a_physical_camera_candidate():
    spec_a = camera_utils.realsense_spec("ABC123")
    spec_b = camera_utils.realsense_spec("DEF456")

    assert camera_utils.parse_realsense_spec(spec_a) == "ABC123"
    assert camera_utils.is_realsense_device(spec_a)

    global_dev, wrist_dev = camera_utils.resolve_camera_pair("auto", "auto", devices=[spec_a, spec_b])

    assert global_dev == spec_a
    assert wrist_dev == spec_b


def test_realsense_spec_metadata_uses_sdk_inventory(monkeypatch):
    info = camera_utils.RealSenseDeviceInfo(
        serial="ABC123",
        name="Intel RealSense D405",
        usb_type="3.2",
        video_nodes=("/dev/video2", "/dev/video4"),
        groups=("2-5",),
    )
    monkeypatch.setattr(camera_utils, "list_realsense_physical_devices", lambda: [info])

    assert camera_utils.video_device_name("realsense:ABC123") == "Intel RealSense D405 serial=ABC123"
    assert camera_utils.video_device_group("realsense:ABC123") == "2-5"
