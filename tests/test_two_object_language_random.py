import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from piper_smolvla.cameras import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_DATASET_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_two_object_language_random.py"


def import_record_script():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("record_two_object_language_random", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["record_two_object_language_random"] = module
    spec.loader.exec_module(module)
    return module


def test_generate_task_schedule_is_balanced_and_reproducible():
    mod = import_record_script()

    first = mod.generate_task_schedule(100, 100, seed=42)
    second = mod.generate_task_schedule(100, 100, seed=42)
    other = mod.generate_task_schedule(100, 100, seed=43)

    assert first == second
    assert first != other
    assert Counter(first) == {"blue": 100, "green": 100}
    assert first[:100] != ["blue"] * 100
    assert first[:100] != ["green"] * 100


def test_targets_from_schedule_json_supports_task_rows():
    mod = import_record_script()

    data = {
        "tasks": [
            {"schedule_index": 0, "target": "blue", "task": mod.BLUE_TASK},
            {"schedule_index": 1, "target": "green", "task": mod.GREEN_TASK},
        ]
    }

    assert mod.targets_from_schedule_json(data) == ["blue", "green"]


def test_verify_schedule_prefix_rejects_mismatch():
    mod = import_record_script()

    with pytest.raises(SystemExit):
        mod.verify_schedule_prefix(["blue", "green"], ["green"])


def test_verify_schedule_prefix_accepts_saved_prefix():
    mod = import_record_script()

    mod.verify_schedule_prefix(["blue", "green", "blue"], ["blue", "green"])


def test_record_script_camera_defaults_match_project_camera_config():
    mod = import_record_script()

    args = mod.parse_args(["--dataset-root", "data/two_obj_language_200"])

    assert args.global_camera == DEFAULT_GLOBAL_CAMERA
    assert args.wrist_camera == DEFAULT_WRIST_CAMERA
    assert args.camera_fps == DEFAULT_CAMERA_FPS
    assert args.fps == DEFAULT_DATASET_FPS
