import pytest

from piper_smolvla.dataset_compat import (
    check_lerobot_dataset,
    check_metadata,
    standardize_frame,
)
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


class FakeImage:
    def __init__(self, shape):
        self.shape = shape


class FakeDataset:
    def __init__(self):
        self.meta = {
            "features": {
                STATE_KEY: {"shape": [7], "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]},
                ACTION_KEY: {"shape": [7], "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]},
                GLOBAL_IMAGE_KEY: {"shape": [3, 480, 640]},
                WRIST_IMAGE_KEY: {"shape": [3, 480, 640]},
            },
            "camera_keys": [GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY],
            "total_episodes": 64,
        }
        self.num_episodes = 64
        self.num_frames = 2
        self.hf_dataset = [
            {STATE_KEY: VALID_VECTOR, ACTION_KEY: VALID_VECTOR},
            {STATE_KEY: VALID_VECTOR, ACTION_KEY: VALID_VECTOR},
        ]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            STATE_KEY: VALID_VECTOR,
            ACTION_KEY: VALID_VECTOR,
            GLOBAL_IMAGE_KEY: FakeImage((3, 480, 640)),
            WRIST_IMAGE_KEY: FakeImage((480, 640, 3)),
            "task": f"task {index}",
        }


def test_standardize_frame_accepts_canonical_smolvla_frame():
    frame = {
        STATE_KEY: VALID_VECTOR,
        ACTION_KEY: VALID_VECTOR,
        GLOBAL_IMAGE_KEY: FakeImage((3, 480, 640)),
        WRIST_IMAGE_KEY: FakeImage((480, 640, 3)),
        "task": "pick block",
    }

    standard = standardize_frame(frame)

    assert standard.state == VALID_VECTOR
    assert standard.action == VALID_VECTOR
    assert standard.task == "pick block"


def test_standardize_frame_rejects_missing_task_and_bad_image():
    frame = {
        STATE_KEY: VALID_VECTOR,
        ACTION_KEY: VALID_VECTOR,
        GLOBAL_IMAGE_KEY: FakeImage((3, 480, 640)),
        WRIST_IMAGE_KEY: FakeImage((1, 480, 640)),
    }

    with pytest.raises(ValueError):
        standardize_frame(frame)

    frame["task"] = "pick block"
    with pytest.raises(ValueError):
        standardize_frame(frame)


def test_check_metadata_accepts_locked_schema():
    assert check_metadata(FakeDataset().meta, expected_episodes=64) == []


def test_check_metadata_reports_schema_mismatch():
    meta = FakeDataset().meta
    meta["features"][STATE_KEY] = {"shape": [8], "names": ["bad"]}

    errors = check_metadata(meta, expected_episodes=64)

    assert any("observation.state shape" in error for error in errors)
    assert any("observation.state names" in error for error in errors)


def test_check_lerobot_dataset_validates_table_and_decoded_samples():
    result = check_lerobot_dataset(FakeDataset(), name="fake", expected_episodes=64)

    assert result.ok
    assert result.total_episodes == 64
    assert result.total_frames == 2
    assert result.checked_state_action_frames == 2
    assert result.decoded_image_frames == 2
    assert result.tasks == {"task 0", "task 1"}

