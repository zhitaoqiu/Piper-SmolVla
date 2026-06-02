import numpy as np

from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource
from piper_smolvla.collection import (
    CollectionConfig,
    EpisodeBuffer,
    build_collection_features,
    collect_dry_run_episode,
    make_lerobot_frame,
    write_episode,
)
from piper_smolvla.config import PiperSmolVLAAdapterConfig
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def test_build_collection_features_uses_standard_keys():
    features = build_collection_features(CollectionConfig(image_shape_chw=(3, 240, 320)))

    assert features[STATE_KEY]["shape"] == (7,)
    assert features[STATE_KEY]["dtype"] == "float32"
    assert features[ACTION_KEY]["shape"] == (7,)
    assert features[ACTION_KEY]["dtype"] == "float32"
    assert features[GLOBAL_IMAGE_KEY]["shape"] == (3, 240, 320)
    assert features[GLOBAL_IMAGE_KEY]["dtype"] == "video"
    assert features[WRIST_IMAGE_KEY]["shape"] == (3, 240, 320)
    assert features[WRIST_IMAGE_KEY]["dtype"] == "video"


def test_make_lerobot_frame_preserves_standard_schema():
    image = np.zeros((3, 480, 640), dtype=np.uint8)
    observation = {
        STATE_KEY: VALID_VECTOR,
        GLOBAL_IMAGE_KEY: image,
        WRIST_IMAGE_KEY: image,
        "task": "pick block",
    }

    frame = make_lerobot_frame(observation, VALID_VECTOR)

    np.testing.assert_allclose(frame[STATE_KEY], np.asarray(VALID_VECTOR, dtype=np.float32))
    np.testing.assert_allclose(frame[ACTION_KEY], np.asarray(VALID_VECTOR, dtype=np.float32))
    assert frame[STATE_KEY].dtype == np.float32
    assert frame[ACTION_KEY].dtype == np.float32
    assert frame["task"] == "pick block"


def test_collect_dry_run_episode_builds_valid_frames():
    image = np.zeros((3, 480, 640), dtype=np.uint8)
    adapter = PiperSmolVLAAdapter(
        state_source=DryRunPiperIO(VALID_VECTOR),
        image_source=StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image}),
        config=PiperSmolVLAAdapterConfig(),
    )

    buffer = collect_dry_run_episode(adapter, num_frames=3, task="pick block")

    assert isinstance(buffer, EpisodeBuffer)
    assert len(buffer) == 3
    for frame in buffer.frames:
        np.testing.assert_allclose(frame[STATE_KEY], np.asarray(VALID_VECTOR, dtype=np.float32))
        np.testing.assert_allclose(frame[ACTION_KEY], np.asarray(VALID_VECTOR, dtype=np.float32))


def test_write_episode_uses_dataset_api():
    dataset = FakeDataset()
    frame = {STATE_KEY: VALID_VECTOR, ACTION_KEY: VALID_VECTOR, "task": "pick"}

    count = write_episode(dataset, [frame, frame])

    assert count == 2
    assert len(dataset.frames) == 2
    assert dataset.saved == 1
