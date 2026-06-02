import numpy as np
import pytest

from piper_smolvla.adapter import ActionSinkDisabledError
from piper_smolvla.piper_io import PiperIO, PiperIOConfig
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


class FakeHardware:
    def __init__(self):
        self.state = VALID_VECTOR
        self.connected = False
        self.enabled = False

    @property
    def is_connected(self):
        return self.connected

    @property
    def is_enabled(self):
        return self.enabled

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def enable(self, *, blocking=True):
        self.enabled = True
        return True

    def disable(self):
        self.enabled = False

    def read_state(self):
        return self.state

    def write_action(self, action):
        self.state = tuple(action)
        return self.state


class FakeImages:
    def read_images(self):
        image = np.zeros((3, 480, 640), dtype=np.uint8)
        return {GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image}


def test_piper_io_outputs_standard_observation_and_lifecycle():
    hardware = FakeHardware()
    io = PiperIO(hardware=hardware, image_source=FakeImages(), config=PiperIOConfig(task="pick block"))

    io.connect()
    io.enable()
    observation = io.get_observation()

    assert io.is_connected
    assert io.is_enabled
    assert observation[STATE_KEY] == VALID_VECTOR
    assert observation["task"] == "pick block"
    assert GLOBAL_IMAGE_KEY in observation
    assert WRIST_IMAGE_KEY in observation


def test_piper_io_blocks_action_writes_by_default():
    io = PiperIO(hardware=FakeHardware(), image_source=FakeImages(), config=PiperIOConfig(task="pick block"))

    with pytest.raises(ActionSinkDisabledError):
        io.send_action(VALID_VECTOR)


def test_piper_io_can_send_action_when_enabled():
    hardware = FakeHardware()
    io = PiperIO(
        hardware=hardware,
        image_source=FakeImages(),
        config=PiperIOConfig(task="pick block", allow_action_writes=True),
    )
    target = (0.1, 1.1, -1.1, 0.4, -0.4, 0.2, 0.02)

    sent = io.send_action(target)

    assert sent == target
    assert hardware.state == target

