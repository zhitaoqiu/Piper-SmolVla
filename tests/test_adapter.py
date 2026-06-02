from dataclasses import replace

import pytest

from piper_smolvla.adapter import (
    ActionSinkDisabledError,
    DryRunPiperIO,
    MissingImageError,
    MissingTaskError,
    PiperSmolVLAAdapter,
    StaticImageSource,
)
from piper_smolvla.config import PiperSmolVLAAdapterConfig, validate_adapter_config
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import LimitValidationError, ShapeValidationError

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)
IMAGES = {GLOBAL_IMAGE_KEY: "global-image", WRIST_IMAGE_KEY: "wrist-image"}


def make_adapter(**kwargs):
    state_source = kwargs.pop("state_source", DryRunPiperIO(VALID_VECTOR))
    image_source = kwargs.pop("image_source", StaticImageSource(IMAGES))
    return PiperSmolVLAAdapter(state_source=state_source, image_source=image_source, **kwargs)


def test_read_observation_returns_canonical_smolvla_frame():
    adapter = make_adapter()

    observation = adapter.read_observation(task="pick bottle")

    assert observation[STATE_KEY] == VALID_VECTOR
    assert observation[GLOBAL_IMAGE_KEY] == "global-image"
    assert observation[WRIST_IMAGE_KEY] == "wrist-image"
    assert observation["task"] == "pick bottle"


def test_read_observation_requires_task_by_default():
    adapter = make_adapter()

    with pytest.raises(MissingTaskError):
        adapter.read_observation(task="")


def test_read_observation_requires_locked_image_keys():
    adapter = make_adapter(image_source=StaticImageSource({GLOBAL_IMAGE_KEY: "global-image"}))

    with pytest.raises(MissingImageError):
        adapter.read_observation(task="pick bottle")


def test_prepare_policy_batch_validates_existing_observation():
    adapter = make_adapter()
    observation = {
        STATE_KEY: list(VALID_VECTOR),
        GLOBAL_IMAGE_KEY: "global-image",
        WRIST_IMAGE_KEY: "wrist-image",
        "task": "pick bottle",
    }

    batch = adapter.prepare_policy_batch(observation)

    assert batch[STATE_KEY] == VALID_VECTOR
    assert batch["task"] == "pick bottle"


def test_prepare_policy_batch_rejects_missing_keys_and_bad_shape():
    adapter = make_adapter()

    with pytest.raises(MissingImageError):
        adapter.prepare_policy_batch({STATE_KEY: VALID_VECTOR, GLOBAL_IMAGE_KEY: "global", "task": "pick"})

    with pytest.raises(ShapeValidationError):
        adapter.prepare_policy_batch(
            {
                STATE_KEY: [0.0] * 6,
                GLOBAL_IMAGE_KEY: "global",
                WRIST_IMAGE_KEY: "wrist",
                "task": "pick",
            }
        )


def test_send_action_is_disabled_by_default_after_validation():
    adapter = make_adapter(action_sink=DryRunPiperIO(VALID_VECTOR))

    with pytest.raises(ActionSinkDisabledError):
        adapter.send_action(VALID_VECTOR)

    with pytest.raises(ShapeValidationError):
        adapter.send_action([0.0] * 6)


def test_send_action_can_update_dry_run_sink_when_explicitly_enabled():
    io = DryRunPiperIO(VALID_VECTOR)
    config = PiperSmolVLAAdapterConfig(allow_action_sink=True)
    adapter = make_adapter(state_source=io, action_sink=io, config=config)
    action = (0.1, 1.1, -1.1, 0.4, -0.4, 0.2, 0.02)

    sent = adapter.send_action(action)

    assert sent == action
    assert io.read_state() == action


def test_send_action_rejects_limit_errors_before_sink_write():
    io = DryRunPiperIO(VALID_VECTOR)
    config = PiperSmolVLAAdapterConfig(allow_action_sink=True)
    adapter = make_adapter(state_source=io, action_sink=io, config=config)
    bad_action = list(VALID_VECTOR)
    bad_action[0] = 3.0

    with pytest.raises(LimitValidationError):
        adapter.send_action(bad_action)

    assert io.read_state() == VALID_VECTOR


def test_adapter_config_preserves_can_mode_by_default():
    config = PiperSmolVLAAdapterConfig()

    assert config.can_topology_policy == "preserve_existing"
    assert config.call_master_slave_config is False
    assert config.allow_action_sink is False
    validate_adapter_config(config)


def test_adapter_config_rejects_implicit_master_slave_change():
    config = replace(PiperSmolVLAAdapterConfig(), call_master_slave_config=True)

    with pytest.raises(ValueError):
        validate_adapter_config(config)

