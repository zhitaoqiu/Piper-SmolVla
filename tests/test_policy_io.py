import numpy as np
import pytest

from piper_smolvla.policy_io import (
    HoldCurrentPolicy,
    ProcessedPolicy,
    extract_action,
    image_to_chw_float32,
    prepare_policy_batch,
    select_policy_action,
    select_policy_action_with_options,
)
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


def make_observation():
    image_hwc = np.zeros((480, 640, 3), dtype=np.uint8)
    image_chw = np.zeros((3, 480, 640), dtype=np.uint8)
    return {
        STATE_KEY: VALID_VECTOR,
        GLOBAL_IMAGE_KEY: image_hwc,
        WRIST_IMAGE_KEY: image_chw,
        "task": "pick block",
    }


def test_image_to_chw_float32_accepts_hwc_and_chw():
    hwc = np.zeros((480, 640, 3), dtype=np.uint8)
    chw = np.zeros((3, 480, 640), dtype=np.uint8)

    assert image_to_chw_float32(hwc).shape == (3, 480, 640)
    assert image_to_chw_float32(chw).shape == (3, 480, 640)


def test_prepare_policy_batch_uses_standard_keys():
    batch = prepare_policy_batch(make_observation())

    assert tuple(batch[STATE_KEY].shape) == (1, 7)
    assert tuple(batch[GLOBAL_IMAGE_KEY].shape) == (1, 3, 480, 640)
    assert tuple(batch[WRIST_IMAGE_KEY].shape) == (1, 3, 480, 640)
    assert batch["task"] == ["pick block"]


def test_extract_action_accepts_common_policy_output_shapes():
    assert extract_action(VALID_VECTOR) == VALID_VECTOR
    assert extract_action({ACTION_KEY: [VALID_VECTOR]}) == VALID_VECTOR
    assert extract_action({"actions": [[VALID_VECTOR]]}) == VALID_VECTOR
    assert extract_action([[(*VALID_VECTOR, 99.0)]]) == VALID_VECTOR


def test_extract_action_rejects_missing_action_key():
    with pytest.raises(KeyError):
        extract_action({"not_action": VALID_VECTOR})


def test_extract_action_rejects_nan_and_can_skip_limit_validation():
    with pytest.raises(ValueError):
        extract_action([float("nan")] * 7)

    out_of_limits = (0.0, -0.01, 0.01, 0.0, 0.0, 0.0, 0.03)
    assert extract_action(out_of_limits, validate_limits=False) == out_of_limits


def test_select_policy_action_with_hold_current_policy():
    batch = prepare_policy_batch(make_observation())

    action = select_policy_action(HoldCurrentPolicy(), batch)

    assert action == pytest.approx(VALID_VECTOR)


def test_processed_policy_runs_preprocessor_and_postprocessor():
    calls = []

    class FakePreprocessor:
        def __call__(self, batch):
            calls.append("pre")
            out = dict(batch)
            out["preprocessed"] = True
            return out

    class FakePolicy:
        def select_action(self, batch):
            calls.append("policy")
            assert batch["preprocessed"] is True
            return [VALID_VECTOR]

    class FakePostprocessor:
        def __call__(self, action):
            calls.append("post")
            return action

    processed = ProcessedPolicy(FakePolicy(), FakePreprocessor(), FakePostprocessor())
    batch = prepare_policy_batch(make_observation())

    action = select_policy_action(processed, batch)

    assert calls == ["pre", "policy", "post"]
    assert action == pytest.approx(VALID_VECTOR)


def test_select_policy_action_can_skip_limit_validation():
    out_of_limits = (0.0, -0.01, 0.01, 0.0, 0.0, 0.0, 0.03)

    action = select_policy_action_with_options(
        lambda batch: out_of_limits,
        {},
        validate_limits=False,
    )

    assert action == out_of_limits
