import pytest

from piper_smolvla.features import (
    MOTOR_POS_KEYS,
    build_lerobot_feature_spec,
    motor_pos_dict_to_vector,
    required_observation_keys,
    vector_to_motor_pos_dict,
)
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, PIPER_JOINT_ORDER, STATE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


def test_vector_to_motor_pos_dict_preserves_flat_motor_key_order():
    values = vector_to_motor_pos_dict(VALID_VECTOR)

    assert tuple(values) == MOTOR_POS_KEYS
    assert values["j1.pos"] == VALID_VECTOR[0]
    assert values["gripper.pos"] == VALID_VECTOR[6]


def test_motor_pos_dict_to_vector_accepts_pos_and_bare_keys():
    pos_values = vector_to_motor_pos_dict(VALID_VECTOR)
    bare_values = {
        "j1": VALID_VECTOR[0],
        "j2": VALID_VECTOR[1],
        "j3": VALID_VECTOR[2],
        "j4": VALID_VECTOR[3],
        "j5": VALID_VECTOR[4],
        "j6": VALID_VECTOR[5],
        "gripper": VALID_VECTOR[6],
    }

    assert motor_pos_dict_to_vector(pos_values) == VALID_VECTOR
    assert motor_pos_dict_to_vector(bare_values) == VALID_VECTOR


def test_motor_pos_dict_to_vector_rejects_missing_keys():
    with pytest.raises(KeyError):
        motor_pos_dict_to_vector({"j1.pos": 0.0})


def test_build_lerobot_feature_spec_uses_canonical_keys():
    features = build_lerobot_feature_spec(
        image_shapes={
            GLOBAL_IMAGE_KEY: (3, 480, 640),
            WRIST_IMAGE_KEY: (3, 480, 640),
        }
    )

    assert features[STATE_KEY] == {"dtype": "float32", "shape": (7,), "names": list(PIPER_JOINT_ORDER)}
    assert features[ACTION_KEY] == {"dtype": "float32", "shape": (7,), "names": list(PIPER_JOINT_ORDER)}
    assert features[GLOBAL_IMAGE_KEY] == {"dtype": "video", "shape": (3, 480, 640)}
    assert features[WRIST_IMAGE_KEY] == {"dtype": "video", "shape": (3, 480, 640)}
    assert required_observation_keys() == (STATE_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY)
