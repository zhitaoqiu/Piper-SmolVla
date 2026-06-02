from piper_smolvla import schema


def test_locked_joint_order_and_dimensions():
    assert schema.PIPER_JOINT_ORDER == ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
    assert schema.ARM_JOINT_ORDER == ("j1", "j2", "j3", "j4", "j5", "j6")
    assert schema.STATE_DIM == 7
    assert schema.ACTION_DIM == 7
    assert schema.ARM_JOINT_DIM == 6


def test_locked_lerobot_keys_and_units():
    assert schema.STATE_KEY == "observation.state"
    assert schema.GLOBAL_IMAGE_KEY == "observation.images.global_rgb"
    assert schema.WRIST_IMAGE_KEY == "observation.images.wrist_rgb"
    assert schema.ACTION_KEY == "action"
    assert schema.DEFAULT_TASK_INSTRUCTION == "Pick up the cube and put it into the box."
    assert schema.IMAGE_KEYS == ("observation.images.global_rgb", "observation.images.wrist_rgb")
    assert schema.JOINT_UNIT == "radians"
    assert schema.GRIPPER_UNIT == "meters"


def test_can_topology_defaults_do_not_change_mode():
    assert schema.DEFAULT_CAN_TOPOLOGY_POLICY == schema.CAN_TOPOLOGY_PRESERVE
    assert schema.DEFAULT_CALL_MASTER_SLAVE_CONFIG is False
    assert schema.REFERENCE_FOLLOWER_MASTER_SLAVE_CONFIG == (0xFC, 0, 0, 0)


def test_verified_start_pose_contract():
    assert schema.VERIFIED_START_QPOS == (
        0.06292,
        0.00750,
        -0.00396,
        0.02732,
        0.30946,
        -0.09826,
        0.0995,
    )
    assert schema.START_GUARD_ZONE_ARM_TOLERANCE_RAD == (0.10, 0.10, 0.10, 0.12, 0.12, 0.12)
    assert schema.START_GUARD_GRIPPER_OPEN_MIN_M == 0.09
