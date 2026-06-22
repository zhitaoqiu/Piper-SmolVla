"""schema 常量定义。

本文件锁定 Piper + SmolVLA 项目的状态/动作顺序、维度、单位和 LeRobot
字段名，是所有采集、推理、部署代码共同遵守的接口合同。
"""

from __future__ import annotations

PIPER_JOINT_ORDER: tuple[str, ...] = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
ARM_JOINT_ORDER: tuple[str, ...] = PIPER_JOINT_ORDER[:6]
GRIPPER_NAME = "gripper"

STATE_DIM = 7
ACTION_DIM = 7
ARM_JOINT_DIM = 6

STATE_KEY = "observation.state"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
ACTION_KEY = "action"
DEFAULT_TASK_INSTRUCTION = "Pick up the cube and put it into the box."

IMAGE_KEYS: tuple[str, ...] = (GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY)
LEROBOT_KEYS: tuple[str, ...] = (STATE_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY, ACTION_KEY)

JOINT_UNIT = "radians"
GRIPPER_UNIT = "meters"

CAN_TOPOLOGY_PRESERVE = "preserve_existing"
DEFAULT_CAN_TOPOLOGY_POLICY = CAN_TOPOLOGY_PRESERVE
DEFAULT_CALL_MASTER_SLAVE_CONFIG = False
REFERENCE_FOLLOWER_MASTER_SLAVE_CONFIG = (0xFC, 0, 0, 0)

# 本项目采集默认固定起点，来自 170 条 two-object language 数据采集前验证的 Piper 起点。
VERIFIED_START_QPOS: tuple[float, ...] = (
    0.02430,
    0.00670,
    -0.00390,
    0.01610,
    0.31150,
    -0.07480,
    0.09870,
)

# 起点 zone guard：J1-J3 稍紧，J4-J6 稍宽，夹爪要求打开。
START_GUARD_ZONE_ARM_TOLERANCE_RAD: tuple[float, ...] = (0.08, 0.08, 0.08, 0.10, 0.08, 0.10)
START_GUARD_GRIPPER_OPEN_MIN_M = 0.09
