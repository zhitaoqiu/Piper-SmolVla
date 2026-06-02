import numpy as np

from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource
from piper_smolvla.config import PiperSmolVLAAdapterConfig
from piper_smolvla.deployment import ActionLimitConfig, ActionLimiter, DeploymentConfig, DeploymentRunner
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


class TargetPolicy:
    def __init__(self, action):
        self.action = action

    def __call__(self, batch):
        return self.action


def make_adapter(*, allow_action_sink=False):
    image = np.zeros((3, 480, 640), dtype=np.uint8)
    io = DryRunPiperIO(VALID_VECTOR)
    adapter = PiperSmolVLAAdapter(
        state_source=io,
        image_source=StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image}),
        action_sink=io,
        config=PiperSmolVLAAdapterConfig(allow_action_sink=allow_action_sink),
    )
    return adapter, io


def test_action_limiter_caps_arm_wrist_and_gripper_deltas():
    limiter = ActionLimiter(ActionLimitConfig(max_delta_arm_rad=0.1, max_delta_wrist_rad=0.01, max_delta_gripper_m=0.002))
    target = (1.0, 2.0, -2.0, 1.0, -1.0, 1.0, 0.08)

    limited = limiter.limit(VALID_VECTOR, target)

    assert limited[:3] == (0.1, 1.1, -1.1)
    assert limited[3:6] == (0.51, -0.51, 0.26)
    assert limited[6] == 0.032


def test_deployment_runner_dry_run_does_not_send_actions():
    adapter, io = make_adapter()
    runner = DeploymentRunner(
        adapter=adapter,
        policy=TargetPolicy(VALID_VECTOR),
        config=DeploymentConfig(task="pick block", max_steps=2, send_actions=False),
    )

    steps = runner.run()

    assert len(steps) == 2
    assert all(step.sent_action is None for step in steps)
    assert io.read_state() == VALID_VECTOR


def test_deployment_runner_can_send_to_dry_run_sink_when_enabled():
    adapter, io = make_adapter(allow_action_sink=True)
    target = (0.01, 1.01, -1.01, 0.49, -0.49, 0.24, 0.029)
    runner = DeploymentRunner(
        adapter=adapter,
        policy=TargetPolicy(target),
        config=DeploymentConfig(task="pick block", max_steps=1, send_actions=True),
    )

    steps = runner.run()

    assert steps[0].sent_action == target
    assert io.read_state() == target

