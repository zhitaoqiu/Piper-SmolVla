import math
import sys
import types

import pytest

from piper_smolvla.hardware import (
    OfficialPiperSdkBackend,
    PiperHardwareConfig,
)

VALID_VECTOR = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.03)


class FakeJointState:
    joint_1 = 90000
    joint_2 = 0
    joint_3 = 0
    joint_4 = 0
    joint_5 = 0
    joint_6 = 0


class FakeJointMsg:
    joint_state = FakeJointState()


class FakeGripperState:
    grippers_angle = 10000


class FakeGripperMsg:
    gripper_state = FakeGripperState()


class FakePiper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakePiper.instances.append(self)

    def ConnectPort(self):
        self.calls.append(("ConnectPort",))

    def MasterSlaveConfig(self, *args):
        self.calls.append(("MasterSlaveConfig", *args))

    def EnablePiper(self):
        self.calls.append(("EnablePiper",))
        return True

    def DisablePiper(self):
        self.calls.append(("DisablePiper",))

    def GetArmJointMsgs(self):
        return FakeJointMsg()

    def GetArmGripperMsgs(self):
        return FakeGripperMsg()

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", *args))

    def JointCtrl(self, *args):
        self.calls.append(("JointCtrl", *args))

    def GripperCtrl(self, *args):
        self.calls.append(("GripperCtrl", *args))

    def EmergencyStop(self, *args):
        self.calls.append(("EmergencyStop", *args))


def install_fake_sdk(monkeypatch):
    module = types.ModuleType("piper_sdk")
    module.C_PiperInterface = FakePiper
    monkeypatch.setitem(sys.modules, "piper_sdk", module)


def test_connect_does_not_enable_or_change_topology_by_default(monkeypatch):
    FakePiper.instances.clear()
    install_fake_sdk(monkeypatch)
    backend = OfficialPiperSdkBackend(
        PiperHardwareConfig(can_port="can-test", connect_settle_sec=0.0)
    )

    backend.connect()

    fake = FakePiper.instances[0]
    assert fake.kwargs == {"can_name": "can-test"}
    assert ("ConnectPort",) in fake.calls
    assert not any(call[0] == "EnablePiper" for call in fake.calls)
    assert not any(call[0] == "MasterSlaveConfig" for call in fake.calls)


def test_reads_and_writes_project_units(monkeypatch):
    FakePiper.instances.clear()
    install_fake_sdk(monkeypatch)
    backend = OfficialPiperSdkBackend(
        PiperHardwareConfig(can_port="can-test", connect_settle_sec=0.0)
    )
    backend.connect()

    state = backend.read_state()
    assert state[0] == pytest.approx(math.pi / 2, abs=math.radians(0.0005))
    assert state[6] == pytest.approx(0.01)

    action = (math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    sent = backend.write_action(action)

    fake = FakePiper.instances[0]
    assert sent == pytest.approx(action)
    assert ("MotionCtrl_2", 0x01, 0x01, 30, 0x00) in fake.calls
    assert ("JointCtrl", 90000, 0, 0, 0, 0, 0) in fake.calls
    assert ("GripperCtrl", 10000, 1000, 0x01, 0) in fake.calls
