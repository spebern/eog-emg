import subprocess
from enum import Enum


class Position(Enum):
    UP_LEFT = 1
    UP_RIGHT = 2
    DOWN_LEFT = 3
    DOWN_RIGHT = 4
    HOME = 5


class GripperState(Enum):
    OPEN = "open"
    CLOSED = "close"


class Robot:
    def __init__(self):
        self.position = None
        self.gripper_state = None

        self.move_home()
        self.open_gripper()

    def move_up_right(self):
        if self.position == Position.HOME:
            _move2point(Position.UP_RIGHT.value)
            self.position = Position.UP_RIGHT
            return True
        else:
            return False

    def move_up_left(self):
        if self.position == Position.HOME:
            _move2point(Position.UP_LEFT.value)
            self.position = Position.UP_LEFT
            return True
        else:
            return False

    def move_down_right(self):
        if self.position == Position.HOME:
            _move2point(Position.DOWN_RIGHT.value)
            self.position = Position.DOWN_RIGHT
            return True
        else:
            return False

    def move_down_left(self):
        if self.position == Position.HOME:
            _move2point(Position.DOWN_LEFT.value)
            self.position = Position.DOWN_LEFT
            return True
        else:
            return False

    def move_home(self):
        if self.position != Position.HOME:
            _move2point(Position.HOME.value)
            self.position = Position.HOME
            return True
        else:
            return False

    def close_gripper(self):
        if self.gripper_state != GripperState.CLOSED:
            _set_gripper_state(GripperState.CLOSED.value)
            self.free_gripper()
            self.gripper_state = GripperState.CLOSED
            return True
        else:
            return False

    def open_gripper(self):
        if self.gripper_state != GripperState.OPEN:
            _set_gripper_state(GripperState.OPEN.value)
            self.free_gripper()
            self.gripper_state = GripperState.OPEN
            return True
        else:
            return False

    def free_gripper(self):
        _set_gripper_state("free")


def _move2point(point_id, duration=5.0):
    subprocess.run(
        [
            "rosservice",
            "call",
            "/Move2Point",
            "point_id: {}\nduration: {}".format(point_id, duration),
        ]
    )


def _set_gripper_state(state):
    subprocess.run(
        ["rosservice", "call", "/setGripperState", "newState: '{}'".format(state)]
    )
