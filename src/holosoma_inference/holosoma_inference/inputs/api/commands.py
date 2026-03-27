"""Command enums for input providers.

Commands represent user *intent* (e.g. "start the policy") decoupled from the
physical input that triggered it (e.g. the A button on a joystick, or the ]
key on a keyboard).  Policies dispatch on these enums — input providers only
need to translate device events into the right enum value.

Device-to-command mappings live in their respective impl modules
(``keyboard.py``, ``joystick.py``, ``ros2.py``).
"""

from __future__ import annotations

from enum import Enum, auto


class Command(Enum):
    """Commands shared across all policy types."""

    START = auto()
    STOP = auto()
    INIT = auto()
    NEXT_POLICY = auto()
    KILL = auto()
    KP_UP = auto()
    KP_DOWN = auto()
    KP_UP_FINE = auto()
    KP_DOWN_FINE = auto()
    KP_RESET = auto()
    SWITCH_POLICY_1 = auto()
    SWITCH_POLICY_2 = auto()
    SWITCH_POLICY_3 = auto()
    SWITCH_POLICY_4 = auto()
    SWITCH_POLICY_5 = auto()
    SWITCH_POLICY_6 = auto()
    SWITCH_POLICY_7 = auto()
    SWITCH_POLICY_8 = auto()
    SWITCH_POLICY_9 = auto()


class LocomotionCommand(Enum):
    """Locomotion-specific commands."""

    STAND_TOGGLE = auto()
    ZERO_VELOCITY = auto()
    WALK = auto()
    STAND = auto()
    VEL_FORWARD = auto()
    VEL_BACKWARD = auto()
    VEL_LEFT = auto()
    VEL_RIGHT = auto()
    ANG_VEL_LEFT = auto()
    ANG_VEL_RIGHT = auto()


class WbtCommand(Enum):
    """Whole-body tracking specific commands."""

    START_MOTION_CLIP = auto()


class DualModeCommand(Enum):
    """Dual-mode policy switching."""

    SWITCH_MODE = auto()


# Maps SWITCH_POLICY_N commands to 0-based policy indices.
SWITCH_POLICY_INDEX: dict[Command, int] = {Command[f"SWITCH_POLICY_{n}"]: n - 1 for n in range(1, 10)}
