"""Command enums and device-to-command mappings for input providers.

Commands represent user *intent* (e.g. "start the policy") decoupled from the
physical input that triggered it (e.g. the A button on a joystick, or the ]
key on a keyboard).  Policies dispatch on these enums — input providers only
need to translate device events into the right enum value.
"""

from __future__ import annotations

from enum import Enum, auto

# ---------------------------------------------------------------------------
# Command enums
# ---------------------------------------------------------------------------


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


class WbtCommand(Enum):
    """Whole-body tracking specific commands."""

    START_MOTION_CLIP = auto()


class DualModeCommand(Enum):
    """Dual-mode policy switching."""

    SWITCH_MODE = auto()


# Maps SWITCH_POLICY_N commands to 0-based policy indices.
SWITCH_POLICY_INDEX: dict[Command, int] = {
    Command.SWITCH_POLICY_1: 0,
    Command.SWITCH_POLICY_2: 1,
    Command.SWITCH_POLICY_3: 2,
    Command.SWITCH_POLICY_4: 3,
    Command.SWITCH_POLICY_5: 4,
    Command.SWITCH_POLICY_6: 5,
    Command.SWITCH_POLICY_7: 6,
    Command.SWITCH_POLICY_8: 7,
    Command.SWITCH_POLICY_9: 8,
}


# ---------------------------------------------------------------------------
# Joystick mappings
# ---------------------------------------------------------------------------

JOYSTICK_BASE: dict[str, Enum] = {
    "A": Command.START,
    "B": Command.STOP,
    "Y": Command.INIT,
    "up": Command.KP_UP,
    "down": Command.KP_DOWN,
    "left": Command.KP_DOWN_FINE,
    "right": Command.KP_UP_FINE,
    "F1": Command.KP_RESET,
    "select": Command.NEXT_POLICY,
    "L1+R1": Command.KILL,
}

JOYSTICK_LOCOMOTION: dict[str, Enum] = {
    **JOYSTICK_BASE,
    "start": LocomotionCommand.STAND_TOGGLE,
    "L2": LocomotionCommand.ZERO_VELOCITY,
}

JOYSTICK_WBT: dict[str, Enum] = {
    **JOYSTICK_BASE,
    "start": WbtCommand.START_MOTION_CLIP,
}


# ---------------------------------------------------------------------------
# Keyboard mappings
# ---------------------------------------------------------------------------

KEYBOARD_BASE: dict[str, Enum] = {
    "]": Command.START,
    "o": Command.STOP,
    "i": Command.INIT,
    "v": Command.KP_DOWN_FINE,
    "b": Command.KP_UP_FINE,
    "f": Command.KP_DOWN,
    "g": Command.KP_UP,
    "r": Command.KP_RESET,
    **{str(n): Command[f"SWITCH_POLICY_{n}"] for n in range(1, 10)},
}

KEYBOARD_LOCOMOTION: dict[str, Enum] = {
    **KEYBOARD_BASE,
    "=": LocomotionCommand.STAND_TOGGLE,
}

KEYBOARD_WBT: dict[str, Enum] = {
    **KEYBOARD_BASE,
    "s": WbtCommand.START_MOTION_CLIP,
}


# ---------------------------------------------------------------------------
# ROS2 string-to-command mapping
# ---------------------------------------------------------------------------

ROS2_COMMAND_MAP: dict[str, Enum] = {
    "start": Command.START,
    "stop": Command.STOP,
    "init": Command.INIT,
    "walk": LocomotionCommand.WALK,
    "stand": LocomotionCommand.STAND,
}
