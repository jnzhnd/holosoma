"""Command types for the input system.

``StateCommand`` enums represent discrete user *intent* (e.g. "start the
policy") decoupled from the physical input that triggered it.

``VelocityCommand`` is a value object carrying absolute velocity state
produced by ``VelocityInput`` providers each cycle.

Device-to-command mappings live in their respective impl modules
(``keyboard.py``, ``joystick.py``, ``ros2.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    """Absolute velocity state emitted by VelocityInput providers.

    Each field matches the shape used by the policy's internal state.
    """

    lin_vel: np.ndarray  # shape [1, 2] — linear x, y (m/s)
    ang_vel: np.ndarray  # shape [1, 1] — angular z (rad/s)


class StateCommand(Enum):
    """All discrete commands dispatched through the input system."""

    # --- Common ---
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

    # --- Locomotion ---
    STAND_TOGGLE = auto()
    ZERO_VELOCITY = auto()
    WALK = auto()
    STAND = auto()

    # --- Whole-body tracking ---
    START_MOTION_CLIP = auto()

    # --- Dual mode ---
    SWITCH_MODE = auto()


# Maps SWITCH_POLICY_N commands to 0-based policy indices.
SWITCH_POLICY_INDEX: dict[StateCommand, int] = {StateCommand[f"SWITCH_POLICY_{n}"]: n - 1 for n in range(1, 10)}
