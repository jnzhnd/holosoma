"""Joystick device-to-command mappings."""

from __future__ import annotations

from holosoma_inference.inputs.api.commands import StateCommand

# ---------------------------------------------------------------------------
# Joystick mappings
# ---------------------------------------------------------------------------

JOYSTICK_BASE: dict[str, StateCommand] = {
    "A": StateCommand.START,
    "B": StateCommand.STOP,
    "Y": StateCommand.INIT,
    "up": StateCommand.KP_UP,
    "down": StateCommand.KP_DOWN,
    "left": StateCommand.KP_DOWN_FINE,
    "right": StateCommand.KP_UP_FINE,
    "F1": StateCommand.KP_RESET,
    "select": StateCommand.NEXT_POLICY,
    "L1+R1": StateCommand.KILL,
}

JOYSTICK_LOCOMOTION: dict[str, StateCommand] = {
    **JOYSTICK_BASE,
    "start": StateCommand.STAND_TOGGLE,
    "L2": StateCommand.ZERO_VELOCITY,
}

JOYSTICK_WBT: dict[str, StateCommand] = {
    **JOYSTICK_BASE,
    "start": StateCommand.START_MOTION_CLIP,
}
