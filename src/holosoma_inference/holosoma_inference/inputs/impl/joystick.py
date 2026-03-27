"""Joystick input providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.base import StateCommandProvider, VelocityInput
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy

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


class JoystickVelocityInput(VelocityInput):
    """Reads joystick sticks for velocity. Caches button states for shared use."""

    def __init__(self, policy: BasePolicy):
        self.policy = policy
        self.key_states: dict[str, bool] = {}
        self.last_key_states: dict[str, bool] = {}

    def start(self) -> None:
        pass  # Joystick hardware initialized by SDK

    def poll(self) -> VelCmd | None:
        if self.policy.interface.get_joystick_msg() is None:
            return None
        self.last_key_states = self.key_states.copy()
        lin_vel, ang_vel, self.key_states = self.policy.interface.process_joystick_input(
            self.policy.lin_vel_command,
            self.policy.ang_vel_command,
            self.policy.stand_command,
            False,
        )
        return VelCmd(
            (float(lin_vel[0, 0]), float(lin_vel[0, 1])),
            float(ang_vel[0, 0]),
        )


class JoystickStateCommandProvider(StateCommandProvider):
    """Reads joystick buttons and translates rising edges to commands.

    If the velocity provider is also joystick, reads cached button states
    from it. Otherwise, reads buttons directly from the SDK.
    """

    def __init__(self, policy: BasePolicy, mapping: dict[str, StateCommand]):
        super().__init__(mapping)
        self.policy = policy
        self._shared_velocity: JoystickVelocityInput | None = None
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}

    def poll(self) -> list[StateCommand]:
        if self._shared_velocity is not None:
            key_states = self._shared_velocity.key_states
            last_key_states = self._shared_velocity.last_key_states
        else:
            # Read buttons only (velocity comes from another source)
            wc_msg = self.policy.interface.get_joystick_msg()
            if wc_msg is None:
                return []
            self._last_key_states = self._key_states.copy()
            cur_key = self.policy.interface.get_joystick_key(wc_msg)
            if cur_key:
                self._key_states[cur_key] = True
            else:
                self._key_states = dict.fromkeys(self._key_states.keys(), False)
            key_states = self._key_states
            last_key_states = self._last_key_states

        # Edge detection: return commands for rising edges only
        commands: list[StateCommand] = []
        for key, is_pressed in key_states.items():
            if is_pressed and not last_key_states.get(key, False):
                cmd = self._mapping.get(key)
                if cmd is not None:
                    commands.append(cmd)
        return commands
