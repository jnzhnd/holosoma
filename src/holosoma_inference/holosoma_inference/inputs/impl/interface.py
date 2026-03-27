"""Robot interface input providers.

These providers read from the robot SDK's wireless controller (e.g. Unitree
G1/H1 remote, or Booster's emulated version).  They talk to the SDK
``interface`` object directly — no policy reference needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

if TYPE_CHECKING:
    from holosoma_inference.sdk.base.base_interface import BaseInterface


STICK_DEADZONE = 0.1


class InterfaceVelCmdProvider(VelCmdProvider):
    """Reads joystick sticks from the robot SDK interface.

    Returns raw stick values (after deadzone) without any policy-level gating.
    Caches button states so that :class:`InterfaceStateCommandProvider` can
    share the same joystick read via the ``_shared_velocity`` pattern.
    """

    def __init__(self, interface: BaseInterface):
        self.interface = interface
        self.key_states: dict[str, bool] = {}
        self.last_key_states: dict[str, bool] = {}

    def start(self) -> None:
        pass  # Joystick hardware initialized by SDK

    def poll(self) -> VelCmd | None:
        wc_msg = self.interface.get_joystick_msg()
        if wc_msg is None:
            return None

        # Cache button states for the command provider (always, even when
        # sticks are suppressed during button presses).
        self.last_key_states = self.key_states.copy()
        cur_key = self.interface.get_joystick_key(wc_msg)
        if cur_key:
            self.key_states[cur_key] = True
        else:
            self.key_states = dict.fromkeys(self.key_states.keys(), False)

        # Sticks are only read when no buttons are pressed, matching the
        # behaviour of BaseInterface.process_joystick_input.
        if getattr(wc_msg, "keys", 0) != 0:
            return None

        lx = getattr(wc_msg, "lx", 0.0)
        ly = getattr(wc_msg, "ly", 0.0)
        rx = getattr(wc_msg, "rx", 0.0)

        lin_x = ly if abs(ly) > STICK_DEADZONE else 0.0
        lin_y = -lx if abs(lx) > STICK_DEADZONE else 0.0
        ang_z = -rx if abs(rx) > STICK_DEADZONE else 0.0

        return VelCmd((lin_x, lin_y), ang_z)


class InterfaceStateCommandProvider(StateCommandProvider):
    """Reads joystick buttons and translates rising edges to commands.

    If the velocity provider is also interface-based, reads cached button
    states from it.  Otherwise, reads buttons directly from the SDK.
    """

    def __init__(self, interface: BaseInterface, mapping: dict[str, StateCommand]):
        super().__init__(mapping)
        self.interface = interface
        self._shared_velocity: InterfaceVelCmdProvider | None = None
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}

    def poll(self) -> list[StateCommand]:
        if self._shared_velocity is not None:
            key_states = self._shared_velocity.key_states
            last_key_states = self._shared_velocity.last_key_states
        else:
            # Read buttons only (velocity comes from another source)
            wc_msg = self.interface.get_joystick_msg()
            if wc_msg is None:
                return []
            self._last_key_states = self._key_states.copy()
            cur_key = self.interface.get_joystick_key(wc_msg)
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
