"""Joystick input providers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.base import OtherInput, VelocityInput

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class JoystickVelocityInput(VelocityInput):
    """Reads joystick sticks for velocity. Caches button states for shared use."""

    def __init__(self, policy: BasePolicy):
        super().__init__(policy)
        self.key_states: dict[str, bool] = {}
        self.last_key_states: dict[str, bool] = {}

    def start(self) -> None:
        pass  # Joystick hardware initialized by SDK

    def poll(self) -> None:
        if self.policy.interface.get_joystick_msg() is None:
            return
        self.last_key_states = self.key_states.copy()
        self.policy.lin_vel_command, self.policy.ang_vel_command, self.key_states = (
            self.policy.interface.process_joystick_input(
                self.policy.lin_vel_command,
                self.policy.ang_vel_command,
                self.policy.stand_command,
                False,
            )
        )


class JoystickOtherInput(OtherInput):
    """Reads joystick buttons and translates rising edges to commands.

    If the velocity provider is also joystick, reads cached button states
    from it. Otherwise, reads buttons directly from the SDK.
    """

    def __init__(self, policy: BasePolicy, mapping: dict[str, Enum]):
        super().__init__(mapping)
        self.policy = policy
        self._shared_velocity: JoystickVelocityInput | None = None
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}

    def poll(self) -> list[Enum]:
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
        commands: list[Enum] = []
        for key, is_pressed in key_states.items():
            if is_pressed and not last_key_states.get(key, False):
                cmd = self._mapping.get(key)
                if cmd is not None:
                    commands.append(cmd)
        return commands
