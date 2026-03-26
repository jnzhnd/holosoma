"""Joystick input providers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from termcolor import colored

from holosoma_inference.inputs.base import OtherInput, VelocityInput

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
    """Reads joystick buttons for discrete commands.

    If the velocity provider is also joystick, reads cached button states
    from it. Otherwise, reads buttons directly from the SDK.
    """

    def __init__(self, policy: BasePolicy):
        super().__init__(policy)
        self._shared_velocity: JoystickVelocityInput | None = None
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}

    def start(self) -> None:
        pass

    def poll(self) -> None:
        if self._shared_velocity is not None:
            key_states = self._shared_velocity.key_states
            last_key_states = self._shared_velocity.last_key_states
        else:
            # Read buttons only (velocity comes from another source)
            wc_msg = self.policy.interface.get_joystick_msg()
            if wc_msg is None:
                return
            self._last_key_states = self._key_states.copy()
            cur_key = self.policy.interface.get_joystick_key(wc_msg)
            if cur_key:
                self._key_states[cur_key] = True
            else:
                self._key_states = dict.fromkeys(self._key_states.keys(), False)
            key_states = self._key_states
            last_key_states = self._last_key_states

        # Edge detection: dispatch on rising edge only
        for key, is_pressed in key_states.items():
            if is_pressed and not last_key_states.get(key, False):
                self.policy.handle_joystick_button(key)

    def handle_joystick_button(self, key: str) -> bool:
        if key == "A":
            self.policy._handle_start_policy()
            return True
        if key == "B":
            self.policy._handle_stop_policy()
            return True
        if key == "Y":
            self.policy._handle_init_state()
            return True
        if key in ("up", "down", "left", "right", "F1"):
            self.policy._handle_joystick_kp_control(key)
            return True
        if key == "select":
            next_index = (self.policy.active_policy_index + 1) % len(self.policy.model_paths)
            self.policy._activate_policy(next_index)
            return True
        if key == "L1+R1":
            self.policy.logger.info(colored("Killing program via joystick command", "red"))
            sys.exit(0)
        return False


# ---------------------------------------------------------------------------
# Locomotion-specific
# ---------------------------------------------------------------------------


class LocomotionJoystickOtherInput(JoystickOtherInput):
    """Adds locomotion buttons: start (stand toggle), L2 (zero velocity)."""

    def handle_joystick_button(self, key: str) -> bool:
        if key == "start":
            self.policy._handle_stand_command()
            return True
        if key == "L2":
            self.policy._handle_zero_velocity()
            return True
        return super().handle_joystick_button(key)


# ---------------------------------------------------------------------------
# WBT-specific
# ---------------------------------------------------------------------------


class WbtJoystickOtherInput(JoystickOtherInput):
    """Adds WBT button: start (begin motion clip)."""

    def handle_joystick_button(self, key: str) -> bool:
        if key == "start":
            self.policy._handle_start_motion_clip()
            return True
        return super().handle_joystick_button(key)
