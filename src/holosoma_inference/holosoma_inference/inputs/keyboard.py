"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from holosoma_inference.inputs.base import OtherInput, VelocityInput

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class KeyboardListener:
    """Shared sshkeyboard listener thread.

    Created lazily by keyboard providers and stored on the policy as
    ``_keyboard_listener``.  Multiple providers share one instance;
    ``start()`` is idempotent.  Keypresses are dispatched through
    ``policy.handle_keyboard_button()`` so DualMode monkey-patching
    still works.
    """

    def __init__(self, policy: BasePolicy) -> None:
        self._policy = policy
        self._started = False

    def start(self) -> None:
        """Start the listener thread (idempotent, skipped for shared-hardware secondaries)."""
        if self._started:
            return
        self._started = True

        if not sys.stdin.isatty():
            self._policy.logger.warning("Not running in a TTY environment - keyboard input disabled")
            self._policy.logger.warning("This is normal for automated tests or non-interactive environments")
            self._policy.logger.info("Auto-starting policy in non-interactive mode")
            self._policy.use_keyboard = False
            self._policy.use_policy_action = True
            return

        self._policy.use_keyboard = True
        self._policy.logger.info("Using keyboard")
        threading.Thread(target=self._listen, daemon=True).start()
        self._policy.logger.info("Keyboard Listener Initialized")

    def _listen(self) -> None:
        from sshkeyboard import listen_keyboard

        def on_press(keycode):
            try:
                self._policy.handle_keyboard_button(keycode)
            except AttributeError:
                pass

        try:
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()
        except OSError as e:
            self._policy.logger.warning("Could not start keyboard listener: %s", e)
            self._policy.logger.warning("Keyboard input will not be available")


def _ensure_keyboard_listener(policy: BasePolicy) -> None:
    """Ensure the shared KeyboardListener exists and is started on *policy*.

    Skipped when the policy is a shared-hardware secondary (the primary
    policy's listener thread already dispatches to both).
    """
    if hasattr(policy, "_shared_hardware_source"):
        return
    if not hasattr(policy, "_keyboard_listener"):
        policy._keyboard_listener = KeyboardListener(policy)
    policy._keyboard_listener.start()


class KeyboardVelocityInput(VelocityInput):
    """Base keyboard velocity — no keys handled.

    Subclass for policy-specific velocity keys (e.g. WASD for locomotion).
    """

    def start(self) -> None:
        _ensure_keyboard_listener(self.policy)


class KeyboardOtherInput(OtherInput):
    """Keyboard handler for common discrete commands.

    Keys: ] (start), o (stop), i (init), v/b/f/g/r (kp tuning), 1-9 (switch policy).
    """

    def start(self) -> None:
        _ensure_keyboard_listener(self.policy)

    def handle_key(self, keycode: str) -> bool:
        if self.policy._try_switch_policy_key(keycode):
            return True
        if keycode == "]":
            self.policy._handle_start_policy()
            return True
        if keycode == "o":
            self.policy._handle_stop_policy()
            return True
        if keycode == "i":
            self.policy._handle_init_state()
            return True
        if keycode in ("v", "b", "f", "g", "r"):
            self.policy._handle_kp_control(keycode)
            return True
        return False


# ---------------------------------------------------------------------------
# Locomotion-specific
# ---------------------------------------------------------------------------


class LocomotionKeyboardVelocityInput(KeyboardVelocityInput):
    """Locomotion velocity keys: W/A/S/D (linear), Q/E (angular), Z (zero)."""

    def handle_key(self, keycode: str) -> bool:
        if keycode in ("w", "s", "a", "d"):
            self.policy._handle_velocity_control(keycode)
            return True
        if keycode in ("q", "e"):
            self.policy._handle_angular_velocity_control(keycode)
            return True
        if keycode == "z":
            self.policy._handle_zero_velocity()
            return True
        return False


class LocomotionKeyboardOtherInput(KeyboardOtherInput):
    """Adds stand/walk toggle (=) to base keyboard commands."""

    def handle_key(self, keycode: str) -> bool:
        if keycode == "=":
            self.policy._handle_stand_command()
            return True
        return super().handle_key(keycode)


# ---------------------------------------------------------------------------
# WBT-specific
# ---------------------------------------------------------------------------


class WbtKeyboardOtherInput(KeyboardOtherInput):
    """Adds motion clip start (s) to base keyboard commands."""

    def handle_key(self, keycode: str) -> bool:
        if keycode == "s":
            self.policy._handle_start_motion_clip()
            return True
        return super().handle_key(keycode)
