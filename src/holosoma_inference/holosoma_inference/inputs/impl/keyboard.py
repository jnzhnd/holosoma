"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from collections import deque
from typing import TYPE_CHECKING

from sshkeyboard import listen_keyboard

from holosoma_inference.inputs.api.base import StateCommandProvider, VelocityInput
from holosoma_inference.inputs.api.commands import StateCommand

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy

# ---------------------------------------------------------------------------
# Keyboard mappings
# ---------------------------------------------------------------------------

KEYBOARD_BASE: dict[str, StateCommand] = {
    "]": StateCommand.START,
    "o": StateCommand.STOP,
    "i": StateCommand.INIT,
    "v": StateCommand.KP_DOWN_FINE,
    "b": StateCommand.KP_UP_FINE,
    "f": StateCommand.KP_DOWN,
    "g": StateCommand.KP_UP,
    "r": StateCommand.KP_RESET,
    **{str(n): StateCommand[f"SWITCH_POLICY_{n}"] for n in range(1, 10)},
}

KEYBOARD_LOCOMOTION: dict[str, StateCommand] = {
    **KEYBOARD_BASE,
    "=": StateCommand.STAND_TOGGLE,
    "z": StateCommand.ZERO_VELOCITY,
    "w": StateCommand.VEL_FORWARD,
    "s": StateCommand.VEL_BACKWARD,
    "a": StateCommand.VEL_LEFT,
    "d": StateCommand.VEL_RIGHT,
    "q": StateCommand.ANG_VEL_LEFT,
    "e": StateCommand.ANG_VEL_RIGHT,
}

KEYBOARD_WBT: dict[str, StateCommand] = {
    **KEYBOARD_BASE,
    "s": StateCommand.START_MOTION_CLIP,
}


class KeyboardListener:
    """Shared sshkeyboard listener thread.

    Created lazily by keyboard providers and stored on the policy as
    ``_keyboard_listener``.  Multiple providers share one instance;
    ``start()`` is idempotent.  Keypresses are queued for the main loop
    to drain via ``KeyboardStateCommandProvider.poll()``.
    """

    def __init__(self, policy: BasePolicy) -> None:
        self._policy = policy
        self._started = False
        self._queue: deque[str] = deque()

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
        def on_press(keycode):
            self._queue.append(keycode)

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
    """No-op keyboard velocity input.

    Velocity for keyboard is handled via command enums in the
    StateCommandProvider mapping (e.g. StateCommand.VEL_FORWARD for the 'w' key).
    This class exists only to start the shared keyboard listener.
    """

    def start(self) -> None:
        _ensure_keyboard_listener(self.policy)


class KeyboardStateCommandProvider(StateCommandProvider):
    """Keyboard handler for discrete commands using a mapping dict.

    The shared ``KeyboardListener`` queues raw keycodes.  ``poll()`` drains
    the queue, maps each keycode via the mapping dict, and returns the
    resulting command enums — same pull pattern as joystick.
    """

    def __init__(self, mapping: dict[str, StateCommand], queue: deque[str]) -> None:
        super().__init__(mapping)
        self._queue = queue

    def poll(self) -> list[StateCommand]:
        commands: list[StateCommand] = []
        while True:
            try:
                keycode = self._queue.popleft()
            except IndexError:
                break
            cmd = self._mapping.get(keycode)
            if cmd is not None:
                commands.append(cmd)
        return commands
