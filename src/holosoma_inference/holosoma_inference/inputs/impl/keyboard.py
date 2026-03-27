"""Keyboard input providers and shared listener."""

from __future__ import annotations

import sys
import threading
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from sshkeyboard import listen_keyboard

from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy

# ---------------------------------------------------------------------------
# Keyboard command mappings (discrete commands)
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
}

KEYBOARD_WBT: dict[str, StateCommand] = {
    **KEYBOARD_BASE,
    "s": StateCommand.START_MOTION_CLIP,
}

# ---------------------------------------------------------------------------
# Keyboard velocity mappings (continuous velocity increments)
#
# Each entry maps a keycode to (array_index, column, delta):
#   array_index 0 = lin_vel, 1 = ang_vel
#   column = which element within that array
#   delta = increment per keypress
# ---------------------------------------------------------------------------

KEYBOARD_VELOCITY_LOCOMOTION: dict[str, tuple[int, int, float]] = {
    "w": (0, 0, +0.1),  # lin_vel[0, 0] += 0.1
    "s": (0, 0, -0.1),  # lin_vel[0, 0] -= 0.1
    "a": (0, 1, +0.1),  # lin_vel[0, 1] += 0.1
    "d": (0, 1, -0.1),  # lin_vel[0, 1] -= 0.1
    "q": (1, 0, -0.1),  # ang_vel[0, 0] -= 0.1
    "e": (1, 0, +0.1),  # ang_vel[0, 0] += 0.1
}


class KeyboardListener:
    """Shared sshkeyboard listener thread.

    Created lazily by the factory and stored on the policy as
    ``_keyboard_listener``.  Multiple providers share one instance;
    ``start()`` is idempotent.  Keypresses are broadcast to all
    subscriber queues.
    """

    def __init__(self, policy: BasePolicy) -> None:
        self._policy = policy
        self._started = False
        self._subscribers: list[deque[str]] = []

    def subscribe(self) -> deque[str]:
        """Create and return a new subscriber queue. All keypresses are broadcast to it."""
        q: deque[str] = deque()
        self._subscribers.append(q)
        return q

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
            for q in self._subscribers:
                q.append(keycode)

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


class KeyboardVelCmdProvider(VelCmdProvider):
    """Tracks keyboard velocity increments and returns absolute velocity.

    Subscribes to its own keyboard queue. Maps velocity keycodes (WASD/QE)
    to increments on internal lin_vel/ang_vel state. Returns a
    ``VelocityCommand`` with current absolute values each cycle.

    If no velocity_keys mapping is provided, always returns None (no-op).
    """

    def __init__(
        self,
        queue: deque[str],
        velocity_keys: dict[str, tuple[int, int, float]] | None = None,
    ) -> None:
        self._queue = queue
        self._velocity_keys = velocity_keys or {}
        self._lin_vel = np.zeros((1, 2))
        self._ang_vel = np.zeros((1, 1))

    def start(self) -> None:
        pass  # Listener already started by factory

    def poll(self) -> VelCmd | None:
        if not self._velocity_keys:
            self._queue.clear()
            return None

        while True:
            try:
                keycode = self._queue.popleft()
            except IndexError:
                break
            action = self._velocity_keys.get(keycode)
            if action is not None:
                array_idx, col, delta = action
                if array_idx == 0:
                    self._lin_vel[0, col] += delta
                else:
                    self._ang_vel[0, col] += delta

        return VelCmd(
            (float(self._lin_vel[0, 0]), float(self._lin_vel[0, 1])),
            float(self._ang_vel[0, 0]),
        )

    def zero(self) -> None:
        """Reset velocity state to zero."""
        self._lin_vel[:] = 0.0
        self._ang_vel[:] = 0.0


class KeyboardStateCommandProvider(StateCommandProvider):
    """Keyboard handler for discrete commands using a mapping dict.

    Subscribes to its own keyboard queue. ``poll()`` drains the queue,
    maps each keycode via the mapping dict, and returns the resulting
    command enums — same pull pattern as joystick.
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
