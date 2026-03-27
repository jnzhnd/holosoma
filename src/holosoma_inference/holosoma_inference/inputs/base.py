"""Abstract base classes for input providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class VelocityInput(ABC):
    """Provides continuous velocity commands (lin_vel, ang_vel) to a policy.

    Implementations write directly to policy.lin_vel_command and
    policy.ang_vel_command via their stored policy reference.
    """

    def __init__(self, policy: BasePolicy):
        self.policy = policy

    @abstractmethod
    def start(self) -> None:
        """Initialize the input source (start threads, subscribe to topics, etc.)."""

    def poll(self) -> None:
        """Called each loop iteration. Override for polled sources (joystick)."""

    def handle_key(self, keycode: str) -> bool:
        """Handle a keyboard keypress. Return True if consumed."""
        return False


class OtherInput(ABC):
    """Provides discrete commands to a policy via command enums.

    Implementations translate device-specific inputs (buttons, keys, messages)
    into command enums defined in ``holosoma_inference.inputs.commands``.
    The policy dispatches these commands via ``_dispatch_command()``.
    """

    def __init__(self, policy: BasePolicy, mapping: dict[str, Enum]):
        self.policy = policy
        self._mapping = mapping

    @abstractmethod
    def start(self) -> None:
        """Initialize the input source."""

    def poll(self) -> list[Enum]:
        """Return all commands detected this cycle."""
        return []

    def map_key(self, keycode: str) -> Enum | None:
        """Map a keyboard keycode to a command, or None if unmapped."""
        return self._mapping.get(keycode)
