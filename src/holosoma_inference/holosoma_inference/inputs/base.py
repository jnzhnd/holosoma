"""Abstract base classes for input providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    """Provides discrete commands (start/stop, walk/stand, kp tuning) to a policy.

    Implementations call policy methods like _handle_start_policy(),
    _handle_stand_command(), etc.
    """

    def __init__(self, policy: BasePolicy):
        self.policy = policy

    @abstractmethod
    def start(self) -> None:
        """Initialize the input source."""

    def poll(self) -> None:
        """Called each loop iteration. Override for polled sources (joystick)."""

    def handle_key(self, keycode: str) -> bool:
        """Handle a keyboard keypress. Return True if consumed."""
        return False

    def handle_joystick_button(self, key: str) -> bool:
        """Handle a joystick button press. Return True if consumed."""
        return False
