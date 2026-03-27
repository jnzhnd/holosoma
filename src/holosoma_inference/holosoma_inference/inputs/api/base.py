"""Abstract base classes for input providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.commands import StateCommand, VelocityCommand

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class VelocityInput(ABC):
    """Provides absolute velocity state each cycle.

    Implementations read from their device (joystick sticks, keyboard
    increments, ROS2 topic) and return a ``VelocityCommand`` with the
    current absolute velocity, or ``None`` if no update is available.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize the input source (start threads, subscribe to topics, etc.)."""

    def poll(self) -> VelocityCommand | None:
        """Return current velocity, or None if this source has no update."""
        return None

    def zero(self) -> None:
        """Reset internal velocity state to zero. Override for stateful sources (keyboard)."""


class StateCommandProvider(ABC):
    """Provides discrete state commands each cycle.

    Implementations read from their device (keyboard keys, joystick buttons,
    ROS2 topic) and return a list of ``StateCommand`` enums representing
    user intent.
    """

    def __init__(self, mapping: dict[str, StateCommand]) -> None:
        self._mapping = mapping

    def start(self) -> None:
        """Initialize the input source."""

    def poll(self) -> list[StateCommand]:
        """Return commands accumulated since last poll."""
        return []
