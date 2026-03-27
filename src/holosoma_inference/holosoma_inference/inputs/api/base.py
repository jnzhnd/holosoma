"""Abstract base classes for input providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.commands import StateCommand

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


class StateCommandProvider(ABC):
    """Provides discrete commands to a policy via ``StateCommand`` enums.

    Implementations translate device-specific inputs (buttons, keys, messages)
    into ``StateCommand`` values.  The policy dispatches these commands via
    ``_dispatch_command()``.
    """

    def __init__(self, mapping: dict[str, StateCommand]):
        self._mapping = mapping

    def start(self) -> None:
        """Initialize the input source. Override if needed."""

    def poll(self) -> list[StateCommand]:
        """Return all commands detected this cycle."""
        return []
