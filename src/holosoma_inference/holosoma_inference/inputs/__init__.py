from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.impl.interface import InterfaceInput
from holosoma_inference.inputs.impl.joystick import JOYSTICK_BASE
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_BASE, KeyboardInput, _ensure_keyboard_listener
from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider, Ros2VelCmdProvider

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


def create_input(policy: BasePolicy, source: InputSource, role: str):
    """Create an input provider for the given source and role ("velocity" or "command")."""
    if not policy.use_joystick and source in (InputSource.interface, InputSource.joystick):
        source = InputSource.keyboard

    if source in (InputSource.interface, InputSource.joystick):
        return InterfaceInput(policy.interface, policy._joystick_command_mapping or JOYSTICK_BASE)

    if source == InputSource.keyboard:
        _ensure_keyboard_listener(policy)
        listener = getattr(policy, "_keyboard_listener", None)
        if listener is None and hasattr(policy, "_shared_hardware_source"):
            listener = getattr(policy._shared_hardware_source, "_keyboard_listener", None)
        queue = listener.subscribe() if listener else deque()
        mapping = policy._keyboard_command_mapping or KEYBOARD_BASE
        return KeyboardInput(mapping, queue, policy._keyboard_velocity_mapping)

    if source == InputSource.ros2:
        if role == "velocity":
            return Ros2VelCmdProvider(policy)
        return Ros2StateCommandProvider(policy)

    raise ValueError(f"Unknown input source: {source}")
