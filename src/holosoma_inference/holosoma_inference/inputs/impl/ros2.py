"""ROS2 input providers."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

from holosoma_inference.inputs.api.base import OtherInput, VelocityInput
from holosoma_inference.inputs.api.commands import ROS2_COMMAND_MAP

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class Ros2VelocityInput(VelocityInput):
    """Subscribes to ROS2 TwistStamped topic for velocity commands."""

    def start(self) -> None:
        self.policy._init_ros_node()
        from geometry_msgs.msg import TwistStamped

        topic = self.policy.config.task.ros_cmd_vel_topic
        self.policy.node.create_subscription(TwistStamped, topic, self._callback, 10)
        self.policy.logger.info(f"Subscribed to ROS2 velocity topic: {topic}")

    def _callback(self, msg):
        """Write velocity commands from ROS2. Clamps to training range."""
        self.policy.lin_vel_command[0, 0] = max(-1.0, min(1.0, msg.twist.linear.x))
        self.policy.lin_vel_command[0, 1] = max(-1.0, min(1.0, msg.twist.linear.y))
        self.policy.ang_vel_command[0, 0] = max(-1.0, min(1.0, msg.twist.angular.z))


class Ros2OtherInput(OtherInput):
    """Subscribes to ROS2 String topic for discrete commands.

    Incoming string commands are mapped to enum values via ``ROS2_COMMAND_MAP``
    and queued.  The main loop drains them via ``poll()``.
    """

    def __init__(self, policy: BasePolicy):
        super().__init__({})  # ROS2 uses its own string-to-command map
        self.policy = policy
        self._queue: deque[Enum] = deque()

    def start(self) -> None:
        self.policy._init_ros_node()
        from std_msgs.msg import String

        topic = self.policy.config.task.ros_other_input_topic
        self.policy.node.create_subscription(String, topic, self._callback, 10)
        self.policy.logger.info(f"Subscribed to ROS2 other_input topic: {topic}")

    def _callback(self, msg):
        """Map ROS2 string command to enum and queue it."""
        cmd_str = msg.data.strip().lower()
        cmd = ROS2_COMMAND_MAP.get(cmd_str)
        if cmd is not None:
            self._queue.append(cmd)
        else:
            self.policy.logger.warning(f"ROS2 command: unknown command '{cmd_str}'")

    def poll(self) -> list[Enum]:
        """Drain all queued commands."""
        commands: list[Enum] = []
        while True:
            try:
                commands.append(self._queue.popleft())
            except IndexError:
                break
        return commands
