"""ROS2 input providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.inputs.base import OtherInput, VelocityInput

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


class Ros2VelocityInput(VelocityInput):
    """Subscribes to ROS2 TwistStamped topic for velocity commands."""

    def __init__(self, policy: BasePolicy):
        super().__init__(policy)

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
    """Subscribes to ROS2 String topic for discrete commands."""

    def __init__(self, policy: BasePolicy):
        super().__init__(policy)

    def start(self) -> None:
        self.policy._init_ros_node()
        from std_msgs.msg import String

        topic = self.policy.config.task.ros_other_input_topic
        self.policy.node.create_subscription(String, topic, self._callback, 10)
        self.policy.logger.info(f"Subscribed to ROS2 other_input topic: {topic}")

    def _callback(self, msg):
        """Handle discrete commands from ROS2 other_input topic."""
        cmd = msg.data.strip().lower()
        if cmd == "walk":
            self.policy.stand_command[0, 0] = 1
            self.policy.base_height_command[0, 0] = self.policy.desired_base_height
            self.policy.logger.info("ROS2 command: walk")
        elif cmd == "stand":
            self.policy.stand_command[0, 0] = 0
            self.policy.logger.info("ROS2 command: stand")
        elif cmd == "start":
            self.policy._handle_start_policy()
        elif cmd == "stop":
            self.policy._handle_stop_policy()
        elif cmd == "init":
            self.policy._handle_init_state()
        else:
            self.policy.logger.warning(f"ROS2 command: unknown command '{cmd}'")
