"""Robot state dataclass for unified state representation across interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RobotState:
    """
    Unified robot state representation.

    This dataclass replaces the brittle numpy array format previously returned by
    get_low_state(). All fields use consistent naming and provide clear semantics.

    Required Fields (all interfaces must provide these):
        base_position: Base position in world frame (3,)
        base_orientation: Base orientation as quaternion wxyz (4,)
        joint_positions: Joint positions in radians (N,)
        base_linear_velocity: Base linear velocity in world frame (3,)
        base_angular_velocity: Base angular velocity in body frame (3,)
        joint_velocities: Joint velocities in rad/s (N,)

    Optional Fields (interface-specific, may be None):
        projected_gravity: Gravity vector projected into body frame (3,)
        joint_torques: Joint torques in Nm (N,)
        base_linear_acceleration: Base linear acceleration (3,)
        base_angular_acceleration: Base angular acceleration (3,)
        joint_accelerations: Joint accelerations in rad/s^2 (N,)
        timestamp: Timestamp in seconds (float)
    """

    # Required fields - all interfaces must provide these
    base_position: np.ndarray
    base_orientation: np.ndarray  # quaternion wxyz
    joint_positions: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    joint_velocities: np.ndarray

    # Optional fields - interface-specific
    projected_gravity: np.ndarray | None = None
    joint_torques: np.ndarray | None = None
    base_linear_acceleration: np.ndarray | None = None
    base_angular_acceleration: np.ndarray | None = None
    joint_accelerations: np.ndarray | None = None
    timestamp: float | None = None

    def to_array(self, include_gravity: bool = False) -> np.ndarray:
        """
        Convert to legacy numpy array format for backward compatibility.

        Args:
            include_gravity: If True and projected_gravity is available,
                           append it to the array.

        Returns:
            np.ndarray with shape (1, 3+4+N+3+3+N[+3]) containing:
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N), [gravity(3)]]
        """
        components = [
            self.base_position,
            self.base_orientation,
            self.joint_positions,
            self.base_linear_velocity,
            self.base_angular_velocity,
            self.joint_velocities,
        ]
        if include_gravity and self.projected_gravity is not None:
            components.append(self.projected_gravity)

        return np.concatenate(components).reshape(1, -1)

    @property
    def num_joints(self) -> int:
        """Return the number of joints."""
        return len(self.joint_positions)
