"""Unit tests for input providers.

Tests cover:
- Command enums and device mappings
- Keyboard providers: queue-based poll(), key-to-command mapping, velocity tracking
- Joystick providers: button-to-command mapping, shared state wiring, edge detection
- ROS2 providers: callback-to-command mapping, velocity clamping
- Factory methods on BasePolicy / LocomotionPolicy / WBT
- DualMode command intercept and switching
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.api.commands import StateCommand, VelocityCommand
from holosoma_inference.inputs.impl.joystick import JOYSTICK_BASE, JOYSTICK_LOCOMOTION, JOYSTICK_WBT
from holosoma_inference.inputs.impl.keyboard import (
    KEYBOARD_BASE,
    KEYBOARD_LOCOMOTION,
    KEYBOARD_VELOCITY_LOCOMOTION,
    KEYBOARD_WBT,
)
from holosoma_inference.inputs.impl.ros2 import ROS2_COMMAND_MAP

# ---------------------------------------------------------------------------
# Fixtures: lightweight mock policy objects
# ---------------------------------------------------------------------------


def _make_policy(**overrides):
    """Build a minimal mock policy with all attributes providers touch."""
    p = MagicMock()
    p.lin_vel_command = np.array([[0.0, 0.0]])
    p.ang_vel_command = np.array([[0.0]])
    p.stand_command = np.array([[0]])
    p.base_height_command = np.array([[0.5]])
    p.desired_base_height = 0.5
    p.active_policy_index = 0
    p.model_paths = ["a.onnx", "b.onnx"]
    p.config = SimpleNamespace(
        task=SimpleNamespace(
            ros_cmd_vel_topic="cmd_vel",
            ros_command_provider_topic="holosoma/other_input",
        )
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture
def policy():
    return _make_policy()


# ============================================================================
# Command enum and mapping tests
# ============================================================================


class TestCommandMappings:
    def test_joystick_base_has_core_commands(self):
        assert JOYSTICK_BASE["A"] == StateCommand.START
        assert JOYSTICK_BASE["B"] == StateCommand.STOP
        assert JOYSTICK_BASE["Y"] == StateCommand.INIT
        assert JOYSTICK_BASE["L1+R1"] == StateCommand.KILL
        assert JOYSTICK_BASE["select"] == StateCommand.NEXT_POLICY

    def test_joystick_locomotion_extends_base(self):
        assert JOYSTICK_LOCOMOTION["A"] == StateCommand.START  # inherited
        assert JOYSTICK_LOCOMOTION["start"] == StateCommand.STAND_TOGGLE
        assert JOYSTICK_LOCOMOTION["L2"] == StateCommand.ZERO_VELOCITY

    def test_joystick_wbt_extends_base(self):
        assert JOYSTICK_WBT["A"] == StateCommand.START  # inherited
        assert JOYSTICK_WBT["start"] == StateCommand.START_MOTION_CLIP

    def test_keyboard_base_has_core_commands(self):
        assert KEYBOARD_BASE["]"] == StateCommand.START
        assert KEYBOARD_BASE["o"] == StateCommand.STOP
        assert KEYBOARD_BASE["i"] == StateCommand.INIT
        assert KEYBOARD_BASE["1"] == StateCommand.SWITCH_POLICY_1
        assert KEYBOARD_BASE["9"] == StateCommand.SWITCH_POLICY_9

    def test_keyboard_base_kp_commands(self):
        assert KEYBOARD_BASE["v"] == StateCommand.KP_DOWN_FINE
        assert KEYBOARD_BASE["b"] == StateCommand.KP_UP_FINE
        assert KEYBOARD_BASE["f"] == StateCommand.KP_DOWN
        assert KEYBOARD_BASE["g"] == StateCommand.KP_UP
        assert KEYBOARD_BASE["r"] == StateCommand.KP_RESET

    def test_keyboard_locomotion_extends_base(self):
        assert KEYBOARD_LOCOMOTION["]"] == StateCommand.START  # inherited
        assert KEYBOARD_LOCOMOTION["="] == StateCommand.STAND_TOGGLE
        assert KEYBOARD_LOCOMOTION["z"] == StateCommand.ZERO_VELOCITY

    def test_keyboard_locomotion_no_velocity_keys(self):
        """Velocity keys are now in KEYBOARD_VELOCITY_LOCOMOTION, not the command mapping."""
        for key in ("w", "a", "s", "d", "q", "e"):
            assert key not in KEYBOARD_LOCOMOTION

    def test_keyboard_velocity_locomotion_mapping(self):
        """KEYBOARD_VELOCITY_LOCOMOTION maps WASD/QE to (array_idx, col, delta)."""
        assert KEYBOARD_VELOCITY_LOCOMOTION["w"] == (0, 0, +0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["s"] == (0, 0, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["a"] == (0, 1, +0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["d"] == (0, 1, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["q"] == (1, 0, -0.1)
        assert KEYBOARD_VELOCITY_LOCOMOTION["e"] == (1, 0, +0.1)

    def test_keyboard_wbt_extends_base(self):
        assert KEYBOARD_WBT["]"] == StateCommand.START  # inherited
        assert KEYBOARD_WBT["s"] == StateCommand.START_MOTION_CLIP

    def test_ros2_command_map(self):
        assert ROS2_COMMAND_MAP["start"] == StateCommand.START
        assert ROS2_COMMAND_MAP["stop"] == StateCommand.STOP
        assert ROS2_COMMAND_MAP["init"] == StateCommand.INIT
        assert ROS2_COMMAND_MAP["walk"] == StateCommand.WALK
        assert ROS2_COMMAND_MAP["stand"] == StateCommand.STAND

    def test_velocity_keys_not_in_base_mapping(self):
        """Base keyboard mapping should NOT include velocity keys."""
        for key in ("w", "a", "s", "d", "q", "e", "z"):
            assert key not in KEYBOARD_BASE

    def test_velocity_keys_not_in_wbt_mapping(self):
        """WBT keyboard mapping should NOT include locomotion velocity keys."""
        for key in ("w", "a", "d", "q", "e", "z"):
            assert key not in KEYBOARD_WBT


# ============================================================================
# Keyboard providers
# ============================================================================


class TestKeyboardListener:
    def test_start_is_idempotent(self, policy):
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener

        # Remove _shared_hardware_source so listener doesn't skip
        del policy._shared_hardware_source
        listener = KeyboardListener(policy)
        listener.start()
        listener.start()  # second call should be a no-op
        assert listener._started is True

    def test_skips_thread_in_non_tty(self, policy, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener

        del policy._shared_hardware_source
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        policy.use_keyboard = True
        policy.use_policy_action = False

        listener = KeyboardListener(policy)
        listener.start()

        assert policy.use_keyboard is False
        assert policy.use_policy_action is True

    def test_ensure_skips_shared_hardware(self, policy):
        """_ensure_keyboard_listener is a no-op when _shared_hardware_source exists."""
        from holosoma_inference.inputs.impl.keyboard import _ensure_keyboard_listener

        policy._shared_hardware_source = MagicMock()
        # Remove auto-created _keyboard_listener from MagicMock so we can detect creation
        del policy._keyboard_listener
        _ensure_keyboard_listener(policy)
        assert not hasattr(policy, "_keyboard_listener")

    def test_ensure_creates_and_shares_listener(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener, _ensure_keyboard_listener

        p = _make_policy()
        # Remove auto-created attributes from MagicMock
        del p._shared_hardware_source
        del p._keyboard_listener
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        _ensure_keyboard_listener(p)
        assert isinstance(p._keyboard_listener, KeyboardListener)

        # Second call reuses the same listener
        first = p._keyboard_listener
        _ensure_keyboard_listener(p)
        assert p._keyboard_listener is first

    def test_broadcast_to_multiple_subscribers(self):
        """subscribe() creates independent queues; on_press broadcasts to all."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener

        p = _make_policy()
        del p._shared_hardware_source
        listener = KeyboardListener(p)
        q1 = listener.subscribe()
        q2 = listener.subscribe()

        # Simulate keypress broadcast
        for q in listener._subscribers:
            q.append("w")

        assert list(q1) == ["w"]
        assert list(q2) == ["w"]

        # Draining q1 doesn't affect q2
        q1.popleft()
        assert len(q1) == 0
        assert len(q2) == 1


class TestKeyboardStateCommandProvider:
    def _make_provider(self, mapping=None):
        from holosoma_inference.inputs.impl.keyboard import KeyboardStateCommandProvider

        queue = deque()
        prov = KeyboardStateCommandProvider(mapping or KEYBOARD_BASE, queue)
        return prov

    def test_poll_returns_mapped_commands(self):
        prov = self._make_provider()
        prov._queue.extend(["]", "o", "i"])
        commands = prov.poll()
        assert commands == [StateCommand.START, StateCommand.STOP, StateCommand.INIT]

    def test_poll_skips_unmapped_keys(self):
        prov = self._make_provider()
        prov._queue.extend(["x", "unknown", "]"])
        commands = prov.poll()
        assert commands == [StateCommand.START]

    def test_poll_drains_queue(self):
        prov = self._make_provider()
        prov._queue.append("]")
        assert prov.poll() == [StateCommand.START]
        assert prov.poll() == []  # second poll is empty

    def test_poll_kp_commands(self):
        prov = self._make_provider()
        prov._queue.extend(["v", "b", "f", "g", "r"])
        commands = prov.poll()
        assert commands == [
            StateCommand.KP_DOWN_FINE,
            StateCommand.KP_UP_FINE,
            StateCommand.KP_DOWN,
            StateCommand.KP_UP,
            StateCommand.KP_RESET,
        ]

    def test_poll_switch_policy(self):
        prov = self._make_provider()
        prov._queue.append("2")
        commands = prov.poll()
        assert commands == [StateCommand.SWITCH_POLICY_2]

    def test_poll_empty_queue_returns_empty(self):
        prov = self._make_provider()
        assert prov.poll() == []

    def test_locomotion_mapping(self):
        prov = self._make_provider(KEYBOARD_LOCOMOTION)
        prov._queue.extend(["=", "]", "z"])
        commands = prov.poll()
        assert commands == [
            StateCommand.STAND_TOGGLE,
            StateCommand.START,
            StateCommand.ZERO_VELOCITY,
        ]

    def test_wbt_mapping(self):
        prov = self._make_provider(KEYBOARD_WBT)
        prov._queue.extend(["s", "o"])
        commands = prov.poll()
        assert commands == [StateCommand.START_MOTION_CLIP, StateCommand.STOP]

    def test_shared_queue_between_instances(self):
        """Two KeyboardStateCommandProvider instances sharing a queue: one drains, other sees empty."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardStateCommandProvider

        shared_queue = deque()
        prov1 = KeyboardStateCommandProvider(KEYBOARD_BASE, shared_queue)
        prov2 = KeyboardStateCommandProvider(KEYBOARD_BASE, shared_queue)

        shared_queue.append("]")
        assert prov1.poll() == [StateCommand.START]
        assert prov2.poll() == []  # already drained


# ============================================================================
# Keyboard velocity input
# ============================================================================


class TestKeyboardVelocityInput:
    def test_poll_returns_velocity_command(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.append("w")
        vc = prov.poll()
        assert isinstance(vc, VelocityCommand)
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.1)
        np.testing.assert_almost_equal(vc.lin_vel[0, 1], 0.0)
        np.testing.assert_almost_equal(vc.ang_vel[0, 0], 0.0)

    def test_poll_accumulates_increments(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.extend(["w", "w", "a"])
        vc = prov.poll()
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.2)
        np.testing.assert_almost_equal(vc.lin_vel[0, 1], 0.1)

    def test_poll_angular_velocity(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.extend(["q", "e", "e"])
        vc = prov.poll()
        np.testing.assert_almost_equal(vc.ang_vel[0, 0], 0.1)  # -0.1 + 0.1 + 0.1

    def test_poll_returns_copy(self):
        """Returned arrays are copies, not references to internal state."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.append("w")
        vc1 = prov.poll()
        queue.append("w")
        vc2 = prov.poll()
        # vc1 should still show 0.1 (not 0.2)
        np.testing.assert_almost_equal(vc1.lin_vel[0, 0], 0.1)
        np.testing.assert_almost_equal(vc2.lin_vel[0, 0], 0.2)

    def test_zero_resets_state(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.extend(["w", "a", "e"])
        prov.poll()
        prov.zero()
        vc = prov.poll()
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.0)
        np.testing.assert_almost_equal(vc.lin_vel[0, 1], 0.0)
        np.testing.assert_almost_equal(vc.ang_vel[0, 0], 0.0)

    def test_no_velocity_keys_returns_none(self):
        """When no velocity keys mapping, poll() returns None and clears queue."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue)
        queue.append("w")
        assert prov.poll() is None
        assert len(queue) == 0  # queue cleared

    def test_unmapped_keys_ignored(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        queue = deque()
        prov = KeyboardVelocityInput(queue, KEYBOARD_VELOCITY_LOCOMOTION)
        queue.extend(["]", "o", "w"])  # ] and o are command keys, not velocity
        vc = prov.poll()
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.1)  # only w applied

    def test_broadcast_isolation(self):
        """Two providers on separate subscriber queues get independent events."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        q1 = deque()
        q2 = deque()
        vel = KeyboardVelocityInput(q1, KEYBOARD_VELOCITY_LOCOMOTION)
        cmd_queue = q2  # command provider would use this

        # Simulate broadcast
        q1.append("w")
        q2.append("w")

        vc = vel.poll()
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.1)
        assert len(q2) == 1  # command queue untouched


# ============================================================================
# Joystick providers
# ============================================================================


class TestJoystickVelocityInput:
    def test_poll_skips_when_no_msg(self, policy):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        policy.interface.get_joystick_msg.return_value = None
        prov = JoystickVelocityInput(policy)
        result = prov.poll()
        assert result is None
        policy.interface.process_joystick_input.assert_not_called()

    def test_poll_returns_velocity_command(self, policy):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        new_lin = np.array([[0.5, 0.0]])
        new_ang = np.array([[0.1]])
        new_keys = {"A": True}
        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.process_joystick_input.return_value = (new_lin, new_ang, new_keys)

        prov = JoystickVelocityInput(policy)
        vc = prov.poll()

        assert isinstance(vc, VelocityCommand)
        np.testing.assert_array_equal(vc.lin_vel, new_lin)
        np.testing.assert_array_equal(vc.ang_vel, new_ang)
        assert prov.key_states == {"A": True}

    def test_poll_preserves_last_key_states(self, policy):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.process_joystick_input.return_value = (
            policy.lin_vel_command,
            policy.ang_vel_command,
            {"A": True},
        )

        prov = JoystickVelocityInput(policy)
        prov.key_states = {"B": True}
        prov.poll()

        assert prov.last_key_states == {"B": True}
        assert prov.key_states == {"A": True}


class TestJoystickStateCommandProvider:
    def test_poll_multiple_rising_edges(self, policy):
        """Multiple buttons pressed simultaneously produce multiple commands."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True, "B": True, "Y": True}
        vel.last_key_states = {"A": False, "B": False, "Y": False}

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert set(commands) == {StateCommand.START, StateCommand.STOP, StateCommand.INIT}

    def test_poll_shared_edge_detection(self, policy):
        """When shared with velocity provider, returns commands for rising edges."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": False}

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == [StateCommand.START]

    def test_poll_shared_no_dispatch_on_hold(self, policy):
        """No commands when button was already held."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": True}

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == []

    def test_poll_standalone_reads_buttons(self, policy):
        """When not shared, reads buttons directly from SDK."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.get_joystick_key.return_value = "B"

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        commands = prov.poll()  # First poll: B goes True (rising edge)

        assert commands == [StateCommand.STOP]

    def test_poll_standalone_skips_when_no_msg(self, policy):
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        policy.interface.get_joystick_msg.return_value = None
        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        commands = prov.poll()
        assert commands == []

    def test_unmapped_button_not_in_commands(self, policy):
        """Buttons not in the mapping are silently ignored."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"UNKNOWN": True}
        vel.last_key_states = {"UNKNOWN": False}

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == []


# ============================================================================
# ROS2 providers
# ============================================================================


class TestRos2VelocityInput:
    def test_callback_stores_velocity(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=-0.3),
                angular=SimpleNamespace(z=0.8),
            )
        )
        prov._callback(msg)
        vc = prov.poll()

        assert isinstance(vc, VelocityCommand)
        np.testing.assert_almost_equal(vc.lin_vel[0, 0], 0.5)
        np.testing.assert_almost_equal(vc.lin_vel[0, 1], -0.3)
        np.testing.assert_almost_equal(vc.ang_vel[0, 0], 0.8)

    def test_callback_clamps_to_range(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=5.0, y=-5.0),
                angular=SimpleNamespace(z=99.0),
            )
        )
        prov._callback(msg)
        vc = prov.poll()

        assert vc.lin_vel[0, 0] == 1.0
        assert vc.lin_vel[0, 1] == -1.0
        assert vc.ang_vel[0, 0] == 1.0

    def test_poll_returns_copy(self, policy):
        """Returned VelocityCommand arrays are copies of internal state."""
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=0.0),
                angular=SimpleNamespace(z=0.0),
            )
        )
        prov._callback(msg)
        vc1 = prov.poll()
        vc2 = prov.poll()
        # Modify vc1, vc2 should be unaffected
        vc1.lin_vel[0, 0] = 999.0
        np.testing.assert_almost_equal(vc2.lin_vel[0, 0], 0.5)


class TestRos2StateCommandProvider:
    def test_known_commands_queued(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="start"))
        prov._callback(SimpleNamespace(data="stop"))
        prov._callback(SimpleNamespace(data="init"))

        commands = prov.poll()
        assert commands == [StateCommand.START, StateCommand.STOP, StateCommand.INIT]

    def test_walk_stand_commands(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="walk"))
        prov._callback(SimpleNamespace(data="stand"))

        commands = prov.poll()
        assert commands == [StateCommand.WALK, StateCommand.STAND]

    def test_unknown_command_warns(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="bogus"))
        policy.logger.warning.assert_called_once()
        assert prov.poll() == []

    def test_whitespace_and_case_normalization(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="  WALK  "))

        commands = prov.poll()
        assert commands == [StateCommand.WALK]

    def test_poll_drains_queue(self, policy):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="start"))
        assert prov.poll() == [StateCommand.START]
        assert prov.poll() == []  # second poll is empty


# ============================================================================
# Factory methods (BasePolicy / Locomotion / WBT)
# ============================================================================


def _try_import_policies():
    """Try to import policy modules; skip tests if heavy deps are missing."""
    try:
        from holosoma_inference.policies.base import BasePolicy  # noqa: F401
        from holosoma_inference.policies.locomotion import LocomotionPolicy  # noqa: F401
        from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_policies = _try_import_policies()
_skip_policies = pytest.mark.skipif(not _has_policies, reason="Policy deps not installed")


@_skip_policies
class TestBasePolicyFactory:
    """Test BasePolicy._create_velocity_input and _create_command_provider."""

    def _make_base(self, monkeypatch=None):
        from holosoma_inference.policies.base import BasePolicy

        bp = BasePolicy.__new__(BasePolicy)
        # Keyboard factory calls _ensure_keyboard_listener which needs logger
        if monkeypatch is not None:
            bp.logger = MagicMock()
            monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        return bp

    def test_keyboard_velocity(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        bp = self._make_base(monkeypatch)
        result = bp._create_velocity_input(InputSource.keyboard)
        assert isinstance(result, KeyboardVelocityInput)

    def test_joystick_velocity(self):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        bp = self._make_base()
        result = bp._create_velocity_input(InputSource.joystick)
        assert isinstance(result, JoystickVelocityInput)

    def test_ros2_velocity(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        bp = self._make_base()
        result = bp._create_velocity_input(InputSource.ros2)
        assert isinstance(result, Ros2VelocityInput)

    def test_keyboard_other(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardStateCommandProvider

        bp = self._make_base(monkeypatch)
        result = bp._create_command_provider(InputSource.keyboard)
        assert isinstance(result, KeyboardStateCommandProvider)
        assert result._mapping is KEYBOARD_BASE

    def test_joystick_other(self):
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        bp = self._make_base()
        result = bp._create_command_provider(InputSource.joystick)
        assert isinstance(result, JoystickStateCommandProvider)
        assert result._mapping is JOYSTICK_BASE

    def test_ros2_other(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        bp = self._make_base()
        result = bp._create_command_provider(InputSource.ros2)
        assert isinstance(result, Ros2StateCommandProvider)

    def test_unknown_source_raises(self, monkeypatch):
        bp = self._make_base(monkeypatch)
        with pytest.raises(ValueError, match="Unknown velocity"):
            bp._create_velocity_input("invalid")
        with pytest.raises(ValueError, match="Unknown command provider"):
            bp._create_command_provider("invalid")


@_skip_policies
class TestLocomotionPolicyFactory:
    """Test LocomotionPolicy overrides for keyboard/joystick providers."""

    def _make_loco(self, monkeypatch=None):
        from holosoma_inference.policies.locomotion import LocomotionPolicy

        lp = LocomotionPolicy.__new__(LocomotionPolicy)
        if monkeypatch is not None:
            lp.logger = MagicMock()
            monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        return lp

    def test_keyboard_velocity_has_locomotion_mapping(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        lp = self._make_loco(monkeypatch)
        result = lp._create_velocity_input(InputSource.keyboard)
        assert type(result) is KeyboardVelocityInput
        assert result._velocity_keys is KEYBOARD_VELOCITY_LOCOMOTION

    def test_keyboard_other_uses_locomotion_mapping(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardStateCommandProvider

        lp = self._make_loco(monkeypatch)
        result = lp._create_command_provider(InputSource.keyboard)
        assert isinstance(result, KeyboardStateCommandProvider)
        assert result._mapping is KEYBOARD_LOCOMOTION

    def test_joystick_other_uses_locomotion_mapping(self):
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        lp = self._make_loco()
        result = lp._create_command_provider(InputSource.joystick)
        assert isinstance(result, JoystickStateCommandProvider)
        assert result._mapping is JOYSTICK_LOCOMOTION

    def test_joystick_velocity_falls_to_base(self):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        lp = self._make_loco()
        result = lp._create_velocity_input(InputSource.joystick)
        assert type(result) is JoystickVelocityInput

    def test_ros2_falls_to_base(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider, Ros2VelocityInput

        lp = self._make_loco()
        assert isinstance(lp._create_velocity_input(InputSource.ros2), Ros2VelocityInput)
        assert isinstance(lp._create_command_provider(InputSource.ros2), Ros2StateCommandProvider)


@_skip_policies
class TestWbtPolicyFactory:
    """Test WholeBodyTrackingPolicy overrides for keyboard/joystick providers."""

    def _make_wbt(self, monkeypatch=None):
        from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

        wp = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
        if monkeypatch is not None:
            wp.logger = MagicMock()
            monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        return wp

    def test_keyboard_other_uses_wbt_mapping(self, monkeypatch):
        from holosoma_inference.inputs.impl.keyboard import KeyboardStateCommandProvider

        wp = self._make_wbt(monkeypatch)
        result = wp._create_command_provider(InputSource.keyboard)
        assert isinstance(result, KeyboardStateCommandProvider)
        assert result._mapping is KEYBOARD_WBT

    def test_joystick_other_uses_wbt_mapping(self):
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        wp = self._make_wbt()
        result = wp._create_command_provider(InputSource.joystick)
        assert isinstance(result, JoystickStateCommandProvider)
        assert result._mapping is JOYSTICK_WBT

    def test_keyboard_velocity_no_velocity_keys(self, monkeypatch):
        """WBT has no velocity keys — KeyboardVelocityInput returns None."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardVelocityInput

        wp = self._make_wbt(monkeypatch)
        result = wp._create_velocity_input(InputSource.keyboard)
        assert type(result) is KeyboardVelocityInput
        assert result._velocity_keys == {}

    def test_ros2_falls_to_base(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider, Ros2VelocityInput

        wp = self._make_wbt()
        assert isinstance(wp._create_velocity_input(InputSource.ros2), Ros2VelocityInput)
        assert isinstance(wp._create_command_provider(InputSource.ros2), Ros2StateCommandProvider)


# ============================================================================
# DualMode X/x switching
# ============================================================================


def _try_import_dual_mode():
    try:
        from holosoma_inference.policies.dual_mode import DualModePolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_dual_mode = _try_import_dual_mode()
_skip_dual_mode = pytest.mark.skipif(not _has_dual_mode, reason="DualMode deps not installed")


def _make_dual():
    """Build a DualModePolicy with mock policies, skipping __init__."""
    from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider
    from holosoma_inference.policies.dual_mode import DualModePolicy

    dual = object.__new__(DualModePolicy)
    dual.primary = _make_policy()
    dual.secondary = _make_policy()
    dual.active = dual.primary
    dual.active_label = "primary"

    dual.primary._velocity_input = MagicMock()
    dual.secondary._velocity_input = MagicMock()

    # Give both policies real JoystickStateCommandProvider with base mappings
    dual.primary._command_provider = JoystickStateCommandProvider(dual.primary, dict(JOYSTICK_BASE))
    dual.secondary._command_provider = JoystickStateCommandProvider(dual.secondary, dict(JOYSTICK_BASE))

    dual.primary._dispatch_command = MagicMock()
    dual.secondary._dispatch_command = MagicMock()

    dual._setup_command_intercept()
    return dual


@_skip_dual_mode
class TestDualModeSwitching:
    def test_switch_mode_injected_in_mappings(self):
        dual = _make_dual()
        assert dual.primary._command_provider._mapping["X"] == StateCommand.SWITCH_MODE
        assert dual.primary._command_provider._mapping["x"] == StateCommand.SWITCH_MODE
        assert dual.secondary._command_provider._mapping["X"] == StateCommand.SWITCH_MODE

    def test_x_joystick_triggers_switch(self):
        dual = _make_dual()
        assert dual.active is dual.primary
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.secondary

    def test_double_switch_returns_to_primary(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.secondary
        dual.secondary._dispatch_command(StateCommand.SWITCH_MODE)
        assert dual.active is dual.primary
        assert dual.active_label == "primary"

    def test_non_switch_command_delegates_to_active(self):
        dual = _make_dual()
        orig_dispatch = dual._orig_dispatch[id(dual.primary)]
        dual.primary._dispatch_command(StateCommand.START)
        orig_dispatch.assert_called_once_with(StateCommand.START)

    def test_delegates_to_secondary_after_switch(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)  # switch to secondary
        orig_secondary = dual._orig_dispatch[id(dual.secondary)]
        dual.secondary._dispatch_command(StateCommand.START)
        orig_secondary.assert_called_once_with(StateCommand.START)

    def test_switch_stops_old_and_starts_new(self):
        dual = _make_dual()
        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)
        dual.primary._handle_stop_policy.assert_called_once()
        dual.secondary._resolve_control_gains.assert_called_once()
        dual.secondary._init_phase_components.assert_called_once()
        dual.secondary._handle_start_policy.assert_called_once()

    def test_joystick_state_carry_over(self):
        from holosoma_inference.inputs.impl.joystick import JoystickVelocityInput

        dual = _make_dual()
        # Replace mock velocity inputs with real JoystickVelocityInput
        pri_vel = JoystickVelocityInput(dual.primary)
        pri_vel.key_states = {"X": True, "A": False}
        sec_vel = JoystickVelocityInput(dual.secondary)

        dual.primary._velocity_input = pri_vel
        dual.secondary._velocity_input = sec_vel

        dual.primary._dispatch_command(StateCommand.SWITCH_MODE)

        assert sec_vel.key_states == {"X": True, "A": False}
        assert sec_vel.last_key_states == {"X": True, "A": False}


@_skip_dual_mode
class TestDualModeKeyboardQueueWiring:
    """Test that both policies get independent subscriber queues via broadcast."""

    def test_broadcast_queues_are_independent(self):
        """Each policy's KeyboardStateCommandProvider gets its own subscriber queue."""
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener, KeyboardStateCommandProvider
        from holosoma_inference.policies.dual_mode import DualModePolicy

        dual = object.__new__(DualModePolicy)
        dual.primary = _make_policy()
        dual.secondary = _make_policy()
        dual.active = dual.primary
        dual.active_label = "primary"

        dual.primary._velocity_input = MagicMock()
        dual.secondary._velocity_input = MagicMock()

        # Simulate broadcast pattern: each provider gets its own subscriber queue
        listener = KeyboardListener(dual.primary)
        q1 = listener.subscribe()
        q2 = listener.subscribe()
        dual.primary._keyboard_listener = listener
        dual.primary._command_provider = KeyboardStateCommandProvider(dict(KEYBOARD_BASE), q1)
        dual.secondary._command_provider = KeyboardStateCommandProvider(dict(KEYBOARD_BASE), q2)

        dual.primary._dispatch_command = MagicMock()
        dual.secondary._dispatch_command = MagicMock()

        dual._setup_command_intercept()

        # Queues are independent
        assert dual.primary._command_provider._queue is not dual.secondary._command_provider._queue

    def test_keyboard_commands_reach_active_via_poll(self):
        from holosoma_inference.inputs.impl.keyboard import KeyboardListener, KeyboardStateCommandProvider
        from holosoma_inference.policies.dual_mode import DualModePolicy

        dual = object.__new__(DualModePolicy)
        dual.primary = _make_policy()
        dual.secondary = _make_policy()
        dual.active = dual.primary
        dual.active_label = "primary"

        dual.primary._velocity_input = MagicMock()
        dual.secondary._velocity_input = MagicMock()

        listener = KeyboardListener(dual.primary)
        q1 = listener.subscribe()
        q2 = listener.subscribe()
        dual.primary._keyboard_listener = listener
        dual.primary._command_provider = KeyboardStateCommandProvider(dict(KEYBOARD_BASE), q1)
        dual.secondary._command_provider = KeyboardStateCommandProvider(dict(KEYBOARD_BASE), q2)

        dual.primary._dispatch_command = MagicMock()
        dual.secondary._dispatch_command = MagicMock()

        dual._setup_command_intercept()

        # Simulate broadcast keypress
        for q in listener._subscribers:
            q.append("]")

        # Active policy (primary) drains its own queue
        commands = dual.active._command_provider.poll()
        assert commands == [StateCommand.START]

        # Secondary's queue still has the event (independent)
        commands2 = dual.secondary._command_provider.poll()
        assert commands2 == [StateCommand.START]


# ============================================================================
# Shared joystick state wiring
# ============================================================================


class TestSharedJoystickWiring:
    def test_shared_velocity_none_by_default(self, policy):
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        other = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        assert other._shared_velocity is None


# ============================================================================
# Separation guarantee: wrong-channel keys are not handled
# ============================================================================


class TestChannelSeparation:
    """Velocity keys only appear in KEYBOARD_VELOCITY_LOCOMOTION, not command mappings."""

    def test_velocity_keys_not_in_base_keyboard_mapping(self):
        """BasePolicy keyboard mapping has no velocity keys."""
        for key in ("w", "a", "s", "d", "q", "e", "z"):
            assert key not in KEYBOARD_BASE

    def test_velocity_keys_in_velocity_mapping(self):
        """Velocity keys are in the dedicated velocity mapping."""
        assert "w" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "a" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "d" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "q" in KEYBOARD_VELOCITY_LOCOMOTION
        assert "e" in KEYBOARD_VELOCITY_LOCOMOTION

    def test_joystick_other_ignores_unmapped_buttons(self, policy):
        """JoystickStateCommandProvider only returns commands for mapped buttons."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"unknown_stick": True}
        vel.last_key_states = {"unknown_stick": False}

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        assert prov.poll() == []


# ============================================================================
# Edge cases and error paths
# ============================================================================


class TestRos2VelocityEdgeCases:
    def test_callback_clamps_negative_angular(self, policy):
        """Angular velocity is clamped at both ends."""
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0),
                angular=SimpleNamespace(z=-99.0),
            )
        )
        prov._callback(msg)
        vc = prov.poll()
        assert vc.ang_vel[0, 0] == -1.0

    def test_callback_exact_boundary_values(self, policy):
        """Values exactly at +/-1.0 pass through unchanged."""
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=1.0, y=-1.0),
                angular=SimpleNamespace(z=1.0),
            )
        )
        prov._callback(msg)
        vc = prov.poll()
        assert vc.lin_vel[0, 0] == 1.0
        assert vc.lin_vel[0, 1] == -1.0
        assert vc.ang_vel[0, 0] == 1.0

    def test_callback_zero_passes_through(self, policy):
        """Zero velocity is not clamped or modified."""
        from holosoma_inference.inputs.impl.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0),
                angular=SimpleNamespace(z=0.0),
            )
        )
        prov._callback(msg)
        vc = prov.poll()
        assert vc.lin_vel[0, 0] == 0.0
        assert vc.lin_vel[0, 1] == 0.0
        assert vc.ang_vel[0, 0] == 0.0


class TestRos2StateCommandProviderEdgeCases:
    def test_empty_string_warns(self, policy):
        """Empty/whitespace-only input is treated as unknown command."""
        from holosoma_inference.inputs.impl.ros2 import Ros2StateCommandProvider

        prov = Ros2StateCommandProvider(policy)
        prov._callback(SimpleNamespace(data="   "))
        policy.logger.warning.assert_called_once()
        assert prov.poll() == []


class TestJoystickStandaloneEdgeCases:
    def test_poll_standalone_button_release_clears_states(self, policy):
        """When no button is pressed, all key states go False."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        policy.interface.get_joystick_msg.return_value = "msg"
        # First poll: press B
        policy.interface.get_joystick_key.return_value = "B"
        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        prov.poll()  # rising edge on B

        # Second poll: no button pressed
        policy.interface.get_joystick_key.return_value = ""
        commands = prov.poll()
        assert commands == []  # no rising edges
        # B should now be False
        assert prov._key_states.get("B") is False

    def test_poll_standalone_repeated_same_button_no_duplicate(self, policy):
        """Holding a button across polls should not re-fire the command."""
        from holosoma_inference.inputs.impl.joystick import JoystickStateCommandProvider

        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.get_joystick_key.return_value = "A"

        prov = JoystickStateCommandProvider(policy, JOYSTICK_BASE)
        first = prov.poll()
        assert first == [StateCommand.START]  # rising edge

        second = prov.poll()
        assert second == []  # held, no rising edge


# ============================================================================
# VelocityCommand dataclass
# ============================================================================


class TestVelocityCommand:
    def test_frozen(self):
        vc = VelocityCommand(np.zeros((1, 2)), np.zeros((1, 1)))
        with pytest.raises(AttributeError):
            vc.lin_vel = np.ones((1, 2))

    def test_equality(self):
        a = VelocityCommand(np.array([[1.0, 2.0]]), np.array([[3.0]]))
        b = VelocityCommand(np.array([[1.0, 2.0]]), np.array([[3.0]]))
        # frozen dataclass with numpy arrays — identity differs but values match
        np.testing.assert_array_equal(a.lin_vel, b.lin_vel)
        np.testing.assert_array_equal(a.ang_vel, b.ang_vel)
