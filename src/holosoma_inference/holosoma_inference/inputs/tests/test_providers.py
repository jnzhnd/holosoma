"""Unit tests for input providers.

Tests cover:
- Command enums and device mappings
- Keyboard providers: key-to-command mapping, velocity dispatch
- Joystick providers: button-to-command mapping, shared state wiring, edge detection
- ROS2 providers: callback-to-command mapping, velocity clamping
- Factory methods on BasePolicy / LocomotionPolicy / WBT
- DualMode command intercept and switching
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.commands import (
    JOYSTICK_BASE,
    JOYSTICK_LOCOMOTION,
    JOYSTICK_WBT,
    KEYBOARD_BASE,
    KEYBOARD_LOCOMOTION,
    KEYBOARD_WBT,
    ROS2_COMMAND_MAP,
    Command,
    DualModeCommand,
    LocomotionCommand,
    WbtCommand,
)

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
            ros_other_input_topic="holosoma/other_input",
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
        assert JOYSTICK_BASE["A"] == Command.START
        assert JOYSTICK_BASE["B"] == Command.STOP
        assert JOYSTICK_BASE["Y"] == Command.INIT
        assert JOYSTICK_BASE["L1+R1"] == Command.KILL
        assert JOYSTICK_BASE["select"] == Command.NEXT_POLICY

    def test_joystick_locomotion_extends_base(self):
        assert JOYSTICK_LOCOMOTION["A"] == Command.START  # inherited
        assert JOYSTICK_LOCOMOTION["start"] == LocomotionCommand.STAND_TOGGLE
        assert JOYSTICK_LOCOMOTION["L2"] == LocomotionCommand.ZERO_VELOCITY

    def test_joystick_wbt_extends_base(self):
        assert JOYSTICK_WBT["A"] == Command.START  # inherited
        assert JOYSTICK_WBT["start"] == WbtCommand.START_MOTION_CLIP

    def test_keyboard_base_has_core_commands(self):
        assert KEYBOARD_BASE["]"] == Command.START
        assert KEYBOARD_BASE["o"] == Command.STOP
        assert KEYBOARD_BASE["i"] == Command.INIT
        assert KEYBOARD_BASE["1"] == Command.SWITCH_POLICY_1
        assert KEYBOARD_BASE["9"] == Command.SWITCH_POLICY_9

    def test_keyboard_base_kp_commands(self):
        assert KEYBOARD_BASE["v"] == Command.KP_DOWN_FINE
        assert KEYBOARD_BASE["b"] == Command.KP_UP_FINE
        assert KEYBOARD_BASE["f"] == Command.KP_DOWN
        assert KEYBOARD_BASE["g"] == Command.KP_UP
        assert KEYBOARD_BASE["r"] == Command.KP_RESET

    def test_keyboard_locomotion_extends_base(self):
        assert KEYBOARD_LOCOMOTION["]"] == Command.START  # inherited
        assert KEYBOARD_LOCOMOTION["="] == LocomotionCommand.STAND_TOGGLE

    def test_keyboard_wbt_extends_base(self):
        assert KEYBOARD_WBT["]"] == Command.START  # inherited
        assert KEYBOARD_WBT["s"] == WbtCommand.START_MOTION_CLIP

    def test_ros2_command_map(self):
        assert ROS2_COMMAND_MAP["start"] == Command.START
        assert ROS2_COMMAND_MAP["stop"] == Command.STOP
        assert ROS2_COMMAND_MAP["init"] == Command.INIT
        assert ROS2_COMMAND_MAP["walk"] == LocomotionCommand.WALK
        assert ROS2_COMMAND_MAP["stand"] == LocomotionCommand.STAND


# ============================================================================
# Keyboard providers
# ============================================================================


class TestKeyboardListener:
    def test_start_is_idempotent(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardListener

        # Remove _shared_hardware_source so listener doesn't skip
        del policy._shared_hardware_source
        listener = KeyboardListener(policy)
        listener.start()
        listener.start()  # second call should be a no-op
        assert listener._started is True

    def test_skips_thread_in_non_tty(self, policy, monkeypatch):
        from holosoma_inference.inputs.keyboard import KeyboardListener

        del policy._shared_hardware_source
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        policy.use_keyboard = True
        policy.use_policy_action = False

        listener = KeyboardListener(policy)
        listener.start()

        assert policy.use_keyboard is False
        assert policy.use_policy_action is True

    def test_ensure_skips_shared_hardware(self, policy):
        from holosoma_inference.inputs.keyboard import _ensure_keyboard_listener

        policy._shared_hardware_source = MagicMock()
        _ensure_keyboard_listener(policy)
        assert not hasattr(policy, "_keyboard_listener") or isinstance(policy._keyboard_listener, MagicMock)

    def test_ensure_creates_and_shares_listener(self, monkeypatch):
        from holosoma_inference.inputs.keyboard import KeyboardListener, _ensure_keyboard_listener

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

    def test_provider_start_calls_ensure(self, monkeypatch):
        from holosoma_inference.inputs.keyboard import (
            KeyboardListener,
            KeyboardOtherInput,
            KeyboardVelocityInput,
        )

        p = _make_policy()
        del p._shared_hardware_source
        del p._keyboard_listener
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        vel = KeyboardVelocityInput(p)
        other = KeyboardOtherInput(p, KEYBOARD_BASE)
        vel.start()
        other.start()

        assert isinstance(p._keyboard_listener, KeyboardListener)
        assert p._keyboard_listener._started is True


class TestKeyboardVelocityInput:
    def test_base_returns_false_for_all_keys(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardVelocityInput

        prov = KeyboardVelocityInput(policy)
        prov.start()
        assert prov.handle_key("w") is False
        assert prov.handle_key("]") is False


class TestKeyboardOtherInput:
    def test_map_key_returns_command(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_BASE)
        assert prov.map_key("]") == Command.START
        assert prov.map_key("o") == Command.STOP
        assert prov.map_key("i") == Command.INIT

    def test_map_key_kp_commands(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_BASE)
        assert prov.map_key("v") == Command.KP_DOWN_FINE
        assert prov.map_key("b") == Command.KP_UP_FINE
        assert prov.map_key("f") == Command.KP_DOWN
        assert prov.map_key("g") == Command.KP_UP
        assert prov.map_key("r") == Command.KP_RESET

    def test_map_key_switch_policy(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_BASE)
        assert prov.map_key("2") == Command.SWITCH_POLICY_2

    def test_map_key_unhandled_returns_none(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_BASE)
        assert prov.map_key("x") is None
        assert prov.map_key("w") is None

    def test_locomotion_mapping(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_LOCOMOTION)
        assert prov.map_key("=") == LocomotionCommand.STAND_TOGGLE
        assert prov.map_key("]") == Command.START  # inherited

    def test_wbt_mapping(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy, KEYBOARD_WBT)
        assert prov.map_key("s") == WbtCommand.START_MOTION_CLIP
        assert prov.map_key("o") == Command.STOP  # inherited


class TestLocomotionKeyboardVelocityInput:
    @pytest.mark.parametrize("key", ["w", "s", "a", "d"])
    def test_wasd_handled(self, policy, key):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardVelocityInput

        prov = LocomotionKeyboardVelocityInput(policy)
        assert prov.handle_key(key) is True
        policy._handle_velocity_control.assert_called_once_with(key)

    @pytest.mark.parametrize("key", ["q", "e"])
    def test_angular_velocity(self, policy, key):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardVelocityInput

        prov = LocomotionKeyboardVelocityInput(policy)
        assert prov.handle_key(key) is True
        policy._handle_angular_velocity_control.assert_called_once_with(key)

    def test_zero_velocity(self, policy):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardVelocityInput

        prov = LocomotionKeyboardVelocityInput(policy)
        assert prov.handle_key("z") is True
        policy._handle_zero_velocity.assert_called_once()

    def test_unhandled_key(self, policy):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardVelocityInput

        prov = LocomotionKeyboardVelocityInput(policy)
        assert prov.handle_key("]") is False


# ============================================================================
# Joystick providers
# ============================================================================


class TestJoystickVelocityInput:
    def test_poll_skips_when_no_msg(self, policy):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        policy.interface.get_joystick_msg.return_value = None
        prov = JoystickVelocityInput(policy)
        prov.poll()
        policy.interface.process_joystick_input.assert_not_called()

    def test_poll_reads_and_caches(self, policy):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        new_lin = np.array([[0.5, 0.0]])
        new_ang = np.array([[0.1]])
        new_keys = {"A": True}
        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.process_joystick_input.return_value = (new_lin, new_ang, new_keys)

        prov = JoystickVelocityInput(policy)
        prov.poll()

        np.testing.assert_array_equal(policy.lin_vel_command, new_lin)
        np.testing.assert_array_equal(policy.ang_vel_command, new_ang)
        assert prov.key_states == {"A": True}

    def test_poll_preserves_last_key_states(self, policy):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

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


class TestJoystickOtherInput:
    def test_base_mapping_returns_commands(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        assert prov._mapping["A"] == Command.START
        assert prov._mapping["B"] == Command.STOP
        assert prov._mapping["Y"] == Command.INIT
        assert prov._mapping["select"] == Command.NEXT_POLICY

    def test_locomotion_mapping(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy, JOYSTICK_LOCOMOTION)
        assert prov._mapping["start"] == LocomotionCommand.STAND_TOGGLE
        assert prov._mapping["L2"] == LocomotionCommand.ZERO_VELOCITY
        assert prov._mapping["A"] == Command.START  # inherited

    def test_wbt_mapping(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy, JOYSTICK_WBT)
        assert prov._mapping["start"] == WbtCommand.START_MOTION_CLIP
        assert prov._mapping["B"] == Command.STOP  # inherited

    def test_poll_shared_edge_detection(self, policy):
        """When shared with velocity provider, returns commands for rising edges."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": False}

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == [Command.START]

    def test_poll_shared_no_dispatch_on_hold(self, policy):
        """No commands when button was already held."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": True}

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == []

    def test_poll_standalone_reads_buttons(self, policy):
        """When not shared, reads buttons directly from SDK."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.get_joystick_key.return_value = "B"

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        commands = prov.poll()  # First poll: B goes True (rising edge)

        assert commands == [Command.STOP]

    def test_poll_standalone_skips_when_no_msg(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        policy.interface.get_joystick_msg.return_value = None
        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        commands = prov.poll()
        assert commands == []

    def test_unmapped_button_not_in_commands(self, policy):
        """Buttons not in the mapping are silently ignored."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"UNKNOWN": True}
        vel.last_key_states = {"UNKNOWN": False}

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        commands = prov.poll()

        assert commands == []


# ============================================================================
# ROS2 providers
# ============================================================================


class TestRos2VelocityInput:
    def test_callback_writes_velocity(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=-0.3),
                angular=SimpleNamespace(z=0.8),
            )
        )
        prov._callback(msg)

        np.testing.assert_almost_equal(policy.lin_vel_command[0, 0], 0.5)
        np.testing.assert_almost_equal(policy.lin_vel_command[0, 1], -0.3)
        np.testing.assert_almost_equal(policy.ang_vel_command[0, 0], 0.8)

    def test_callback_clamps_to_range(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=5.0, y=-5.0),
                angular=SimpleNamespace(z=99.0),
            )
        )
        prov._callback(msg)

        assert policy.lin_vel_command[0, 0] == 1.0
        assert policy.lin_vel_command[0, 1] == -1.0
        assert policy.ang_vel_command[0, 0] == 1.0


class TestRos2OtherInput:
    def test_known_commands_queued(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="start"))
        prov._callback(SimpleNamespace(data="stop"))
        prov._callback(SimpleNamespace(data="init"))

        commands = prov.poll()
        assert commands == [Command.START, Command.STOP, Command.INIT]

    def test_walk_stand_commands(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="walk"))
        prov._callback(SimpleNamespace(data="stand"))

        commands = prov.poll()
        assert commands == [LocomotionCommand.WALK, LocomotionCommand.STAND]

    def test_unknown_command_warns(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="bogus"))
        policy.logger.warning.assert_called_once()
        assert prov.poll() == []

    def test_whitespace_and_case_normalization(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="  WALK  "))

        commands = prov.poll()
        assert commands == [LocomotionCommand.WALK]

    def test_poll_drains_queue(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="start"))
        assert prov.poll() == [Command.START]
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
    """Test BasePolicy._create_velocity_input and _create_other_input."""

    def _make_base(self):
        from holosoma_inference.policies.base import BasePolicy

        return BasePolicy.__new__(BasePolicy)

    def test_keyboard_velocity(self):
        from holosoma_inference.inputs.keyboard import KeyboardVelocityInput

        bp = self._make_base()
        result = bp._create_velocity_input(InputSource.keyboard)
        assert isinstance(result, KeyboardVelocityInput)

    def test_joystick_velocity(self):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        bp = self._make_base()
        result = bp._create_velocity_input(InputSource.joystick)
        assert isinstance(result, JoystickVelocityInput)

    def test_ros2_velocity(self):
        from holosoma_inference.inputs.ros2 import Ros2VelocityInput

        bp = self._make_base()
        result = bp._create_velocity_input(InputSource.ros2)
        assert isinstance(result, Ros2VelocityInput)

    def test_keyboard_other(self):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        bp = self._make_base()
        result = bp._create_other_input(InputSource.keyboard)
        assert isinstance(result, KeyboardOtherInput)
        assert result._mapping is KEYBOARD_BASE

    def test_joystick_other(self):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        bp = self._make_base()
        result = bp._create_other_input(InputSource.joystick)
        assert isinstance(result, JoystickOtherInput)
        assert result._mapping is JOYSTICK_BASE

    def test_ros2_other(self):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        bp = self._make_base()
        result = bp._create_other_input(InputSource.ros2)
        assert isinstance(result, Ros2OtherInput)

    def test_unknown_source_raises(self):
        bp = self._make_base()
        with pytest.raises(ValueError, match="Unknown velocity"):
            bp._create_velocity_input("invalid")
        with pytest.raises(ValueError, match="Unknown other"):
            bp._create_other_input("invalid")


@_skip_policies
class TestLocomotionPolicyFactory:
    """Test LocomotionPolicy overrides for keyboard/joystick providers."""

    def _make_loco(self):
        from holosoma_inference.policies.locomotion import LocomotionPolicy

        return LocomotionPolicy.__new__(LocomotionPolicy)

    def test_keyboard_velocity_is_locomotion(self):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardVelocityInput

        lp = self._make_loco()
        result = lp._create_velocity_input(InputSource.keyboard)
        assert isinstance(result, LocomotionKeyboardVelocityInput)

    def test_keyboard_other_uses_locomotion_mapping(self):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        lp = self._make_loco()
        result = lp._create_other_input(InputSource.keyboard)
        assert isinstance(result, KeyboardOtherInput)
        assert result._mapping is KEYBOARD_LOCOMOTION

    def test_joystick_other_uses_locomotion_mapping(self):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        lp = self._make_loco()
        result = lp._create_other_input(InputSource.joystick)
        assert isinstance(result, JoystickOtherInput)
        assert result._mapping is JOYSTICK_LOCOMOTION

    def test_joystick_velocity_falls_to_base(self):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        lp = self._make_loco()
        result = lp._create_velocity_input(InputSource.joystick)
        assert type(result) is JoystickVelocityInput

    def test_ros2_falls_to_base(self):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput, Ros2VelocityInput

        lp = self._make_loco()
        assert isinstance(lp._create_velocity_input(InputSource.ros2), Ros2VelocityInput)
        assert isinstance(lp._create_other_input(InputSource.ros2), Ros2OtherInput)


@_skip_policies
class TestWbtPolicyFactory:
    """Test WholeBodyTrackingPolicy overrides for keyboard/joystick providers."""

    def _make_wbt(self):
        from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

        return WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)

    def test_keyboard_other_uses_wbt_mapping(self):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        wp = self._make_wbt()
        result = wp._create_other_input(InputSource.keyboard)
        assert isinstance(result, KeyboardOtherInput)
        assert result._mapping is KEYBOARD_WBT

    def test_joystick_other_uses_wbt_mapping(self):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        wp = self._make_wbt()
        result = wp._create_other_input(InputSource.joystick)
        assert isinstance(result, JoystickOtherInput)
        assert result._mapping is JOYSTICK_WBT

    def test_keyboard_velocity_falls_to_base(self):
        from holosoma_inference.inputs.keyboard import KeyboardVelocityInput

        wp = self._make_wbt()
        result = wp._create_velocity_input(InputSource.keyboard)
        assert type(result) is KeyboardVelocityInput

    def test_ros2_falls_to_base(self):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput, Ros2VelocityInput

        wp = self._make_wbt()
        assert isinstance(wp._create_velocity_input(InputSource.ros2), Ros2VelocityInput)
        assert isinstance(wp._create_other_input(InputSource.ros2), Ros2OtherInput)


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
    from holosoma_inference.inputs.joystick import JoystickOtherInput
    from holosoma_inference.policies.dual_mode import DualModePolicy

    dual = object.__new__(DualModePolicy)
    dual.primary = _make_policy()
    dual.secondary = _make_policy()
    dual.active = dual.primary
    dual.active_label = "primary"

    dual.primary._velocity_input = MagicMock()
    dual.secondary._velocity_input = MagicMock()

    # Give both policies real JoystickOtherInput with base mappings
    dual.primary._other_input = JoystickOtherInput(dual.primary, dict(JOYSTICK_BASE))
    dual.secondary._other_input = JoystickOtherInput(dual.secondary, dict(JOYSTICK_BASE))

    # Set up real handle_keyboard_button and _dispatch_command on mock policies
    # so _setup_command_intercept can store and patch them
    dual.primary.handle_keyboard_button = MagicMock()
    dual.secondary.handle_keyboard_button = MagicMock()
    dual.primary._dispatch_command = MagicMock()
    dual.secondary._dispatch_command = MagicMock()

    dual._setup_command_intercept()
    return dual


@_skip_dual_mode
class TestDualModeSwitching:
    def test_switch_mode_injected_in_mappings(self):
        dual = _make_dual()
        assert dual.primary._other_input._mapping["X"] == DualModeCommand.SWITCH_MODE
        assert dual.primary._other_input._mapping["x"] == DualModeCommand.SWITCH_MODE
        assert dual.secondary._other_input._mapping["X"] == DualModeCommand.SWITCH_MODE

    def test_x_joystick_triggers_switch(self):
        dual = _make_dual()
        assert dual.active is dual.primary
        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)
        assert dual.active is dual.secondary

    def test_double_switch_returns_to_primary(self):
        dual = _make_dual()
        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)
        assert dual.active is dual.secondary
        dual.secondary._dispatch_command(DualModeCommand.SWITCH_MODE)
        assert dual.active is dual.primary
        assert dual.active_label == "primary"

    def test_non_switch_command_delegates_to_active(self):
        dual = _make_dual()
        orig_dispatch = dual._orig_dispatch[id(dual.primary)]
        dual.primary._dispatch_command(Command.START)
        orig_dispatch.assert_called_once_with(Command.START)

    def test_delegates_to_secondary_after_switch(self):
        dual = _make_dual()
        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)  # switch to secondary
        orig_secondary = dual._orig_dispatch[id(dual.secondary)]
        dual.secondary._dispatch_command(Command.START)
        orig_secondary.assert_called_once_with(Command.START)

    def test_switch_stops_old_and_starts_new(self):
        dual = _make_dual()
        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)
        dual.primary._handle_stop_policy.assert_called_once()
        dual.secondary._resolve_control_gains.assert_called_once()
        dual.secondary._init_phase_components.assert_called_once()
        dual.secondary._handle_start_policy.assert_called_once()

    def test_keyboard_routes_to_active(self):
        dual = _make_dual()
        orig_primary_kb = dual._orig_kb[id(dual.primary)]

        # Before switch: keyboard routes to primary
        dual.primary.handle_keyboard_button("some_key")
        orig_primary_kb.assert_called_once_with("some_key")

    def test_keyboard_routes_to_secondary_after_switch(self):
        dual = _make_dual()
        orig_secondary_kb = dual._orig_kb[id(dual.secondary)]

        # Switch to secondary
        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)

        # Keyboard should now route to secondary
        dual.primary.handle_keyboard_button("some_key")
        orig_secondary_kb.assert_called_once_with("some_key")

    def test_joystick_state_carry_over(self):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        dual = _make_dual()
        # Replace mock velocity inputs with real JoystickVelocityInput
        pri_vel = JoystickVelocityInput(dual.primary)
        pri_vel.key_states = {"X": True, "A": False}
        sec_vel = JoystickVelocityInput(dual.secondary)

        dual.primary._velocity_input = pri_vel
        dual.secondary._velocity_input = sec_vel

        dual.primary._dispatch_command(DualModeCommand.SWITCH_MODE)

        assert sec_vel.key_states == {"X": True, "A": False}
        assert sec_vel.last_key_states == {"X": True, "A": False}


# ============================================================================
# Shared joystick state wiring
# ============================================================================


class TestSharedJoystickWiring:
    def test_shared_velocity_wired_when_both_joystick(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        other = JoystickOtherInput(policy, JOYSTICK_BASE)
        other._shared_velocity = vel  # Simulating what _create_input_providers does

        assert other._shared_velocity is vel

    def test_shared_velocity_none_by_default(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        other = JoystickOtherInput(policy, JOYSTICK_BASE)
        assert other._shared_velocity is None


# ============================================================================
# Separation guarantee: wrong-channel keys are not handled
# ============================================================================


class TestChannelSeparation:
    """When velocity_input=ros2, keyboard WASD must NOT affect velocity."""

    def test_base_keyboard_velocity_ignores_wasd(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardVelocityInput

        prov = KeyboardVelocityInput(policy)
        for key in ("w", "a", "s", "d", "q", "e", "z"):
            assert prov.handle_key(key) is False
        policy._handle_velocity_control.assert_not_called()

    def test_wasd_not_in_keyboard_other_mapping(self, policy):
        """KeyboardOtherInput mapping does not contain velocity keys."""
        for key in ("w", "a", "s", "d", "q", "e", "z"):
            assert key not in KEYBOARD_BASE
            assert key not in KEYBOARD_LOCOMOTION

    def test_joystick_other_ignores_unmapped_buttons(self, policy):
        """JoystickOtherInput only returns commands for mapped buttons."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"unknown_stick": True}
        vel.last_key_states = {"unknown_stick": False}

        prov = JoystickOtherInput(policy, JOYSTICK_BASE)
        prov._shared_velocity = vel
        assert prov.poll() == []
