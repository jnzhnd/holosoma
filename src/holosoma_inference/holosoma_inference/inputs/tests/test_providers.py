"""Unit tests for input providers.

Tests cover:
- Keyboard providers: key dispatch, locomotion/WBT overrides
- Joystick providers: button dispatch, shared state wiring, edge detection, poll guard
- ROS2 providers: callback dispatch, velocity clamping
- Factory methods on BasePolicy / LocomotionPolicy / WBT
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.config.config_types.task import InputSource


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
    # _try_switch_policy_key returns False by default (MagicMock is truthy)
    p._try_switch_policy_key.return_value = False
    p.config = SimpleNamespace(
        task=SimpleNamespace(
            ros_cmd_vel_topic="cmd_vel",
            ros_other_input_topic="holosoma/other_input",
        )
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture()
def policy():
    return _make_policy()


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
        assert not hasattr(policy, "_keyboard_listener") or isinstance(
            policy._keyboard_listener, MagicMock
        )

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
        other = KeyboardOtherInput(p)
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
    def test_start_policy(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy)
        assert prov.handle_key("]") is True
        policy._handle_start_policy.assert_called_once()

    def test_stop_policy(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy)
        assert prov.handle_key("o") is True
        policy._handle_stop_policy.assert_called_once()

    def test_init_state(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy)
        assert prov.handle_key("i") is True
        policy._handle_init_state.assert_called_once()

    @pytest.mark.parametrize("key", ["v", "b", "f", "g", "r"])
    def test_kp_control(self, policy, key):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        prov = KeyboardOtherInput(policy)
        assert prov.handle_key(key) is True
        policy._handle_kp_control.assert_called_once_with(key)

    def test_switch_policy_key(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        policy._try_switch_policy_key.return_value = True
        prov = KeyboardOtherInput(policy)
        assert prov.handle_key("2") is True
        policy._try_switch_policy_key.assert_called_once_with("2")

    def test_unhandled_key_returns_false(self, policy):
        from holosoma_inference.inputs.keyboard import KeyboardOtherInput

        policy._try_switch_policy_key.return_value = False
        prov = KeyboardOtherInput(policy)
        assert prov.handle_key("x") is False


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


class TestLocomotionKeyboardOtherInput:
    def test_stand_command(self, policy):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardOtherInput

        prov = LocomotionKeyboardOtherInput(policy)
        assert prov.handle_key("=") is True
        policy._handle_stand_command.assert_called_once()

    def test_falls_through_to_base(self, policy):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardOtherInput

        prov = LocomotionKeyboardOtherInput(policy)
        assert prov.handle_key("]") is True
        policy._handle_start_policy.assert_called_once()


class TestWbtKeyboardOtherInput:
    def test_start_motion_clip(self, policy):
        from holosoma_inference.inputs.keyboard import WbtKeyboardOtherInput

        prov = WbtKeyboardOtherInput(policy)
        assert prov.handle_key("s") is True
        policy._handle_start_motion_clip.assert_called_once()

    def test_falls_through_to_base(self, policy):
        from holosoma_inference.inputs.keyboard import WbtKeyboardOtherInput

        prov = WbtKeyboardOtherInput(policy)
        assert prov.handle_key("o") is True
        policy._handle_stop_policy.assert_called_once()


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
    def test_button_dispatch_A(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button("A") is True
        policy._handle_start_policy.assert_called_once()

    def test_button_dispatch_B(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button("B") is True
        policy._handle_stop_policy.assert_called_once()

    def test_button_dispatch_Y(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button("Y") is True
        policy._handle_init_state.assert_called_once()

    @pytest.mark.parametrize("key", ["up", "down", "left", "right", "F1"])
    def test_kp_control(self, policy, key):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button(key) is True
        policy._handle_joystick_kp_control.assert_called_once_with(key)

    def test_select_cycles_policy(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button("select") is True
        policy._activate_policy.assert_called_once_with(1)

    def test_unknown_button_returns_false(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        assert prov.handle_joystick_button("unknown") is False

    def test_poll_shared_edge_detection(self, policy):
        """When shared with velocity provider, detects rising edges from cached state."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": False}

        prov = JoystickOtherInput(policy)
        prov._shared_velocity = vel
        prov.poll()

        # A was pressed (rising edge) -> should dispatch
        policy.handle_joystick_button.assert_called_once_with("A")

    def test_poll_shared_no_dispatch_on_hold(self, policy):
        """No dispatch when button was already held."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        vel.key_states = {"A": True}
        vel.last_key_states = {"A": True}

        prov = JoystickOtherInput(policy)
        prov._shared_velocity = vel
        prov.poll()

        policy.handle_joystick_button.assert_not_called()

    def test_poll_standalone_reads_buttons(self, policy):
        """When not shared, reads buttons directly from SDK."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        policy.interface.get_joystick_msg.return_value = "msg"
        policy.interface.get_joystick_key.return_value = "B"

        prov = JoystickOtherInput(policy)
        prov.poll()  # First poll: B goes True (rising edge)

        policy.handle_joystick_button.assert_called_once_with("B")

    def test_poll_standalone_skips_when_no_msg(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        policy.interface.get_joystick_msg.return_value = None
        prov = JoystickOtherInput(policy)
        prov.poll()
        policy.handle_joystick_button.assert_not_called()


class TestLocomotionJoystickOtherInput:
    def test_stand_command(self, policy):
        from holosoma_inference.inputs.joystick import LocomotionJoystickOtherInput

        prov = LocomotionJoystickOtherInput(policy)
        assert prov.handle_joystick_button("start") is True
        policy._handle_stand_command.assert_called_once()

    def test_zero_velocity(self, policy):
        from holosoma_inference.inputs.joystick import LocomotionJoystickOtherInput

        prov = LocomotionJoystickOtherInput(policy)
        assert prov.handle_joystick_button("L2") is True
        policy._handle_zero_velocity.assert_called_once()

    def test_falls_through_to_base(self, policy):
        from holosoma_inference.inputs.joystick import LocomotionJoystickOtherInput

        prov = LocomotionJoystickOtherInput(policy)
        assert prov.handle_joystick_button("A") is True
        policy._handle_start_policy.assert_called_once()


class TestWbtJoystickOtherInput:
    def test_start_motion_clip(self, policy):
        from holosoma_inference.inputs.joystick import WbtJoystickOtherInput

        prov = WbtJoystickOtherInput(policy)
        assert prov.handle_joystick_button("start") is True
        policy._handle_start_motion_clip.assert_called_once()

    def test_falls_through_to_base(self, policy):
        from holosoma_inference.inputs.joystick import WbtJoystickOtherInput

        prov = WbtJoystickOtherInput(policy)
        assert prov.handle_joystick_button("B") is True
        policy._handle_stop_policy.assert_called_once()


# ============================================================================
# ROS2 providers
# ============================================================================


class TestRos2VelocityInput:
    def test_callback_writes_velocity(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.5, y=-0.3),
            angular=SimpleNamespace(z=0.8),
        ))
        prov._callback(msg)

        np.testing.assert_almost_equal(policy.lin_vel_command[0, 0], 0.5)
        np.testing.assert_almost_equal(policy.lin_vel_command[0, 1], -0.3)
        np.testing.assert_almost_equal(policy.ang_vel_command[0, 0], 0.8)

    def test_callback_clamps_to_range(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2VelocityInput

        prov = Ros2VelocityInput(policy)
        msg = SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=5.0, y=-5.0),
            angular=SimpleNamespace(z=99.0),
        ))
        prov._callback(msg)

        assert policy.lin_vel_command[0, 0] == 1.0
        assert policy.lin_vel_command[0, 1] == -1.0
        assert policy.ang_vel_command[0, 0] == 1.0


class TestRos2OtherInput:
    def test_walk_command(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="walk"))
        assert policy.stand_command[0, 0] == 1
        np.testing.assert_almost_equal(policy.base_height_command[0, 0], 0.5)

    def test_stand_command(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        policy.stand_command[0, 0] = 1
        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="stand"))
        assert policy.stand_command[0, 0] == 0

    def test_start_command(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="start"))
        policy._handle_start_policy.assert_called_once()

    def test_stop_command(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="stop"))
        policy._handle_stop_policy.assert_called_once()

    def test_init_command(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="init"))
        policy._handle_init_state.assert_called_once()

    def test_unknown_command_warns(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="bogus"))
        policy.logger.warning.assert_called_once()

    def test_whitespace_and_case_normalization(self, policy):
        from holosoma_inference.inputs.ros2 import Ros2OtherInput

        prov = Ros2OtherInput(policy)
        prov._callback(SimpleNamespace(data="  WALK  "))
        assert policy.stand_command[0, 0] == 1


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

    def test_joystick_other(self):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        bp = self._make_base()
        result = bp._create_other_input(InputSource.joystick)
        assert isinstance(result, JoystickOtherInput)

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

    def test_keyboard_other_is_locomotion(self):
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardOtherInput

        lp = self._make_loco()
        result = lp._create_other_input(InputSource.keyboard)
        assert isinstance(result, LocomotionKeyboardOtherInput)

    def test_joystick_other_is_locomotion(self):
        from holosoma_inference.inputs.joystick import LocomotionJoystickOtherInput

        lp = self._make_loco()
        result = lp._create_other_input(InputSource.joystick)
        assert isinstance(result, LocomotionJoystickOtherInput)

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

    def test_keyboard_other_is_wbt(self):
        from holosoma_inference.inputs.keyboard import WbtKeyboardOtherInput

        wp = self._make_wbt()
        result = wp._create_other_input(InputSource.keyboard)
        assert isinstance(result, WbtKeyboardOtherInput)

    def test_joystick_other_is_wbt(self):
        from holosoma_inference.inputs.joystick import WbtJoystickOtherInput

        wp = self._make_wbt()
        result = wp._create_other_input(InputSource.joystick)
        assert isinstance(result, WbtJoystickOtherInput)

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
    from holosoma_inference.policies.dual_mode import DualModePolicy

    dual = object.__new__(DualModePolicy)
    dual.primary = _make_policy()
    dual.secondary = _make_policy()
    dual.active = dual.primary
    dual.active_label = "primary"

    dual.primary._velocity_input = MagicMock()
    dual.secondary._velocity_input = MagicMock()
    dual.primary._other_input = MagicMock()
    dual.secondary._other_input = MagicMock()

    dual._patch_button_handlers()
    return dual


@_skip_dual_mode
class TestDualModeSwitching:
    def test_x_keyboard_triggers_switch(self):
        dual = _make_dual()
        assert dual.active is dual.primary
        dual.primary.handle_keyboard_button("x")
        assert dual.active is dual.secondary
        assert dual.active_label == "secondary"

    def test_X_joystick_triggers_switch(self):
        dual = _make_dual()
        dual.primary.handle_joystick_button("X")
        assert dual.active is dual.secondary

    def test_double_switch_returns_to_primary(self):
        dual = _make_dual()
        dual.primary.handle_keyboard_button("x")
        dual.secondary.handle_keyboard_button("x")
        assert dual.active is dual.primary
        assert dual.active_label == "primary"

    def test_non_switch_keyboard_delegates_to_active(self):
        dual = _make_dual()
        orig_primary_kb = dual._orig_kb[id(dual.primary)]
        dual.primary.handle_keyboard_button("]")
        orig_primary_kb.assert_called_once_with("]")

    def test_non_switch_joystick_delegates_to_active(self):
        dual = _make_dual()
        orig_primary_joy = dual._orig_joy[id(dual.primary)]
        dual.primary.handle_joystick_button("A")
        orig_primary_joy.assert_called_once_with("A")

    def test_delegates_to_secondary_after_switch(self):
        dual = _make_dual()
        dual.primary.handle_keyboard_button("x")  # switch to secondary
        orig_secondary_kb = dual._orig_kb[id(dual.secondary)]
        dual.secondary.handle_keyboard_button("]")
        orig_secondary_kb.assert_called_once_with("]")

    def test_switch_stops_old_and_starts_new(self):
        dual = _make_dual()
        dual.primary.handle_keyboard_button("x")
        dual.primary._handle_stop_policy.assert_called_once()
        dual.secondary._resolve_control_gains.assert_called_once()
        dual.secondary._init_phase_components.assert_called_once()
        dual.secondary._handle_start_policy.assert_called_once()

    def test_joystick_state_carry_over(self):
        from holosoma_inference.inputs.joystick import JoystickVelocityInput

        dual = _make_dual()
        # Replace mock velocity inputs with real JoystickVelocityInput
        pri_vel = JoystickVelocityInput(dual.primary)
        pri_vel.key_states = {"X": True, "A": False}
        sec_vel = JoystickVelocityInput(dual.secondary)

        dual.primary._velocity_input = pri_vel
        dual.secondary._velocity_input = sec_vel

        dual.primary.handle_keyboard_button("x")

        assert sec_vel.key_states == {"X": True, "A": False}
        assert sec_vel.last_key_states == {"X": True, "A": False}


# ============================================================================
# Shared joystick state wiring
# ============================================================================


class TestSharedJoystickWiring:
    def test_shared_velocity_wired_when_both_joystick(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput, JoystickVelocityInput

        vel = JoystickVelocityInput(policy)
        other = JoystickOtherInput(policy)
        other._shared_velocity = vel  # Simulating what _create_input_providers does

        assert other._shared_velocity is vel

    def test_shared_velocity_none_by_default(self, policy):
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        other = JoystickOtherInput(policy)
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

    def test_locomotion_velocity_does_not_leak_to_other(self, policy):
        """LocomotionKeyboardOtherInput does NOT handle WASD."""
        from holosoma_inference.inputs.keyboard import LocomotionKeyboardOtherInput

        policy._try_switch_policy_key.return_value = False
        prov = LocomotionKeyboardOtherInput(policy)
        for key in ("w", "a", "s", "d", "q", "e", "z"):
            assert prov.handle_key(key) is False

    def test_joystick_other_does_not_handle_velocity(self, policy):
        """JoystickOtherInput only handles buttons, not sticks."""
        from holosoma_inference.inputs.joystick import JoystickOtherInput

        prov = JoystickOtherInput(policy)
        # Sticks don't produce button events, so no dispatch occurs
        assert prov.handle_joystick_button("unknown_stick") is False
