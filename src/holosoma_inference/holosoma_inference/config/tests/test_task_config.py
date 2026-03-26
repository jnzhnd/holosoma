"""Unit tests for TaskConfig input source configuration.

Covers the InputSource enum, velocity_input/other_input fields,
use_keyboard/use_joystick shortcut aliases, and mutual exclusion validation.
"""

import pytest

from holosoma_inference.config.config_types.task import (
    DEFAULT_OTHER_INPUT,
    DEFAULT_VELOCITY_INPUT,
    InputSource,
    TaskConfig,
)


class TestInputSourceEnum:
    """Verify InputSource enum values and str inheritance."""

    def test_members(self):
        assert set(InputSource) == {InputSource.keyboard, InputSource.joystick, InputSource.ros2}

    def test_string_values(self):
        assert InputSource.keyboard.value == "keyboard"
        assert InputSource.joystick.value == "joystick"
        assert InputSource.ros2.value == "ros2"

    def test_str_comparison(self):
        """InputSource inherits from str, so direct comparison works."""
        assert InputSource.keyboard == "keyboard"
        assert InputSource.joystick == "joystick"
        assert InputSource.ros2 == "ros2"

    def test_defaults(self):
        assert DEFAULT_VELOCITY_INPUT == InputSource.keyboard
        assert DEFAULT_OTHER_INPUT == InputSource.keyboard


class TestTaskConfigDefaults:
    """Verify default field values."""

    def test_default_inputs(self):
        tc = TaskConfig(model_path="test.onnx")
        assert tc.velocity_input == InputSource.keyboard
        assert tc.other_input == InputSource.keyboard

    def test_shortcuts_default_false(self):
        tc = TaskConfig(model_path="test.onnx")
        assert tc.use_joystick is False
        assert tc.use_keyboard is False


class TestExplicitInputSelection:
    """Verify explicit velocity_input/other_input combinations."""

    @pytest.mark.parametrize(
        "vel, other",
        [
            (InputSource.keyboard, InputSource.keyboard),
            (InputSource.joystick, InputSource.joystick),
            (InputSource.ros2, InputSource.ros2),
            (InputSource.ros2, InputSource.keyboard),
            (InputSource.ros2, InputSource.joystick),
            (InputSource.joystick, InputSource.keyboard),
            (InputSource.keyboard, InputSource.ros2),
            (InputSource.joystick, InputSource.ros2),
            (InputSource.keyboard, InputSource.joystick),
        ],
    )
    def test_all_combinations(self, vel, other):
        tc = TaskConfig(model_path="test.onnx", velocity_input=vel, other_input=other)
        assert tc.velocity_input == vel
        assert tc.other_input == other


class TestUseJoystickShortcut:
    """Verify --use-joystick shortcut behavior."""

    def test_sets_both_channels(self):
        tc = TaskConfig(model_path="test.onnx", use_joystick=True)
        assert tc.velocity_input == InputSource.joystick
        assert tc.other_input == InputSource.joystick

    def test_conflicts_with_velocity_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_joystick=True, velocity_input=InputSource.ros2)

    def test_conflicts_with_other_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_joystick=True, other_input=InputSource.ros2)

    def test_conflicts_with_both_inputs(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(
                model_path="test.onnx",
                use_joystick=True,
                velocity_input=InputSource.ros2,
                other_input=InputSource.ros2,
            )


class TestUseKeyboardShortcut:
    """Verify --use-keyboard shortcut behavior."""

    def test_sets_both_channels(self):
        tc = TaskConfig(model_path="test.onnx", use_keyboard=True)
        assert tc.velocity_input == InputSource.keyboard
        assert tc.other_input == InputSource.keyboard

    def test_conflicts_with_velocity_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, velocity_input=InputSource.ros2)

    def test_conflicts_with_other_input(self):
        with pytest.raises(Exception, match="Cannot combine"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, other_input=InputSource.joystick)


class TestShortcutMutualExclusion:
    """Verify use_keyboard and use_joystick cannot be combined."""

    def test_both_shortcuts_rejected(self):
        with pytest.raises(Exception, match="Cannot combine.*use-keyboard.*use-joystick"):
            TaskConfig(model_path="test.onnx", use_keyboard=True, use_joystick=True)


class TestDefaultConfigs:
    """Verify the preset default configs are valid."""

    def test_all_defaults_load(self):
        from holosoma_inference.config.config_values.task import DEFAULTS

        for name, config in DEFAULTS.items():
            assert isinstance(config.velocity_input, InputSource), f"{name}: velocity_input not InputSource"
            assert isinstance(config.other_input, InputSource), f"{name}: other_input not InputSource"

    def test_default_configs_use_keyboard(self):
        from holosoma_inference.config.config_values.task import DEFAULTS

        for name, config in DEFAULTS.items():
            assert config.velocity_input == InputSource.keyboard, f"{name}: unexpected velocity_input"
            assert config.other_input == InputSource.keyboard, f"{name}: unexpected other_input"
