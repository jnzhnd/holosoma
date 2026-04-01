"""Elgato Stream Deck XL input provider.

Implements both :class:`VelCmdProvider` and :class:`StateCommandProvider`.
Velocity buttons work like keyboard WASD — each press nudges by a fixed delta.

Button layout (4 rows x 8 cols, index = row*8 + col)::

    ┌─────┬─────────────┬───────────────────────────────────────────┐
    │ Col │   Purpose   │            Layout (top->bottom)           │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 0   │ KP          │ +, level (live), RESET, -                  │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 1   │ KD          │ +, level (live), RESET, -                  │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 2   │ locomotion  │ STAND, ZERO VEL, SWITCH MODE, MOTION CLIP │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 3-5 │ velocity    │ row 1: ↶  ↑  ↷   row 2: ←  ↓  →           │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 6   │ policy nav  │ prev (top), next (bottom)                 │
    ├─────┼─────────────┼───────────────────────────────────────────┤
    │ 7   │ lifecycle   │ START, STOP, INIT, KILL                   │
    └─────┴─────────────┴───────────────────────────────────────────┘
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from loguru import logger

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd

# ---------------------------------------------------------------------------
# Visual style per button
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ButtonStyle:
    """Display properties for a single Stream Deck key."""

    command: StateCommand | None
    label: str
    bg: tuple[int, int, int]
    fg: tuple[int, int, int] = (255, 255, 255)
    small: bool = False


# Colour palette
_GREEN = (20, 140, 60)
_RED = (180, 30, 30)
_ORANGE = (200, 100, 20)
_BLUE = (30, 80, 160)
_CYAN = (20, 130, 140)
_TEAL = (0, 150, 130)
_ROSE = (160, 50, 90)
_PURPLE = (90, 50, 160)
_SLATE = (60, 75, 90)
_DARK = (40, 40, 40)

# Velocity button sentinel — not a StateCommand, handled separately
_VEL_FWD = "vel_fwd"
_VEL_BWD = "vel_bwd"
_VEL_LEFT = "vel_left"
_VEL_RIGHT = "vel_right"
_VEL_YAW_L = "vel_yaw_l"
_VEL_YAW_R = "vel_yaw_r"

VEL_DELTA = 0.1


def _pos(row: int, col: int) -> int:
    """Stream Deck XL button index from row/col (0-indexed)."""
    return row * 8 + col


# ---------------------------------------------------------------------------
# Grid layout — Stream Deck XL: 8 cols x 4 rows
#
# Col 0: KP gain   (+, RESET, −)
# Col 1: KD gain   (same layout, different hue)
# Col 2: locomotion helpers
# Col 3: lin X velocity (fwd / bwd)
# Col 4: lin Y velocity (left / right)
# Col 5: yaw velocity (yaw left / yaw right)
# Col 6: policy prev/next
# Col 7: lifecycle (START → STOP → INIT → KILL, top to bottom)
# ---------------------------------------------------------------------------

STREAMDECK_BUTTONS: dict[int, ButtonStyle] = {
    # Col 0 — KP tuning (+, level, reset, −)
    _pos(0, 0): ButtonStyle(StateCommand.KP_UP_FINE, "KP\n+", _TEAL),
    _pos(1, 0): ButtonStyle(None, "KP\n1.00", _DARK, _TEAL),  # live display
    _pos(2, 0): ButtonStyle(StateCommand.KP_RESET, "KP\nRESET", _DARK, (200, 200, 200)),
    _pos(3, 0): ButtonStyle(StateCommand.KP_DOWN_FINE, "KP\n−", _TEAL),
    # Col 1 — KD tuning (+, level, reset, −)
    _pos(0, 1): ButtonStyle(StateCommand.KD_UP_FINE, "KD\n+", _ROSE),
    _pos(1, 1): ButtonStyle(None, "KD\n1.00", _DARK, _ROSE),  # live display
    _pos(2, 1): ButtonStyle(StateCommand.KD_RESET, "KD\nRESET", _DARK, (200, 200, 200)),
    _pos(3, 1): ButtonStyle(StateCommand.KD_DOWN_FINE, "KD\n−", _ROSE),
    # Col 2 — locomotion helpers
    _pos(0, 2): ButtonStyle(StateCommand.STAND_TOGGLE, "STAND↔\nWALK", _BLUE),
    _pos(1, 2): ButtonStyle(StateCommand.ZERO_VELOCITY, "ZERO\nVELOCITY", _BLUE, small=True),
    _pos(2, 2): ButtonStyle(StateCommand.SWITCH_MODE, "SWITCH\nMODE", _BLUE, small=True),
    _pos(3, 2): ButtonStyle(StateCommand.START_MOTION_CLIP, "MOTION\nCLIP", _BLUE),
    # Cols 3-5 — velocity d-pad
    #   row 1: ↶ yaw L,  ↑ fwd,   ↷ yaw R
    #   row 2: ← left,   ↓ bwd,   → right
    _pos(1, 3): ButtonStyle(None, "\u21b6", _SLATE),  # yaw left
    _pos(1, 4): ButtonStyle(None, "\u2191", _SLATE),  # fwd
    _pos(1, 5): ButtonStyle(None, "\u21b7", _SLATE),  # yaw right
    _pos(2, 3): ButtonStyle(None, "\u2190", _SLATE),  # left
    _pos(2, 4): ButtonStyle(None, "\u2193", _SLATE),  # bwd
    _pos(2, 5): ButtonStyle(None, "\u2192", _SLATE),  # right
    # Col 6 — policy navigation
    _pos(0, 6): ButtonStyle(StateCommand.PREV_POLICY, "Next\npolicy", _PURPLE),
    _pos(3, 6): ButtonStyle(StateCommand.NEXT_POLICY, "Previous\n policy", _PURPLE),
    # Col 7 — lifecycle
    _pos(0, 7): ButtonStyle(StateCommand.START, "START", _GREEN),
    _pos(1, 7): ButtonStyle(StateCommand.STOP, "STOP", _ORANGE),
    _pos(2, 7): ButtonStyle(StateCommand.INIT, "INIT", _CYAN),
    _pos(3, 7): ButtonStyle(StateCommand.KILL, "KILL", _RED),
}

# Command lookup used at runtime (index → StateCommand), excludes velocity buttons
STREAMDECK_COMMANDS: dict[int, StateCommand] = {
    k: btn.command for k, btn in STREAMDECK_BUTTONS.items() if btn.command is not None
}

# Velocity button lookup (index → (lin_x_delta, lin_y_delta, ang_z_delta))
STREAMDECK_VELOCITY: dict[int, tuple[float, float, float]] = {
    # Row 1: yaw L, fwd, yaw R
    _pos(1, 3): (0.0, 0.0, +VEL_DELTA),  # ↶ yaw left
    _pos(1, 4): (+VEL_DELTA, 0.0, 0.0),  # ↑ fwd
    _pos(1, 5): (0.0, 0.0, -VEL_DELTA),  # ↷ yaw right
    # Row 2: left, bwd, right
    _pos(2, 3): (0.0, +VEL_DELTA, 0.0),  # ← left
    _pos(2, 4): (-VEL_DELTA, 0.0, 0.0),  # ↓ bwd
    _pos(2, 5): (0.0, -VEL_DELTA, 0.0),  # → right
}


class StreamDeckInput:
    """Stream Deck XL input provider.

    Satisfies both ``VelCmdProvider`` and ``StateCommandProvider`` protocols.
    Button presses on a background thread are queued (commands) or accumulated
    (velocity) and drained each policy cycle.
    """

    _KP_DISPLAY_KEY = _pos(1, 0)
    _KD_DISPLAY_KEY = _pos(1, 1)
    _SWITCH_MODE_KEY = _pos(2, 2)

    def __init__(self) -> None:
        self._deck = None
        self._pil_helper = None
        self._font = None
        self._command_queue: deque[StateCommand] = deque()
        self._lin_vel = [0.0, 0.0]
        self._ang_vel = 0.0
        self._lock = threading.Lock()
        # _mapping is used by DualModePolicy to inject SWITCH_MODE for
        # keyboard/joystick keys.  Stream Deck maps buttons by index, not
        # string keys, so this dict is unused but must exist for compatibility.
        self._mapping: dict[str, StateCommand] = {}

    def start(self) -> None:
        from StreamDeck.DeviceManager import DeviceManager
        from StreamDeck.ImageHelpers import PILHelper

        decks = DeviceManager().enumerate()
        if not decks:
            raise RuntimeError(
                "No Stream Deck devices found. Is the device plugged in and "
                "accessible? (check udev rules for non-root access)"
            )
        self._deck = decks[0]
        self._deck.open()
        self._deck.reset()
        self._deck.set_brightness(80)
        self._deck.set_key_callback(self._on_key)

        self._pil_helper = PILHelper
        self._draw_all_keys()
        logger.info(
            "Stream Deck {} opened ({} keys)",
            self._deck.deck_type(),
            self._deck.key_count(),
        )

    def _draw_all_keys(self) -> None:
        """Render every key: styled buttons for mapped keys, dark for unmapped."""
        from PIL import ImageFont

        _font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        try:
            self._font = ImageFont.truetype(_font_path, 18)
            self._font_small = ImageFont.truetype(_font_path, 14)
            self._font_large = ImageFont.truetype(_font_path, 48)
        except OSError:
            self._font = ImageFont.load_default()
            self._font_small = self._font
            self._font_large = self._font

        for k in range(self._deck.key_count()):
            btn = STREAMDECK_BUTTONS.get(k)
            bg = btn.bg if btn else _DARK
            fg = btn.fg if btn else (80, 80, 80)
            label = btn.label if btn else ""
            font = self._font_small if btn and btn.small else None
            self._draw_key(k, label, bg, fg, font=font)

    def _draw_key(
        self,
        key: int,
        label: str,
        bg: tuple,
        fg: tuple,
        font=None,
    ) -> None:
        """Render a single key image."""
        from PIL import ImageDraw

        if font is None:
            # Single-line, single-char labels (arrows) get the large font
            font = self._font_large if "\n" not in label and len(label) <= 2 else self._font

        img = self._pil_helper.create_image(self._deck)
        draw = ImageDraw.Draw(img)
        draw.rectangle(((0, 0), img.size), fill=bg)

        if label:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (img.width - tw) // 2
            y = (img.height - th) // 2
            draw.text((x, y), label, font=font, fill=fg, align="center")

        self._deck.set_key_image(key, self._pil_helper.to_native_format(self._deck, img))

    def update_gain_display(self, kp_level: float, kd_level: float) -> None:
        """Redraw the KP/KD display buttons with current values."""
        if self._deck is None:
            return
        self._draw_key(self._KP_DISPLAY_KEY, f"KP\n{kp_level:.2f}", _DARK, _TEAL)
        self._draw_key(self._KD_DISPLAY_KEY, f"KD\n{kd_level:.2f}", _DARK, _ROSE)

    def update_mode_display(self, active_label: str) -> None:
        """Redraw the SWITCH MODE button to show which mode will be switched to."""
        if self._deck is None:
            return
        target = "PRIMARY" if active_label == "secondary" else "SECONDARY"
        self._draw_key(
            self._SWITCH_MODE_KEY, f"SWITCH\n{target}", _BLUE, (255, 255, 255),
            font=self._font_small,
        )

    def _on_key(self, deck, key: int, pressed: bool) -> None:
        if not pressed:
            return

        with self._lock:
            # Check velocity buttons first
            vel = STREAMDECK_VELOCITY.get(key)
            if vel is not None:
                dx, dy, dz = vel
                self._lin_vel[0] += dx
                self._lin_vel[1] += dy
                self._ang_vel += dz
                logger.debug(
                    "Stream Deck key {} -> vel ({:.1f}, {:.1f}, {:.1f})",
                    key,
                    self._lin_vel[0],
                    self._lin_vel[1],
                    self._ang_vel,
                )
                return

            # Check command buttons
            cmd = STREAMDECK_COMMANDS.get(key)
            if cmd is not None:
                self._command_queue.append(cmd)
                logger.debug("Stream Deck key {} -> {}", key, cmd.name)

    # -- VelCmdProvider protocol ----------------------------------------------

    def poll_velocity(self) -> VelCmd | None:
        with self._lock:
            return VelCmd((self._lin_vel[0], self._lin_vel[1]), self._ang_vel)

    def zero(self) -> None:
        with self._lock:
            self._lin_vel = [0.0, 0.0]
            self._ang_vel = 0.0

    # -- StateCommandProvider protocol ----------------------------------------

    def poll_commands(self) -> list[StateCommand]:
        with self._lock:
            cmds = list(self._command_queue)
            self._command_queue.clear()
        return cmds
