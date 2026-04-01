"""Elgato Stream Deck XL input provider.

Implements :class:`StateCommandProvider` only — the Stream Deck has no
analog inputs, so velocity commands should come from another source
(keyboard, joystick, ros2).

Button layout (4 rows x 8 cols, index = row*8 + col):

    col:    0       1       2     3       4       5     6       7
    row 0:  KP++    KD++    ·     STAND   KP RST  ·     POL▲    START
    row 1:  KP+     KD+     ·     ZERO    KD RST  ·     ·       STOP
    row 2:  KP−     KD−     ·     SWITCH  ·       ·     ·       INIT
    row 3:  KP−−    KD−−    ·     MOTION  ·       ·     POL▼    KILL
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from loguru import logger

from holosoma_inference.inputs.api.commands import StateCommand

# ---------------------------------------------------------------------------
# Visual style per button
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ButtonStyle:
    """Display properties for a single Stream Deck key."""

    command: StateCommand
    label: str
    bg: tuple[int, int, int]
    fg: tuple[int, int, int] = (255, 255, 255)


# Colour palette
_GREEN = (20, 140, 60)
_RED = (180, 30, 30)
_ORANGE = (200, 100, 20)
_BLUE = (30, 80, 160)
_CYAN = (20, 130, 140)
_AMBER = (160, 120, 20)
_AMBER_KD = (120, 90, 40)
_PURPLE = (90, 50, 160)
_DARK = (40, 40, 40)


def _pos(row: int, col: int) -> int:
    """Stream Deck XL button index from row/col (0-indexed)."""
    return row * 8 + col


# ---------------------------------------------------------------------------
# Grid layout — Stream Deck XL: 8 cols x 4 rows
#
# Col 0: KP gain   (top = increase, bottom = decrease)
# Col 1: KD gain   (same layout, different hue)
# Col 2: empty break
# Col 3: locomotion helpers
# Col 4: gain resets
# Col 5: empty
# Col 6: policy prev/next
# Col 7: lifecycle (START → STOP → INIT → KILL, top to bottom)
# ---------------------------------------------------------------------------

STREAMDECK_BUTTONS: dict[int, ButtonStyle] = {
    # Col 0 — KP tuning (top = up, bottom = down)
    _pos(0, 0): ButtonStyle(StateCommand.KP_UP, "KP\n+ +", _AMBER),
    _pos(1, 0): ButtonStyle(StateCommand.KP_UP_FINE, "KP\n+", _AMBER),
    _pos(2, 0): ButtonStyle(StateCommand.KP_DOWN_FINE, "KP\n−", _AMBER),
    _pos(3, 0): ButtonStyle(StateCommand.KP_DOWN, "KP\n− −", _AMBER),
    # Col 1 — KD tuning
    _pos(0, 1): ButtonStyle(StateCommand.KD_UP, "KD\n+ +", _AMBER_KD),
    _pos(1, 1): ButtonStyle(StateCommand.KD_UP_FINE, "KD\n+", _AMBER_KD),
    _pos(2, 1): ButtonStyle(StateCommand.KD_DOWN_FINE, "KD\n−", _AMBER_KD),
    _pos(3, 1): ButtonStyle(StateCommand.KD_DOWN, "KD\n− −", _AMBER_KD),
    # Col 2 — empty break
    # Col 3 — locomotion helpers
    _pos(0, 3): ButtonStyle(StateCommand.STAND_TOGGLE, "STAND", _BLUE),
    _pos(1, 3): ButtonStyle(StateCommand.ZERO_VELOCITY, "ZERO\nVEL", _BLUE),
    _pos(2, 3): ButtonStyle(StateCommand.SWITCH_MODE, "SWITCH\nMODE", _BLUE),
    _pos(3, 3): ButtonStyle(StateCommand.START_MOTION_CLIP, "MOTION\nCLIP", _BLUE),
    # Col 4 — gain resets
    _pos(0, 4): ButtonStyle(StateCommand.KP_RESET, "KP\nRESET", _DARK, (200, 200, 200)),
    _pos(1, 4): ButtonStyle(StateCommand.KD_RESET, "KD\nRESET", _DARK, (200, 200, 200)),
    # Col 6 — policy navigation
    _pos(0, 6): ButtonStyle(StateCommand.PREV_POLICY, "POL\n\u25b2", _PURPLE),
    _pos(3, 6): ButtonStyle(StateCommand.NEXT_POLICY, "POL\n\u25bc", _PURPLE),
    # Col 7 — lifecycle (green → orange → cyan → red, top to bottom)
    _pos(0, 7): ButtonStyle(StateCommand.START, "START", _GREEN),
    _pos(1, 7): ButtonStyle(StateCommand.STOP, "STOP", _ORANGE),
    _pos(2, 7): ButtonStyle(StateCommand.INIT, "INIT", _CYAN),
    _pos(3, 7): ButtonStyle(StateCommand.KILL, "KILL", _RED),
}

# Command lookup used at runtime (index → StateCommand)
STREAMDECK_COMMANDS: dict[int, StateCommand] = {k: btn.command for k, btn in STREAMDECK_BUTTONS.items()}


class StreamDeckInput:
    """Stream Deck XL command provider.

    Satisfies the ``StateCommandProvider`` protocol.  Button presses on a
    background thread are queued and drained each policy cycle via
    :meth:`poll_commands`.
    """

    def __init__(self) -> None:
        self._deck = None
        self._queue: deque[StateCommand] = deque()
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

        self._draw_all_keys(pil_helper=PILHelper)
        logger.info(
            "Stream Deck {} opened ({} keys)",
            self._deck.deck_type(),
            self._deck.key_count(),
        )

    def _draw_all_keys(self, pil_helper) -> None:
        """Render every key: styled buttons for mapped keys, dark for unmapped."""
        from PIL import ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        for k in range(self._deck.key_count()):
            btn = STREAMDECK_BUTTONS.get(k)
            bg = btn.bg if btn else _DARK
            fg = btn.fg if btn else (80, 80, 80)
            label = btn.label if btn else ""

            img = pil_helper.create_image(self._deck)
            draw = ImageDraw.Draw(img)
            draw.rectangle(((0, 0), img.size), fill=bg)

            if label:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x = (img.width - tw) // 2
                y = (img.height - th) // 2
                draw.text((x, y), label, font=font, fill=fg, align="center")

            self._deck.set_key_image(k, pil_helper.to_native_format(self._deck, img))

    def _on_key(self, deck, key: int, pressed: bool) -> None:
        if not pressed:
            return
        cmd = STREAMDECK_COMMANDS.get(key)
        if cmd is not None:
            with self._lock:
                self._queue.append(cmd)
            logger.debug("Stream Deck key {} -> {}", key, cmd.name)

    # -- StateCommandProvider protocol ----------------------------------------

    def poll_commands(self) -> list[StateCommand]:
        with self._lock:
            cmds = list(self._queue)
            self._queue.clear()
        return cmds
