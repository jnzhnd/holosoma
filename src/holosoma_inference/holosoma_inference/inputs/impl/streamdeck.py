"""Elgato Stream Deck XL input provider.

Implements :class:`StateCommandProvider` only — the Stream Deck has no
analog inputs, so velocity commands should come from another source
(keyboard, joystick, ros2).

Buttons are mapped to :class:`StateCommand` values via
:data:`STREAMDECK_COMMANDS`.  Unmapped keys are ignored.
"""

from __future__ import annotations

import threading
from collections import deque

from loguru import logger

from holosoma_inference.inputs.api.commands import StateCommand

# ---------------------------------------------------------------------------
# Button-to-command mapping (Stream Deck XL: 8 cols x 4 rows, 0 = top-left)
# ---------------------------------------------------------------------------

STREAMDECK_COMMANDS: dict[int, StateCommand] = {
    # Row 0 — lifecycle
    0: StateCommand.INIT,
    1: StateCommand.START,
    2: StateCommand.STOP,
    3: StateCommand.STAND_TOGGLE,
    4: StateCommand.ZERO_VELOCITY,
    5: StateCommand.SWITCH_MODE,
    6: StateCommand.START_MOTION_CLIP,
    7: StateCommand.KILL,
    # Row 1 — policy switching
    8: StateCommand.SWITCH_POLICY_1,
    9: StateCommand.SWITCH_POLICY_2,
    10: StateCommand.SWITCH_POLICY_3,
    11: StateCommand.SWITCH_POLICY_4,
    12: StateCommand.SWITCH_POLICY_5,
    13: StateCommand.SWITCH_POLICY_6,
    14: StateCommand.SWITCH_POLICY_7,
    15: StateCommand.SWITCH_POLICY_8,
    # Row 2 — KP tuning
    16: StateCommand.KP_DOWN,
    17: StateCommand.KP_UP,
    18: StateCommand.KP_DOWN_FINE,
    19: StateCommand.KP_UP_FINE,
    20: StateCommand.KP_RESET,
}


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

        self._draw_labels(PILHelper)
        logger.info(
            "Stream Deck {} opened ({} keys)",
            self._deck.deck_type(),
            self._deck.key_count(),
        )

    def _draw_labels(self, PILHelper) -> None:
        """Draw command labels on mapped keys, dim grey on unmapped ones."""
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
        except OSError:
            font = ImageFont.load_default()

        for k in range(self._deck.key_count()):
            cmd = STREAMDECK_COMMANDS.get(k)
            if cmd is not None:
                bg = (30, 80, 160)
                label = cmd.name.replace("_", "\n")
            else:
                bg = (40, 40, 40)
                label = ""

            img = PILHelper.create_image(self._deck)
            draw = ImageDraw.Draw(img)
            draw.rectangle(((0, 0), img.size), fill=bg)

            if label:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x = (img.width - tw) // 2
                y = (img.height - th) // 2
                draw.text((x, y), label, font=font, fill="white", align="center")

            self._deck.set_key_image(k, PILHelper.to_native_format(self._deck, img))

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
