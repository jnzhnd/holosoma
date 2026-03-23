"""Config types for eval callbacks."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Configuration for trajectory recording during evaluation."""

    output_path: str | None = None
    """Path to save NPZ recording. None disables recording."""

    env_id: int = 0
    """Environment ID to record."""
