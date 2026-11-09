"""Errors raised by the voice front-end."""

from __future__ import annotations

INSTALL_HINT = (
    "voice mode needs the optional 'voice' extra: run `uv sync --extra voice` "
    "(or `pip install 'food-order-agent[voice]'`)."
)
PORTAUDIO_HINT = (
    "the PortAudio system library was not found. Install it "
    "(macOS: `brew install portaudio`, Debian/Ubuntu: `sudo apt install libportaudio2`)."
)


class VoiceUnavailableError(RuntimeError):
    """Voice mode cannot start: missing extra, system audio library or API key."""


class SpeechError(RuntimeError):
    """A speech-to-text or text-to-speech call failed. The conversation carries on."""
