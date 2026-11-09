"""Optional voice front-end: speech in, speech out, around the unchanged text agent.

Install with ``uv sync --extra voice``. This package is safe to import without the
extra: the third-party modules (``elevenlabs``, ``sounddevice``) are imported only
when a speech client or an audio device is actually created, and a missing extra
surfaces as ``VoiceUnavailableError`` with an install hint, not an ImportError.
The core agent never imports anything from here.
"""

from food_agent.voice.audio import Audio
from food_agent.voice.channel import VoiceChannel, build_voice_channel
from food_agent.voice.errors import SpeechError, VoiceUnavailableError
from food_agent.voice.speakable import to_speech

__all__ = [
    "Audio",
    "SpeechError",
    "VoiceChannel",
    "VoiceUnavailableError",
    "build_voice_channel",
    "to_speech",
]
