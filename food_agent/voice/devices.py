"""Microphone and speaker through ``sounddevice`` (PortAudio), imported lazily."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

from food_agent.voice.audio import SAMPLE_WIDTH, Audio, aligned_frames
from food_agent.voice.errors import INSTALL_HINT, PORTAUDIO_HINT, VoiceUnavailableError

MIC_SAMPLE_RATE = 16_000  # what speech recognition needs; more is just upload size


def _sounddevice() -> ModuleType:
    try:
        import sounddevice  # noqa: PLC0415 - optional extra, imported on first use
    except ImportError as exc:
        raise VoiceUnavailableError(INSTALL_HINT) from exc
    except OSError as exc:  # the wheel is installed but the PortAudio library is not
        raise VoiceUnavailableError(PORTAUDIO_HINT) from exc
    return sounddevice


class Microphone:
    """Push-to-talk recorder: captures from the default input until told to stop."""

    def __init__(self, sample_rate: int = MIC_SAMPLE_RATE) -> None:
        self._sd = _sounddevice()
        self.sample_rate = sample_rate

    def record(self, wait_for_stop: Callable[[], object], max_seconds: float) -> Audio:
        """Record until ``wait_for_stop()`` returns; keeps at most ``max_seconds``."""
        limit = int(max_seconds * self.sample_rate) * SAMPLE_WIDTH
        chunks: list[bytes] = []
        size = 0

        def on_audio(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            nonlocal size
            if size < limit:  # past the cap, keep the stream open but drop the audio
                chunks.append(bytes(indata))
                size += len(indata)

        with self._sd.RawInputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", callback=on_audio
        ):
            wait_for_stop()
        return Audio(b"".join(chunks)[:limit], self.sample_rate)


class Speaker:
    """Plays a PCM stream as it arrives, so speech starts before synthesis finishes."""

    def __init__(self) -> None:
        self._sd = _sounddevice()

    def play(self, chunks: Iterable[bytes], sample_rate: int) -> None:
        with self._sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
            try:
                for frame in aligned_frames(chunks):
                    out.write(frame)
            except KeyboardInterrupt:
                out.abort()  # stop now instead of draining the buffer
                raise
