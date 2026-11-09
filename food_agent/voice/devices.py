"""Microphone and speaker through ``sounddevice`` (PortAudio), imported lazily."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any, Self

from food_agent.voice.audio import SAMPLE_WIDTH, aligned_frames, loudness, split_frames
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
    """Always-on capture from the default input, delivered in fixed-size frames.

    ``on_audio`` runs on the audio thread for every frame; keep it cheap (the
    ``Listener`` just runs the endpointer). Use as a context manager.
    """

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        *,
        sample_rate: int = MIC_SAMPLE_RATE,
        frame_ms: int = 20,
    ) -> None:
        self._sd = _sounddevice()
        self._on_audio = on_audio
        self.sample_rate = sample_rate
        self._blocksize = sample_rate * frame_ms // 1000
        self._stream: Any = None

    def __enter__(self) -> Self:
        def callback(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            self._on_audio(bytes(indata))

        try:
            self._stream = self._sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._blocksize,
                callback=callback,
            )
            self._stream.start()
        except self._sd.PortAudioError as exc:
            raise VoiceUnavailableError(
                f"could not open the microphone ({exc}). Check that one is connected and that "
                "your terminal has microphone permission (macOS: System Settings > Privacy & "
                "Security > Microphone)."
            ) from exc
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None


class Speaker:
    """Plays a PCM stream as it arrives, so speech starts before synthesis finishes."""

    def __init__(self) -> None:
        self._sd = _sounddevice()

    def play(
        self,
        chunks: Iterable[bytes],
        sample_rate: int,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        """Play a PCM16 mono stream; report each 20 ms frame's loudness as it is queued."""
        frame_bytes = sample_rate // 50 * SAMPLE_WIDTH
        with self._sd.RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", latency="low"
        ) as out:
            try:
                for chunk in aligned_frames(chunks):
                    for frame in split_frames(chunk, frame_bytes):
                        if on_level is not None:
                            on_level(loudness(frame))
                        out.write(frame)
            except KeyboardInterrupt:
                out.abort()  # stop now instead of draining the buffer
                raise
