"""Audio value type and PCM helpers (pure Python, no optional dependencies)."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

SAMPLE_WIDTH = 2  # bytes per sample: signed 16-bit little-endian PCM, mono, end to end


@dataclass(frozen=True)
class Audio:
    """A mono, 16-bit PCM recording."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (SAMPLE_WIDTH * self.sample_rate)

    def to_wav(self) -> bytes:
        """WAV container, so the STT service does not have to guess the raw format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(self.sample_rate)
            w.writeframes(self.pcm)
        return buf.getvalue()


def aligned_frames(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Re-chunk a PCM byte stream on sample boundaries.

    Streamed TTS chunks are split wherever the network splits them, so one can
    end halfway through a 16-bit sample. Writing that to the sound card would
    shift every later sample by a byte (loud noise), so the odd byte is carried
    over to the next chunk.
    """
    carry = b""
    for chunk in chunks:
        data = carry + chunk
        cut = len(data) - len(data) % SAMPLE_WIDTH
        carry = data[cut:]
        if cut:
            yield data[:cut]
