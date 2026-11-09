"""Audio value type and PCM helpers (pure Python, no optional dependencies)."""

from __future__ import annotations

import io
import math
import sys
import wave
from array import array
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


SILENCE_DB = -96.0  # level reported for digital silence (and empty input)
_METER_FLOOR_DB = -55.0  # quieter than this shows as an empty bar
_METER_CEIL_DB = -12.0  # louder than this shows as a full bar


def rms_dbfs(pcm: bytes) -> float:
    """Loudness of a PCM16 frame in dB relative to full scale (0 = clipping)."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH])
    if sys.byteorder == "big":
        samples.byteswap()  # PCM on the wire is little-endian
    if not samples:
        return SILENCE_DB
    mean_square = sum(s * s for s in samples) / len(samples)
    if mean_square == 0:
        return SILENCE_DB
    return max(SILENCE_DB, 10 * math.log10(mean_square / 32768**2))


def loudness(pcm: bytes) -> float:
    """Frame loudness mapped to 0..1 for the level bars (speech lands mid-range)."""
    db = rms_dbfs(pcm)
    return min(1.0, max(0.0, (db - _METER_FLOOR_DB) / (_METER_CEIL_DB - _METER_FLOOR_DB)))


def split_frames(pcm: bytes, frame_bytes: int) -> Iterator[bytes]:
    for start in range(0, len(pcm), frame_bytes):
        yield pcm[start : start + frame_bytes]
