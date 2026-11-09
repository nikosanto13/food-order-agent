"""Hands-free turn taking: an energy-based endpointer with an adaptive noise floor.

The microphone streams 20 ms frames. A frame is *voiced* when it is clearly
louder than the room: ``margin_db`` above the noise floor, and never quieter
than ``min_speech_db``. A turn starts after ``start_ms`` of consecutive voiced
audio (a click or a cough is shorter) and ends after ``end_silence_ms`` of
silence. ``pre_roll_ms`` of audio from before the start is kept, so the first
syllable is not clipped. Turns with too little voiced audio are dropped, and
very long ones are cut at ``max_utterance_s``.

The noise floor is the quietest frame of the last few seconds ("minimum
statistics"). Speech has gaps between words, so talking does not raise it, while
a steady noise that starts later (a fan, the dishwasher) becomes the new floor
within ``noise_window_s`` and stops counting as speech. No model, no extra
dependency, and every decision can be tested with synthetic frames.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

from food_agent.voice.audio import SAMPLE_WIDTH, SILENCE_DB, Audio, loudness, rms_dbfs


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 20
    margin_db: float = 12.0  # how far above the room a frame must be to count as speech
    hysteresis_db: float = 3.0  # once talking, quieter syllables still count
    min_speech_db: float = -50.0  # never trigger on a very quiet room
    calibration_ms: int = 300  # listen to the room before the first turn can start
    start_ms: int = 100
    end_silence_ms: int = 900
    pre_roll_ms: int = 300
    min_voiced_ms: int = 250
    max_utterance_s: float = 30.0
    noise_window_s: float = 3.0


@dataclass(frozen=True)
class Utterance:
    """One customer turn, with what the latency trace needs to know about its end."""

    audio: Audio
    # Silence waited for after the last voiced frame before the turn was ended. This is
    # part of the response time the customer feels, before any network call starts.
    end_of_speech_wait_ms: int
    truncated: bool = False  # cut at max_utterance_s rather than ended by a pause
    detected_at: float = 0.0  # time.monotonic() when the turn was handed over

    @property
    def speech_ended_at(self) -> float:
        """When the customer actually stopped talking (monotonic)."""
        return self.detected_at - self.end_of_speech_wait_ms / 1000


class Endpointer:
    """Turns a stream of frames into utterances. Not thread-safe: see ``Listener``."""

    def __init__(self, config: VadConfig | None = None, sample_rate: int = 16_000) -> None:
        self.config = cfg = config or VadConfig()
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * cfg.frame_ms // 1000 * SAMPLE_WIDTH

        def frames(ms: float) -> int:
            return max(1, int(ms // cfg.frame_ms))

        self._calibration_frames = frames(cfg.calibration_ms)
        self._start_frames = frames(cfg.start_ms)
        self._end_frames = frames(cfg.end_silence_ms)
        self._min_voiced_frames = frames(cfg.min_voiced_ms)
        self._max_frames = frames(cfg.max_utterance_s * 1000)
        self._levels: deque[float] = deque(maxlen=frames(cfg.noise_window_s * 1000))
        self._pre_roll: deque[bytes] = deque(maxlen=frames(cfg.pre_roll_ms))
        self._utterance: list[bytes] = []
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor_db(self) -> float:
        return min(self._levels, default=SILENCE_DB)

    def reset(self) -> None:
        """Drop any partial turn. The noise floor is kept: the room has not changed."""
        self._in_speech = False
        self._run = 0
        self._voiced = 0
        self._silence = 0
        self._utterance = []
        self._pre_roll.clear()

    def feed(self, frame: bytes) -> Utterance | None:
        """Process one frame; return the utterance when a turn has just ended."""
        cfg = self.config
        db = rms_dbfs(frame)
        if db > SILENCE_DB:  # exact zeros (device start-up, muted input) say nothing about the room
            self._levels.append(db)
        threshold = max(self.noise_floor_db + cfg.margin_db, cfg.min_speech_db)
        if self._in_speech:
            threshold -= cfg.hysteresis_db
        voiced = db >= threshold

        if not self._in_speech:
            self._pre_roll.append(frame)
            calibrated = len(self._levels) >= self._calibration_frames
            self._run = self._run + 1 if voiced and calibrated else 0
            if self._run >= self._start_frames:
                self._in_speech = True
                self._utterance = list(self._pre_roll)
                self._voiced = self._run
                self._silence = 0
            return None

        self._utterance.append(frame)
        if voiced:
            self._voiced += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence < self._end_frames and len(self._utterance) < self._max_frames:
            return None

        # Turn over. Keep a short tail of the trailing silence, drop the rest.
        keep = len(self._utterance) - max(0, self._silence - self._start_frames)
        frames, voiced_frames = self._utterance[:keep], self._voiced
        wait_ms = self._silence * cfg.frame_ms
        truncated = self._silence < self._end_frames
        self.reset()
        if voiced_frames < self._min_voiced_frames:
            return None  # a bump or a cough, not a turn
        return Utterance(
            audio=Audio(b"".join(frames), self.sample_rate),
            end_of_speech_wait_ms=wait_ms,
            truncated=truncated,
        )


class Listener:
    """Bridges the microphone thread and the conversation.

    The audio callback calls ``feed()`` with whatever the device delivers; the
    conversation calls ``listen()`` when it is the customer's turn and
    ``next_utterance()`` to wait for it. Outside the customer's turn (the agent
    is thinking or speaking) audio is discarded, so the agent never transcribes
    itself from the speakers.
    """

    def __init__(
        self,
        endpointer: Endpointer,
        on_level: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._endpointer = endpointer
        self._on_level = on_level
        self._clock = clock
        self._lock = threading.Lock()
        self._utterances: queue.Queue[Utterance] = queue.Queue()
        self._active = False
        self._skip_frames = 0
        self._buffer = b""

    @property
    def in_speech(self) -> bool:
        return self._endpointer.in_speech

    def listen(self, ignore_ms: int = 0) -> None:
        """Start the customer's turn, ignoring the first ``ignore_ms`` (echo tail)."""
        with self._lock:
            self._endpointer.reset()
            self._buffer = b""
            self._skip_frames = ignore_ms // self._endpointer.config.frame_ms
            while not self._utterances.empty():
                self._utterances.get_nowait()
            self._active = True

    def pause(self) -> None:
        with self._lock:
            self._active = False

    def next_utterance(self, timeout: float) -> Utterance | None:
        try:
            return self._utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    def feed(self, data: bytes) -> None:
        """Called from the audio thread with raw PCM16 mono."""
        with self._lock:
            if not self._active:
                return
            self._buffer += data
            size = self._endpointer.frame_bytes
            while len(self._buffer) >= size:
                frame, self._buffer = self._buffer[:size], self._buffer[size:]
                if self._skip_frames:
                    self._skip_frames -= 1
                    continue
                if self._on_level is not None:
                    self._on_level(loudness(frame))
                utterance = self._endpointer.feed(frame)
                if utterance is not None:
                    self._active = False  # one utterance per turn
                    self._buffer = b""
                    self._utterances.put(replace(utterance, detected_at=self._clock()))
                    return
