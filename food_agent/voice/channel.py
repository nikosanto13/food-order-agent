"""Hands-free voice channel: just talk; pause to end your turn.

``main.py`` drives the REPL through a channel (``read`` / ``reply``, used as a
context manager); this is the voice implementation. Each turn:

    mic (always on) -> endpointer: speech ... silence -> STT -> (agent.send)
        -> speakable text -> TTS -> speaker, with live "you" / "agent" level bars

Turn-taking is half-duplex: the microphone is ignored while the agent thinks and
speaks, and for a moment afterwards, so it never transcribes its own voice from
the speakers. Speech failures never end the conversation: a failed transcription
asks the customer to say it again, a failed synthesis leaves the printed reply.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, Self

import structlog

from food_agent.config import Settings
from food_agent.menu import Menu
from food_agent.pricing import STT_PRICING, TTS_PRICING, estimate_stt_cost, estimate_tts_cost
from food_agent.voice.audio import Audio
from food_agent.voice.devices import MIC_SAMPLE_RATE, Microphone, Speaker
from food_agent.voice.display import RichDisplay, Role, Status
from food_agent.voice.eleven import ElevenLabsSpeech, menu_keyterms
from food_agent.voice.errors import SpeechError, VoiceUnavailableError
from food_agent.voice.meter import Lane
from food_agent.voice.speakable import to_speech
from food_agent.voice.trace import SttAttempt, SttOutcome, TtsOutcome, TtsResult, VoiceTurnTrace
from food_agent.voice.vad import Endpointer, Listener, Utterance, VadConfig

logger = structlog.get_logger(__name__)

ECHO_GUARD_MS = 300  # after the agent speaks, ignore the room's echo of it
POLL_S = 0.05


class SpeechToText(Protocol):
    stt_model: str

    @property
    def keyterms(self) -> Sequence[str]: ...

    def transcribe(self, audio: Audio) -> str: ...


class TextToSpeech(Protocol):
    tts_model: str
    voice_id: str
    sample_rate: int

    def synthesize(self, text: str) -> Iterable[bytes]: ...


class TurnListener(Protocol):
    @property
    def in_speech(self) -> bool: ...
    def listen(self, ignore_ms: int = 0) -> None: ...
    def pause(self) -> None: ...
    def next_utterance(self, timeout: float) -> Utterance | None: ...


class Player(Protocol):
    def play(
        self,
        chunks: Iterable[bytes],
        sample_rate: int,
        on_level: Callable[[float], None] | None = None,
    ) -> None: ...


class Display(Protocol):
    def set_status(self, status: Status) -> None: ...
    def level(self, lane: Lane, level: float) -> None: ...
    def say(self, role: Role, text: str) -> None: ...


class HandsFreeChannel:
    def __init__(
        self,
        *,
        stt: SpeechToText,
        tts: TextToSpeech,
        listener: TurnListener,
        player: Player,
        display: Display,
        resources: Sequence[AbstractContextManager[object]] = (),
        echo_guard_ms: int = ECHO_GUARD_MS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._listener = listener
        self._player = player
        self._display = display
        self._resources = resources
        self._echo_guard_ms = echo_guard_ms
        self._just_spoke = False
        self._stack = contextlib.ExitStack()
        self._clock = clock
        self._trace: VoiceTurnTrace | None = None

    def __enter__(self) -> Self:
        with contextlib.ExitStack() as stack:
            for resource in self._resources:  # e.g. the live display, then the microphone
                stack.enter_context(resource)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._listener.pause()
        self._stack.close()

    def read(self) -> str:
        """Block until the customer has said something; return the transcript."""
        trace = self._new_trace()
        while True:
            self._listener.listen(ignore_ms=self._echo_guard_ms if self._just_spoke else 0)
            self._just_spoke = False
            utterance = self._wait_for_utterance()
            self._display.set_status("transcribing")
            text = self._transcribe(utterance, trace)
            if text:
                trace.speech_ended_at = utterance.speech_ended_at
                trace.transcribed_at = self._clock()
                self._display.say("you", text)
                self._display.set_status("thinking")
                return text

    def trace_sections(self) -> tuple[VoiceTurnTrace, ...]:
        """The ``voice`` section for the turn just completed (once per turn)."""
        trace, self._trace = self._trace, None
        return () if trace is None else (trace,)

    def reply(self, message: str) -> None:
        """Show the reply and speak it. Ctrl+C stops the audio, not the session."""
        trace = self._trace or self._new_trace()
        trace.reply_started_at = self._clock()
        self._listener.pause()
        self._display.say("agent", message)
        spoken = to_speech(message)
        if not spoken:
            return
        self._display.set_status("speaking")
        self._just_spoke = True

        def timed(chunks: Iterable[bytes]) -> Iterator[bytes]:
            for chunk in chunks:
                if trace.first_audio_at is None:
                    trace.first_audio_at = self._clock()
                yield chunk

        trace.tts_requested_at = started = self._clock()
        outcome: TtsOutcome = "ok"
        error: str | None = None
        try:
            self._player.play(
                timed(self._tts.synthesize(spoken)),
                self._tts.sample_rate,
                on_level=lambda level: self._display.level("agent", level),
            )
        except SpeechError as exc:
            outcome, error = "error", str(exc)
            logger.warning("voice.tts_failed", error=error)
            self._display.say("note", "(Couldn't play the audio; the reply is shown above.)")
        except KeyboardInterrupt:
            outcome = "interrupted"
            logger.info("voice.tts_interrupted")
            self._display.say("note", "(stopped)")
        first_audio_ms = (
            None
            if trace.first_audio_at is None
            else round((trace.first_audio_at - started) * 1000, 1)
        )
        # Billed per character requested; a request that produced no audio is assumed free.
        cost = (
            estimate_tts_cost(self._tts.tts_model, chars=len(spoken))
            if trace.first_audio_at is not None
            else 0.0
        )
        trace.tts = TtsResult(
            chars=len(spoken),
            first_audio_ms=first_audio_ms,
            total_ms=round((self._clock() - started) * 1000, 1),
            cost_usd=cost,
            outcome=outcome,
            error=error,
        )
        logger.info("voice.tts", **trace.tts.to_record())

    def _new_trace(self) -> VoiceTurnTrace:
        self._trace = VoiceTurnTrace(
            stt_model=self._stt.stt_model,
            keyterms=len(self._stt.keyterms),
            tts_model=self._tts.tts_model,
            voice_id=self._tts.voice_id,
        )
        return self._trace

    def _wait_for_utterance(self) -> Utterance:
        # Short polls keep Ctrl+C responsive and the status line current.
        while True:
            self._display.set_status("hearing" if self._listener.in_speech else "listening")
            utterance = self._listener.next_utterance(timeout=POLL_S)
            if utterance is not None:
                return utterance

    def _transcribe(self, utterance: Utterance, trace: VoiceTurnTrace) -> str:
        audio = utterance.audio
        started = self._clock()
        outcome: SttOutcome = "ok"
        error: str | None = None
        try:
            text = self._stt.transcribe(audio)
        except SpeechError as exc:
            text, outcome, error = "", "error", str(exc)
            self._display.say("note", "Sorry, I couldn't transcribe that. Please say it again.")
        else:
            if not text:
                outcome = "empty"  # background noise, not words: keep listening quietly
        attempt = SttAttempt(
            audio_s=audio.duration_s,
            end_of_speech_wait_ms=utterance.end_of_speech_wait_ms,
            truncated=utterance.truncated,
            latency_ms=round((self._clock() - started) * 1000, 1),
            chars=len(text),
            # A failed request is assumed not to be billed.
            cost_usd=0.0
            if outcome == "error"
            else estimate_stt_cost(
                self._stt.stt_model,
                audio_s=audio.duration_s,
                keyterms=bool(self._stt.keyterms),
            ),
            outcome=outcome,
            error=error,
        )
        trace.attempts.append(attempt)
        logger.log(30 if error else 20, "voice.stt", **attempt.to_record())
        return text


def build_voice_channel(settings: Settings, menu: Menu) -> HandsFreeChannel:
    """Wire the real devices, UI and ElevenLabs. Raises ``VoiceUnavailableError`` early."""
    key = settings.elevenlabs_api_key
    if key is None:
        raise VoiceUnavailableError("ELEVENLABS_API_KEY is not set (see .env.example).")
    vad = VadConfig(
        margin_db=settings.voice_vad_margin_db,
        end_silence_ms=settings.voice_end_silence_ms,
        max_utterance_s=settings.voice_max_utterance_s,
    )
    if settings.elevenlabs_stt_model not in STT_PRICING:
        logger.warning("voice.pricing_missing", model=settings.elevenlabs_stt_model)
    if settings.elevenlabs_tts_model not in TTS_PRICING:
        logger.warning("voice.pricing_missing", model=settings.elevenlabs_tts_model)
    endpointer = Endpointer(vad, sample_rate=MIC_SAMPLE_RATE)
    display = RichDisplay()
    listener = Listener(endpointer, on_level=lambda level: display.level("you", level))
    microphone = Microphone(listener.feed, sample_rate=MIC_SAMPLE_RATE, frame_ms=vad.frame_ms)
    speech = ElevenLabsSpeech(
        key.get_secret_value(),
        voice_id=settings.elevenlabs_voice_id,
        tts_model=settings.elevenlabs_tts_model,
        stt_model=settings.elevenlabs_stt_model,
        language_code=settings.voice_language,
        keyterms=menu_keyterms(menu) if settings.voice_stt_keyterms else (),
        timeout_s=settings.voice_timeout_s,
    )
    return HandsFreeChannel(
        stt=speech,
        tts=speech,
        listener=listener,
        player=Speaker(),
        display=display,
        resources=(display, microphone),
    )
