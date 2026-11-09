"""The voice conversation channel: push-to-talk in, spoken replies out.

``main.py`` drives the REPL through a two-method channel (``read`` / ``reply``);
this is the voice implementation. Each turn:

    Enter -> record until Enter -> STT -> (agent.send) -> speakable text -> TTS -> speaker

Typing a message instead of pressing Enter still works, as do the slash commands.
Speech failures never end the conversation: a failed transcription asks the
customer to try again, a failed synthesis falls back to the printed reply.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

import structlog

from food_agent.config import Settings
from food_agent.menu import Menu
from food_agent.voice.audio import Audio
from food_agent.voice.devices import Microphone, Speaker
from food_agent.voice.eleven import ElevenLabsSpeech, menu_keyterms
from food_agent.voice.errors import SpeechError, VoiceUnavailableError
from food_agent.voice.speakable import to_speech

log = structlog.get_logger(__name__)

MIN_SPEECH_S = 0.3  # shorter than this is a double-tap on Enter, not speech
PROMPT = "You (Enter to talk, or type): "


class SpeechToText(Protocol):
    def transcribe(self, audio: Audio) -> str: ...


class TextToSpeech(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> Iterable[bytes]: ...


class Recorder(Protocol):
    def record(self, wait_for_stop: Callable[[], object], max_seconds: float) -> Audio: ...


class Player(Protocol):
    def play(self, chunks: Iterable[bytes], sample_rate: int) -> None: ...


class VoiceChannel:
    def __init__(
        self,
        *,
        stt: SpeechToText,
        tts: TextToSpeech,
        recorder: Recorder,
        player: Player,
        max_record_s: float = 30,
        prompt: Callable[[str], str] = input,
        echo: Callable[[str], object] = print,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._recorder = recorder
        self._player = player
        self._max_record_s = max_record_s
        self._prompt = prompt
        self._echo = echo

    def read(self) -> str:
        """Next user message: typed text as-is, or a transcribed recording."""
        while True:
            typed = self._prompt(PROMPT).strip()
            if typed:
                return typed
            self._echo("  Listening... press Enter when you're done.")
            audio = self._recorder.record(lambda: self._prompt(""), self._max_record_s)
            if audio.duration_s < MIN_SPEECH_S:
                self._echo("  (That was too short. Press Enter, speak, then press Enter again.)")
                continue
            text = self._transcribe(audio)
            if text:
                self._echo(f"You (voice): {text}")
                return text

    def reply(self, message: str) -> None:
        """Print the reply, then speak it. Ctrl+C stops the audio, not the session."""
        self._echo(f"Agent: {message}")
        spoken = to_speech(message)
        if not spoken:
            return
        started = time.monotonic()
        first_audio: list[float] = []

        def timed(chunks: Iterable[bytes]) -> Iterator[bytes]:
            for chunk in chunks:
                if not first_audio:
                    first_audio.append(time.monotonic())
                yield chunk

        try:
            self._player.play(timed(self._tts.synthesize(spoken)), self._tts.sample_rate)
        except SpeechError as exc:
            log.warning("voice.tts_failed", error=str(exc))
            self._echo("  (Couldn't play the audio; the reply is shown above.)")
            return
        except KeyboardInterrupt:
            log.info("voice.tts_interrupted")
            self._echo("  (stopped)")
            return
        log.info(
            "voice.tts",
            chars=len(spoken),
            first_audio_ms=round((first_audio[0] - started) * 1000) if first_audio else None,
            total_ms=round((time.monotonic() - started) * 1000),
        )

    def _transcribe(self, audio: Audio) -> str:
        started = time.monotonic()
        try:
            text = self._stt.transcribe(audio)
        except SpeechError as exc:
            log.warning("voice.stt_failed", audio_s=round(audio.duration_s, 2), error=str(exc))
            self._echo("  Sorry, I couldn't transcribe that. Try again, or type your message.")
            return ""
        log.info(
            "voice.stt",
            audio_s=round(audio.duration_s, 2),
            latency_ms=round((time.monotonic() - started) * 1000),
            chars=len(text),
        )
        if not text:
            self._echo("  (I didn't catch anything. Try again, or type your message.)")
        return text


def build_voice_channel(settings: Settings, menu: Menu) -> VoiceChannel:
    """Wire the real devices and ElevenLabs. Raises ``VoiceUnavailableError`` early."""
    key = settings.elevenlabs_api_key
    if key is None:
        raise VoiceUnavailableError("ELEVENLABS_API_KEY is not set (see .env.example).")
    recorder, player = Microphone(), Speaker()  # fail fast on a missing extra / PortAudio
    speech = ElevenLabsSpeech(
        key.get_secret_value(),
        voice_id=settings.elevenlabs_voice_id,
        tts_model=settings.elevenlabs_tts_model,
        stt_model=settings.elevenlabs_stt_model,
        language_code=settings.voice_language,
        keyterms=menu_keyterms(menu) if settings.voice_stt_keyterms else (),
        timeout_s=settings.voice_timeout_s,
    )
    return VoiceChannel(
        stt=speech,
        tts=speech,
        recorder=recorder,
        player=player,
        max_record_s=settings.voice_max_record_s,
    )
