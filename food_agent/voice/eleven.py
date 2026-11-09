"""ElevenLabs speech-to-text (Scribe) and text-to-speech (Flash), behind small protocols.

The SDK is imported only when ``ElevenLabsSpeech`` is created without an injected
client, so this module imports cleanly without the ``voice`` extra.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from food_agent.menu import Menu
from food_agent.voice.audio import Audio
from food_agent.voice.errors import INSTALL_HINT, SpeechError, VoiceUnavailableError

TTS_SAMPLE_RATE = 22_050
MAX_KEYTERMS = 100  # API limits: at most 100 terms, each < 50 chars and <= 5 words
_MAX_KEYTERM_CHARS = 49
_MAX_KEYTERM_WORDS = 5


def menu_keyterms(menu: Menu) -> list[str]:
    """Words the recogniser should expect: item names, option choices and extras.

    Biases transcription towards "margherita" over "margarita", "jalapeños" over
    "hollow pinos", and so on. Derived from the menu, so it never goes stale.
    """
    terms: list[str] = []
    for item in menu.items:
        terms.append(item.name)
        for option in item.options.values():
            terms.extend(choice.replace("_", " ") for choice in option.choices)
        if item.extras:
            terms.extend(extra.id.replace("_", " ") for extra in item.extras.choices)
    seen: set[str] = set()
    keyterms: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen or len(term) > _MAX_KEYTERM_CHARS:
            continue
        if len(term.split()) > _MAX_KEYTERM_WORDS:
            continue
        seen.add(key)
        keyterms.append(term)
    return keyterms[:MAX_KEYTERMS]


class ElevenLabsSpeech:
    """One ElevenLabs client serving both directions.

    Every SDK or transport failure is re-raised as ``SpeechError``, so the voice
    loop can apologise and carry on rather than crash mid-order.
    """

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str,
        tts_model: str,
        stt_model: str,
        language_code: str = "en",
        keyterms: Sequence[str] = (),
        timeout_s: float = 30,
        sample_rate: int = TTS_SAMPLE_RATE,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from elevenlabs.client import ElevenLabs  # noqa: PLC0415 - optional extra
            except ImportError as exc:
                raise VoiceUnavailableError(INSTALL_HINT) from exc
            client = ElevenLabs(api_key=api_key, timeout=timeout_s)
        self._client = client
        self.voice_id = voice_id
        self.tts_model = tts_model
        self.stt_model = stt_model
        self.language_code = language_code
        self.keyterms = list(keyterms)
        self.sample_rate = sample_rate

    def transcribe(self, audio: Audio) -> str:
        options: dict[str, Any] = {}
        if self.keyterms:
            options["keyterms"] = self.keyterms
        if self.stt_model.startswith("scribe_v2"):
            options["no_verbatim"] = True  # drop "um", false starts: cleaner input for the LLM
        try:
            response = self._client.speech_to_text.convert(
                model_id=self.stt_model,
                file=("speech.wav", audio.to_wav(), "audio/wav"),
                language_code=self.language_code,
                tag_audio_events=False,  # no "(laughs)" in what the agent reads
                **options,
            )
        except Exception as exc:  # any SDK/transport failure
            raise SpeechError(f"speech-to-text failed: {type(exc).__name__}: {exc}") from exc
        return str(getattr(response, "text", "") or "").strip()

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Stream raw PCM (16-bit mono at ``sample_rate``) as it is generated."""
        try:
            yield from self._client.text_to_speech.stream(
                self.voice_id,
                text=text,
                model_id=self.tts_model,
                output_format=f"pcm_{self.sample_rate}",
            )
        except Exception as exc:  # any SDK/transport failure
            raise SpeechError(f"text-to-speech failed: {type(exc).__name__}: {exc}") from exc
