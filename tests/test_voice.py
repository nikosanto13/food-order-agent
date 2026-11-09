"""Voice front-end tests: offline, with fake speech services and audio devices."""

import builtins
import importlib
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from food_agent.config import Settings
from food_agent.prompts import build_static_prompt
from food_agent.voice import Audio, SpeechError, VoiceChannel, VoiceUnavailableError, to_speech
from food_agent.voice.audio import aligned_frames
from food_agent.voice.channel import build_voice_channel
from food_agent.voice.eleven import MAX_KEYTERMS, ElevenLabsSpeech, menu_keyterms

ONE_SECOND = Audio(b"\x00\x00" * 16_000, 16_000)

# ------------------------------------------------------------------ speakable text


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Total: $11.50", "Total: 11 dollars 50."),
        ("That's $9.00.", "That's 9 dollars."),
        ("$1.00 extra", "1 dollar extra."),
        ("Only $0.75", "Only 75 cents."),
        ("$11.05 total", "11 dollars and 5 cents total."),
        ("Order #ORD-48271.", "Order O R D, 4 8 2 7 1."),
        ("Estimated time: 15-20 minutes", "Estimated time: 15 to 20 minutes."),
        ("2x Large Classic Burger", "2 Large Classic Burger."),
        ("A diet_cola soda", "A diet cola soda."),
        ("**Done**", "Done."),
        ("Burger with cheese \u2014 $11.50", "Burger with cheese, 11 dollars 50."),
    ],
)
def test_to_speech_rewrites_what_tts_would_misread(text, spoken):
    assert to_speech(text) == spoken


def test_to_speech_flattens_summaries_into_sentences():
    reply = (
        "Your order:\n- Large Classic Burger with cheese: $11.50\n- Medium Soft Drink (cola): "
        "$2.00\nTotal: $13.50\nReady to submit?"
    )
    assert to_speech(reply) == (
        "Your order: Large Classic Burger with cheese: 11 dollars 50. "
        "Medium Soft Drink cola: 2 dollars. Total: 13 dollars 50. Ready to submit?"
    )


def test_to_speech_of_blank_text_is_empty():
    assert to_speech("\n  \n") == ""


# ------------------------------------------------------------------ audio helpers


def test_wav_roundtrip_keeps_format_and_duration():
    import io  # noqa: PLC0415
    import wave  # noqa: PLC0415

    with wave.open(io.BytesIO(ONE_SECOND.to_wav())) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16_000)
        assert w.getnframes() == 16_000
    assert ONE_SECOND.duration_s == pytest.approx(1.0)


def test_aligned_frames_never_split_a_sample():
    chunks = [b"\x01", b"\x02\x03\x04\x05", b"", b"\x06"]
    frames = list(aligned_frames(chunks))
    assert all(len(f) % 2 == 0 for f in frames)
    assert b"".join(frames) == b"\x01\x02\x03\x04\x05\x06"


# ------------------------------------------------------------------ ElevenLabs adapter


@dataclass
class FakeElevenLabsClient:
    """Mimics the two SDK calls we use: ``speech_to_text.convert`` and ``text_to_speech.stream``."""

    transcript: str = "a large margherita"
    error: Exception | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)

    @property
    def speech_to_text(self):
        return self

    @property
    def text_to_speech(self):
        return self

    def convert(self, **kwargs):
        self.calls.append(("stt", kwargs))
        if self.error:
            raise self.error
        return type("Transcript", (), {"text": f"  {self.transcript} "})()

    def stream(self, voice_id, **kwargs):
        self.calls.append(("tts", {"voice_id": voice_id, **kwargs}))
        yield b"\x00\x00"
        if self.error:
            raise self.error
        yield b"\x01\x00"


def make_speech(client, **overrides) -> ElevenLabsSpeech:
    kwargs: dict[str, Any] = {
        "voice_id": "voice",
        "tts_model": "eleven_flash_v2_5",
        "stt_model": "scribe_v2",
        "keyterms": ["Margherita"],
        "client": client,
    }
    return ElevenLabsSpeech("key", **(kwargs | overrides))


def test_transcribe_sends_wav_with_keyterms_and_strips_text():
    client = FakeElevenLabsClient()
    assert make_speech(client).transcribe(ONE_SECOND) == "a large margherita"
    (kind, kwargs), *_ = client.calls
    assert kind == "stt"
    assert kwargs["model_id"] == "scribe_v2"
    assert kwargs["keyterms"] == ["Margherita"]
    assert kwargs["no_verbatim"] is True
    name, data, mime = kwargs["file"]
    assert (name, mime) == ("speech.wav", "audio/wav")
    assert data.startswith(b"RIFF")


def test_transcribe_omits_v2_only_options_for_other_models():
    client = FakeElevenLabsClient()
    make_speech(client, stt_model="scribe_v1", keyterms=[]).transcribe(ONE_SECOND)
    kwargs = client.calls[0][1]
    assert "keyterms" not in kwargs
    assert "no_verbatim" not in kwargs


def test_synthesize_streams_pcm_at_the_declared_rate():
    client = FakeElevenLabsClient()
    speech = make_speech(client)
    assert list(speech.synthesize("Hi")) == [b"\x00\x00", b"\x01\x00"]
    assert client.calls[0][1]["output_format"] == f"pcm_{speech.sample_rate}"


def test_sdk_failures_become_speech_errors():
    client = FakeElevenLabsClient(error=ConnectionError("boom"))
    speech = make_speech(client)
    with pytest.raises(SpeechError, match="speech-to-text failed"):
        speech.transcribe(ONE_SECOND)
    with pytest.raises(SpeechError, match="text-to-speech failed"):
        list(speech.synthesize("Hi"))  # fails mid-stream, after the first chunk


def test_menu_keyterms_cover_hard_words_within_api_limits(menu):
    terms = menu_keyterms(menu)
    for word in ["Margherita Pizza", "Spicy Jalapeño Burger", "jalapeños", "oreo", "lemon lime"]:
        assert word in terms
    assert len(terms) <= MAX_KEYTERMS
    assert len({t.casefold() for t in terms}) == len(terms)
    assert all(len(t) < 50 and len(t.split()) <= 5 for t in terms)


# ------------------------------------------------------------------ voice channel


@dataclass
class FakeRecorder:
    recordings: list[Audio]

    def record(self, wait_for_stop: Callable[[], object], max_seconds: float) -> Audio:
        wait_for_stop()
        return self.recordings.pop(0)


@dataclass
class FakeSTT:
    results: list[str | Exception]
    heard: list[Audio] = field(default_factory=list)

    def transcribe(self, audio: Audio) -> str:
        self.heard.append(audio)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class FakeTTS:
    error: Exception | None = None
    sample_rate: int = 22_050
    spoken: list[str] = field(default_factory=list)

    def synthesize(self, text: str) -> Iterable[bytes]:
        self.spoken.append(text)
        yield b"\x00\x00"
        if self.error:
            raise self.error


@dataclass
class FakePlayer:
    error: BaseException | None = None
    played: list[bytes] = field(default_factory=list)

    def play(self, chunks: Iterable[bytes], sample_rate: int) -> None:
        if self.error:
            raise self.error
        self.played.extend(chunks)


def make_channel(*, typed, recordings=(), stt=(), tts=None, player=None):
    lines = list(typed)
    out: list[str] = []
    channel = VoiceChannel(
        stt=FakeSTT(list(stt)),
        tts=tts or FakeTTS(),
        recorder=FakeRecorder(list(recordings)),
        player=player or FakePlayer(),
        prompt=lambda _prompt: lines.pop(0),
        echo=out.append,
    )
    return channel, out


def test_typed_text_bypasses_the_microphone():
    channel, _ = make_channel(typed=["  /cart  "])
    assert channel.read() == "/cart"


def test_enter_records_and_returns_the_transcript():
    # Enter to start, Enter to stop.
    channel, out = make_channel(typed=["", ""], recordings=[ONE_SECOND], stt=["two fries"])
    assert channel.read() == "two fries"
    assert "You (voice): two fries" in out


def test_too_short_empty_or_failed_transcriptions_ask_again():
    blip = Audio(b"\x00\x00" * 100, 16_000)
    channel, out = make_channel(
        typed=["", "", "", "", "", "", "", ""],
        recordings=[blip, ONE_SECOND, ONE_SECOND, ONE_SECOND],
        stt=["", SpeechError("down"), "a cola"],
    )
    assert channel.read() == "a cola"
    joined = "\n".join(out)
    assert "too short" in joined
    assert "didn't catch anything" in joined
    assert "couldn't transcribe" in joined


def test_reply_prints_the_text_and_speaks_the_speakable_version():
    tts, player = FakeTTS(), FakePlayer()
    channel, out = make_channel(typed=[], tts=tts, player=player)
    channel.reply("Total: $11.50. Ready to submit?")
    assert out == ["Agent: Total: $11.50. Ready to submit?"]
    assert tts.spoken == ["Total: 11 dollars 50. Ready to submit?"]
    assert player.played == [b"\x00\x00"]


@pytest.mark.parametrize(
    ("tts_error", "player_error", "notice"),
    [
        (SpeechError("quota"), None, "Couldn't play the audio"),
        (None, KeyboardInterrupt(), "(stopped)"),
    ],
)
def test_audio_failures_never_end_the_conversation(tts_error, player_error, notice):
    channel, out = make_channel(
        typed=[], tts=FakeTTS(error=tts_error), player=FakePlayer(error=player_error)
    )
    channel.reply("Anything else?")
    assert out[0] == "Agent: Anything else?"
    assert any(notice in line for line in out[1:])


# ------------------------------------------------------------------ optional extra & wiring


def test_voice_prompt_is_opt_in(menu):
    assert "# Voice" not in build_static_prompt(menu)
    assert "speech-to-text transcripts" in build_static_prompt(menu, "voice")


def test_missing_api_key_is_reported_before_touching_devices(menu, tmp_path):
    settings = Settings(elevenlabs_api_key=None, log_dir=tmp_path)
    with pytest.raises(VoiceUnavailableError, match="ELEVENLABS_API_KEY"):
        build_voice_channel(settings, menu)


@pytest.fixture
def without_voice_extra(monkeypatch):
    """Make ``elevenlabs`` and ``sounddevice`` unimportable, as without the extra."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"elevenlabs", "sounddevice"}:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m.split(".")[0] in {"elevenlabs", "sounddevice"}]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.usefixtures("without_voice_extra")
def test_package_imports_without_the_extra_and_explains_how_to_install(menu, tmp_path, monkeypatch):
    for name in [m for m in sys.modules if m.startswith("food_agent.voice")]:
        monkeypatch.delitem(sys.modules, name)  # fresh import, originals restored afterwards
    voice = importlib.import_module("food_agent.voice")
    eleven = importlib.import_module("food_agent.voice.eleven")

    with pytest.raises(voice.VoiceUnavailableError, match="uv sync --extra voice"):
        eleven.ElevenLabsSpeech("k", voice_id="v", tts_model="t", stt_model="s")
    settings = Settings(elevenlabs_api_key="k", log_dir=tmp_path)
    with pytest.raises(voice.VoiceUnavailableError, match="uv sync --extra voice"):
        voice.build_voice_channel(settings, menu)
