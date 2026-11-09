"""Voice front-end tests: offline, with synthetic audio and fake services and devices."""

import builtins
import contextlib
import importlib
import io
import math
import random
import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from food_agent.config import Settings
from food_agent.prompts import build_static_prompt
from food_agent.voice import (
    Audio,
    HandsFreeChannel,
    SpeechError,
    VoiceUnavailableError,
    to_speech,
)
from food_agent.voice.audio import SILENCE_DB, aligned_frames, loudness, rms_dbfs, split_frames
from food_agent.voice.channel import build_voice_channel
from food_agent.voice.eleven import MAX_KEYTERMS, ElevenLabsSpeech, menu_keyterms
from food_agent.voice.meter import LevelMeter, bars
from food_agent.voice.vad import Endpointer, Listener, Utterance, VadConfig

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


# ------------------------------------------------------------------ turn detection

RATE = 16_000
FRAME = Endpointer(sample_rate=RATE).frame_bytes  # 20 ms


def tone(ms: int, amplitude: int = 8000) -> list[bytes]:
    """``ms`` of a loud-ish 400 Hz tone, as 20 ms frames (stands in for speech)."""
    n = RATE * ms // 1000
    samples = array(
        "h", (int(amplitude * math.sin(2 * math.pi * 400 * i / RATE)) for i in range(n))
    )
    return list(split_frames(samples.tobytes(), FRAME))


def hiss(ms: int, amplitude: int = 30) -> list[bytes]:
    """``ms`` of quiet noise (a room)."""
    rng = random.Random(ms)  # noqa: S311 - deterministic test noise, not crypto
    n = RATE * ms // 1000
    samples = array("h", (rng.randint(-amplitude, amplitude) for _ in range(n)))
    return list(split_frames(samples.tobytes(), FRAME))


def run(endpointer: Endpointer, frames: list[bytes]) -> list[Utterance]:
    return [u for f in frames if (u := endpointer.feed(f)) is not None]


def test_levels_separate_silence_room_and_speech():
    assert rms_dbfs(b"\x00\x00" * 320) == SILENCE_DB
    assert rms_dbfs(hiss(20)[0]) < -55
    assert rms_dbfs(tone(20)[0]) > -20
    assert loudness(hiss(20)[0]) == 0
    assert 0.5 < loudness(tone(20)[0]) <= 1


def test_speech_then_a_pause_is_one_turn_with_its_onset_kept():
    ep = Endpointer(sample_rate=RATE)
    turns = run(ep, hiss(1000) + tone(1000) + hiss(1500))
    assert len(turns) == 1
    # 1 s of speech, plus pre-roll before it and a short tail after it.
    assert 1.2 <= turns[0].audio.duration_s <= 1.6
    assert not ep.in_speech
    # The turn ended after exactly the configured pause: that wait is part of latency.
    assert turns[0].end_of_speech_wait_ms == ep.config.end_silence_ms
    assert not turns[0].truncated


def test_turn_does_not_end_on_short_gaps_between_words():
    ep = Endpointer(sample_rate=RATE)
    words = tone(400) + hiss(300) + tone(400) + hiss(300) + tone(400)
    turns = run(ep, hiss(1000) + words + hiss(1500))
    assert len(turns) == 1
    assert turns[0].audio.duration_s > 1.8


def test_clicks_and_coughs_are_not_turns():
    ep = Endpointer(sample_rate=RATE)
    assert run(ep, hiss(1000) + tone(60) + hiss(1500)) == []  # too short to start
    assert run(ep, hiss(500) + tone(160) + hiss(1500)) == []  # started, too little speech


def test_long_monologues_are_cut_at_the_limit():
    ep = Endpointer(VadConfig(max_utterance_s=2), sample_rate=RATE)
    turns = run(ep, hiss(1000) + tone(5000))
    assert turns
    assert turns[0].audio.duration_s == pytest.approx(2, abs=0.05)
    assert turns[0].truncated
    assert turns[0].end_of_speech_wait_ms == 0


def test_a_steady_noise_becomes_the_new_floor():
    ep = Endpointer(VadConfig(noise_window_s=1), sample_rate=RATE)
    fan = tone(6000, amplitude=2000)  # starts loud enough to look like speech
    turns = run(ep, hiss(1000) + fan)
    assert len(turns) == 1  # one bogus turn (STT returns nothing), then it is room noise
    assert ep.noise_floor_db > -40
    assert run(ep, tone(1000, amplitude=2000)) == []
    assert len(run(ep, tone(1000, amplitude=30000) + tone(1500, amplitude=2000))) == 1


def test_listener_only_hears_the_customers_turn():
    levels: list[float] = []
    listener = Listener(Endpointer(sample_rate=RATE), on_level=levels.append, clock=lambda: 100.0)
    speech = b"".join(hiss(1000) + tone(800) + hiss(1500))

    listener.feed(speech)  # not listening yet: e.g. the agent is speaking
    assert listener.next_utterance(timeout=0) is None
    assert levels == []

    listener.listen()
    for i in range(0, len(speech), 1234):  # devices deliver arbitrary block sizes
        listener.feed(speech[i : i + 1234])
    turn = listener.next_utterance(timeout=0)
    assert turn is not None
    assert turn.detected_at == 100.0
    assert turn.speech_ended_at == pytest.approx(100.0 - 0.9)
    assert max(levels) > 0.5
    listener.feed(speech)  # one utterance per turn until listen() is called again
    assert listener.next_utterance(timeout=0) is None


def test_listener_ignores_the_echo_right_after_the_agent_speaks():
    listener = Listener(Endpointer(sample_rate=RATE))
    listener.listen()
    listener.feed(b"".join(hiss(1000)))  # learn the room
    listener.listen(ignore_ms=1000)
    listener.feed(b"".join(tone(800) + hiss(1500)))  # the agent's voice, from the speakers
    assert listener.next_utterance(timeout=0) is None


# ------------------------------------------------------------------ level meter & display


def test_meter_scrolls_with_time_and_falls_flat_in_silence():
    now = [0.0]
    meter = LevelMeter(width=4, tick_s=0.05, clock=lambda: now[0])
    meter.push("you", 0.4)
    meter.push("you", 1.0)  # loudest level within a tick wins
    now[0] = 0.05
    assert meter.snapshot()["you"] == [0, 0, 0, 1.0]
    now[0] = 0.15  # two ticks without audio
    assert meter.snapshot() == {"you": [0, 1.0, 0, 0], "agent": [0, 0, 0, 0]}
    assert bars([0, 0.5, 1]) == " ▄█"


def test_display_renders_status_and_both_bars():
    pytest.importorskip("rich")
    from rich.console import Console  # noqa: PLC0415

    from food_agent.voice.display import RichDisplay  # noqa: PLC0415

    out = io.StringIO()
    console = Console(file=out, width=80, force_terminal=False, color_system=None)
    display = RichDisplay(console=console)
    display.set_status("speaking")
    display.level("agent", 1.0)
    display.say("agent", "Anything else?")
    console.print(display)
    text = out.getvalue()
    assert "Agent: Anything else?" in text
    assert "Speaking" in text
    assert "you" in text
    assert "agent" in text


# ------------------------------------------------------------------ hands-free channel


class Clock:
    """Manual monotonic clock: fakes advance it to simulate how long things take."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def said(audio: Audio = ONE_SECOND, *, at: float = 0.0, wait_ms: int = 900) -> Utterance:
    return Utterance(audio, end_of_speech_wait_ms=wait_ms, detected_at=at)


@dataclass
class FakeListener:
    turns: list[Utterance | None]
    calls: list[str] = field(default_factory=list)
    in_speech: bool = False

    def listen(self, ignore_ms: int = 0) -> None:
        self.calls.append(f"listen:{ignore_ms}")

    def pause(self) -> None:
        self.calls.append("pause")

    def next_utterance(self, timeout: float) -> Utterance | None:
        self.in_speech = not self.in_speech
        return self.turns.pop(0) if self.turns else None


@dataclass
class FakeSTT:
    results: list[str | Exception]
    clock: Clock | None = None
    latency_s: float = 0.0
    stt_model: str = "scribe_v2"
    keyterms: list[str] = field(default_factory=lambda: ["Margherita"])

    def transcribe(self, audio: Audio) -> str:
        if self.clock:
            self.clock.advance(self.latency_s)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class FakeTTS:
    error: Exception | None = None
    clock: Clock | None = None
    first_chunk_s: float = 0.0
    sample_rate: int = 22_050
    tts_model: str = "eleven_flash_v2_5"
    voice_id: str = "voice"
    spoken: list[str] = field(default_factory=list)

    def synthesize(self, text: str) -> Iterable[bytes]:
        self.spoken.append(text)
        if self.clock:
            self.clock.advance(self.first_chunk_s)
        yield b"\x00\x40" * 441
        if self.error:
            raise self.error


@dataclass
class FakePlayer:
    error: BaseException | None = None
    clock: Clock | None = None
    playback_s: float = 0.0
    played: list[bytes] = field(default_factory=list)

    def play(self, chunks: Iterable[bytes], sample_rate: int, on_level=None) -> None:
        if self.error:
            raise self.error
        for chunk in chunks:
            self.played.append(chunk)
            if on_level:
                on_level(loudness(chunk))
        if self.clock:
            self.clock.advance(self.playback_s)


@dataclass
class FakeDisplay:
    statuses: list[str] = field(default_factory=list)
    lines: list[tuple[str, str]] = field(default_factory=list)
    levels: list[tuple[str, float]] = field(default_factory=list)

    def set_status(self, status) -> None:
        if not self.statuses or self.statuses[-1] != status:
            self.statuses.append(status)

    def level(self, lane, level: float) -> None:
        self.levels.append((lane, level))

    def say(self, role, text: str) -> None:
        self.lines.append((role, text))


def make_channel(*, turns=(), stt=(), tts=None, player=None, stt_fake=None, clock=None):
    listener, display = FakeListener(list(turns)), FakeDisplay()
    channel = HandsFreeChannel(
        stt=stt_fake or FakeSTT(list(stt)),
        tts=tts or FakeTTS(),
        listener=listener,
        player=player or FakePlayer(),
        display=display,
        echo_guard_ms=300,
        clock=clock or Clock(),
    )
    return channel, listener, display


def test_read_waits_for_a_turn_and_returns_its_transcript():
    channel, listener, display = make_channel(turns=[None, None, said()], stt=["two fries"])
    assert channel.read() == "two fries"
    assert listener.calls == ["listen:0"]
    assert ("you", "two fries") in display.lines
    assert display.statuses == ["listening", "hearing", "listening", "transcribing", "thinking"]


def test_noise_is_ignored_and_failed_transcriptions_ask_again():
    channel, listener, display = make_channel(
        turns=[said(), said(), said()], stt=["", SpeechError("down"), "a cola"]
    )
    assert channel.read() == "a cola"
    assert listener.calls == ["listen:0"] * 3
    notes = [text for role, text in display.lines if role == "note"]
    assert len(notes) == 1
    assert "say it again" in notes[0]


def test_reply_pauses_the_mic_speaks_and_drives_the_agent_bar():
    tts, player = FakeTTS(), FakePlayer()
    channel, listener, display = make_channel(turns=[said()], stt=["yes"], tts=tts, player=player)
    channel.reply("Total: $11.50. Ready to submit?")
    assert listener.calls == ["pause"]
    assert ("agent", "Total: $11.50. Ready to submit?") in display.lines
    assert tts.spoken == ["Total: 11 dollars 50. Ready to submit?"]
    assert display.statuses == ["speaking"]
    assert display.levels
    assert all(lane == "agent" for lane, _ in display.levels)
    channel.read()  # the next turn skips the echo of what was just said
    assert listener.calls[-1] == "listen:300"


@pytest.mark.parametrize(
    ("tts_error", "player_error", "notice"),
    [
        (SpeechError("quota"), None, "Couldn't play the audio"),
        (None, KeyboardInterrupt(), "(stopped)"),
    ],
)
def test_audio_failures_never_end_the_conversation(tts_error, player_error, notice):
    channel, _, display = make_channel(
        tts=FakeTTS(error=tts_error), player=FakePlayer(error=player_error)
    )
    channel.reply("Anything else?")
    assert display.lines[0] == ("agent", "Anything else?")
    assert any(notice in text for role, text in display.lines if role == "note")


def test_channel_opens_and_closes_its_resources_in_order():
    events: list[str] = []

    @contextlib.contextmanager
    def resource(name):
        events.append(f"open {name}")
        yield
        events.append(f"close {name}")

    channel = HandsFreeChannel(
        stt=FakeSTT([]),
        tts=FakeTTS(),
        listener=FakeListener([]),
        player=FakePlayer(),
        display=FakeDisplay(),
        resources=(resource("display"), resource("mic")),
    )
    with channel:
        assert events == ["open display", "open mic"]
    assert events == ["open display", "open mic", "close mic", "close display"]


# ------------------------------------------------------------------ voice trace & cost


def test_a_turn_trace_breaks_down_the_response_time():
    clock = Clock()
    stt = FakeSTT(["a large margherita"], clock=clock, latency_s=0.4)
    tts = FakeTTS(clock=clock, first_chunk_s=0.3)
    player = FakePlayer(clock=clock, playback_s=2.0)
    # The endpointer handed the turn over 20 ms ago, after a 900 ms pause.
    turn = said(ONE_SECOND, at=clock.now - 0.02, wait_ms=900)
    channel, _, _ = make_channel(turns=[turn], stt_fake=stt, tts=tts, player=player, clock=clock)

    channel.read()
    clock.advance(1.2)  # agent.send()
    channel.reply("Large Margherita Pizza. Anything else?")
    (section,) = channel.trace_sections()
    record = section.to_record()

    assert section.name == "voice"
    assert record["latency_ms"] == {
        "total": pytest.approx(2820, abs=0.2),
        "end_of_speech": 900,
        "stt": pytest.approx(400),
        "agent": pytest.approx(1200),
        "tts_first_audio": pytest.approx(300),
        "other": pytest.approx(20, abs=0.2),  # the hand-off after detection
    }
    (attempt,) = record["stt"]["attempts"]
    assert attempt["outcome"] == "ok"
    assert record["stt"]["keyterms"] == 1
    assert record["tts"]["model"] == "eleven_flash_v2_5"
    assert record["tts"]["total_ms"] == pytest.approx(2300)
    # 1 s at $0.27/h (Scribe + keyterms), and the spoken text at $0.04 per 1K chars.
    spoken = len(tts.spoken[0])
    assert record["cost_usd"] == pytest.approx(0.27 / 3600 + spoken * 0.04 / 1000, abs=1e-6)
    assert channel.trace_sections() == ()  # handed over once


def test_retries_are_traced_and_billed_but_failed_calls_are_not():
    channel, _, _ = make_channel(
        turns=[said(), said(), said()], stt=["", SpeechError("down"), "a cola"]
    )
    channel.read()
    channel.reply("")  # nothing to speak
    (section,) = channel.trace_sections()
    attempts = section.to_record()["stt"]["attempts"]
    assert [a["outcome"] for a in attempts] == ["empty", "error", "ok"]
    assert attempts[1]["cost_usd"] == 0
    assert attempts[1]["error"] == "down"
    assert section.cost_usd == pytest.approx(2 * 0.27 / 3600, abs=1e-9)
    assert section.to_record()["latency_ms"] is None  # no audio was played


def test_unpriced_speech_models_make_the_cost_unknown_not_zero():
    channel, _, _ = make_channel(
        turns=[said()], stt_fake=FakeSTT(["hi"], stt_model="some_new_model")
    )
    channel.read()
    channel.reply("Hello!")
    (section,) = channel.trace_sections()
    assert section.cost_usd is None


@dataclass
class InterruptingPlayer:
    """Ctrl+C after ``chunks`` chunks have been played."""

    chunks: int

    def play(self, chunks: Iterable[bytes], sample_rate: int, on_level=None) -> None:
        for n, _ in enumerate(chunks, start=1):
            if n >= self.chunks:
                raise KeyboardInterrupt


@pytest.mark.parametrize(("played_chunks", "billed"), [(1, True), (0, False)])
def test_interrupted_speech_is_billed_once_audio_has_arrived(played_chunks, billed):
    player = InterruptingPlayer(played_chunks) if played_chunks else FakePlayer(KeyboardInterrupt())
    channel, _, _ = make_channel(turns=[said()], stt=["hi"], player=player)
    channel.read()
    channel.reply("Hello!")
    (section,) = channel.trace_sections()
    tts = section.to_record()["tts"]
    assert tts["outcome"] == "interrupted"
    assert (tts["cost_usd"] > 0) is billed


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
    """Make the voice extra's packages unimportable, as when it is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"elevenlabs", "sounddevice", "rich"}:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m.split(".")[0] in {"elevenlabs", "sounddevice", "rich"}]:
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
