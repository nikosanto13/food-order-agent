"""Per-turn voice trace: where the time went and what the speech APIs cost.

Attached to the agent's turn record as its ``voice`` section, so one record holds the
whole exchange. The headline number is the response time the customer feels, from the
moment they stop talking to the agent's first audio, split into its parts:

    end_of_speech   silence the endpointer waits for before ending the turn
    stt             upload + transcription (the accepted attempt)
    agent           agent.send(): LLM calls, tools, MCP submission
    tts_first_audio TTS request until the first PCM chunk arrives
    other           everything else: hand-off polling, text rewriting, ...

Re-tries are listed too (noise that transcribed to nothing, failed calls): they cost
time and money even though the customer only sees the accepted transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SttOutcome = Literal["ok", "empty", "error"]
TtsOutcome = Literal["ok", "interrupted", "error"]


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _sum_costs(costs: list[float | None]) -> float | None:
    """``None`` if anything could not be priced: a partial sum would under-report."""
    if any(c is None for c in costs):
        return None
    return round(sum(c for c in costs if c is not None), 6)


@dataclass(frozen=True)
class SttAttempt:
    audio_s: float
    end_of_speech_wait_ms: int
    truncated: bool
    latency_ms: float
    chars: int
    cost_usd: float | None
    outcome: SttOutcome
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "outcome": self.outcome,
            "audio_s": round(self.audio_s, 2),
            "end_of_speech_wait_ms": self.end_of_speech_wait_ms,
            "latency_ms": self.latency_ms,
            "chars": self.chars,
            "cost_usd": self.cost_usd,
        }
        if self.truncated:
            record["truncated"] = True
        if self.error:
            record["error"] = self.error
        return record


@dataclass(frozen=True)
class TtsResult:
    chars: int
    first_audio_ms: float | None  # request -> first PCM chunk (time to first byte)
    total_ms: float  # request -> playback finished
    cost_usd: float | None
    outcome: TtsOutcome
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "outcome": self.outcome,
            "chars": self.chars,
            "first_audio_ms": self.first_audio_ms,
            "total_ms": self.total_ms,
            "cost_usd": self.cost_usd,
        }
        if self.error:
            record["error"] = self.error
        return record


@dataclass
class VoiceTurnTrace:
    """Filled in by the voice channel across ``read()`` and ``reply()``."""

    stt_model: str
    keyterms: int
    tts_model: str
    voice_id: str
    name: str = "voice"  # key of this section in the turn record
    attempts: list[SttAttempt] = field(default_factory=list)
    # monotonic timestamps of the accepted turn
    speech_ended_at: float | None = None
    transcribed_at: float | None = None
    reply_started_at: float | None = None
    tts_requested_at: float | None = None
    first_audio_at: float | None = None
    tts: TtsResult | None = None

    @property
    def cost_usd(self) -> float | None:
        return _sum_costs(list(self.cost_breakdown().values()))

    def cost_breakdown(self) -> dict[str, float | None]:
        return {
            "stt": _sum_costs([a.cost_usd for a in self.attempts]),
            "tts": 0.0 if self.tts is None else self.tts.cost_usd,
        }

    def latency_breakdown(self) -> dict[str, float] | None:
        """Response time split into parts, or ``None`` if no audio was played."""
        accepted = self.attempts[-1] if self.attempts else None
        if (
            accepted is None
            or accepted.outcome != "ok"
            or self.speech_ended_at is None
            or self.transcribed_at is None
            or self.reply_started_at is None
            or self.tts_requested_at is None
            or self.first_audio_at is None
        ):
            return None
        parts = {
            "end_of_speech": float(accepted.end_of_speech_wait_ms),
            "stt": accepted.latency_ms,
            "agent": _ms(self.reply_started_at - self.transcribed_at),
            "tts_first_audio": _ms(self.first_audio_at - self.tts_requested_at),
        }
        total = _ms(self.first_audio_at - self.speech_ended_at)
        return {
            "total": total,
            **parts,
            "other": round(max(0.0, total - sum(parts.values())), 1),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_breakdown(),
            "stt": {
                "model": self.stt_model,
                "keyterms": self.keyterms,
                "attempts": [a.to_record() for a in self.attempts],
            },
            "tts": (
                None
                if self.tts is None
                else {"model": self.tts_model, "voice_id": self.voice_id, **self.tts.to_record()}
            ),
            "cost_usd": self.cost_usd,
        }
