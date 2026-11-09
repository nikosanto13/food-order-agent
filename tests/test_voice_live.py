"""Live ElevenLabs check: synthesize a known order, transcribe it back.

Run with ``uv run pytest -m live tests/test_voice_live.py`` (needs the ``voice``
extra and ELEVENLABS_API_KEY). Costs a few hundred characters of credits.
"""

import pytest

from food_agent.config import Settings
from food_agent.voice.audio import Audio
from food_agent.voice.eleven import ElevenLabsSpeech, menu_keyterms

pytestmark = pytest.mark.live


def test_tts_then_stt_roundtrip(menu):
    pytest.importorskip("elevenlabs")
    settings = Settings()
    if settings.elevenlabs_api_key is None:
        pytest.skip("ELEVENLABS_API_KEY is not set")
    speech = ElevenLabsSpeech(
        settings.elevenlabs_api_key.get_secret_value(),
        voice_id=settings.elevenlabs_voice_id,
        tts_model=settings.elevenlabs_tts_model,
        stt_model=settings.elevenlabs_stt_model,
        keyterms=menu_keyterms(menu),
    )
    pcm = b"".join(speech.synthesize("A large margherita with olives and an oreo milkshake."))
    audio = Audio(pcm[: len(pcm) - len(pcm) % 2], speech.sample_rate)
    assert audio.duration_s > 1

    text = speech.transcribe(audio).lower()
    for word in ["large", "margherita", "olives", "oreo", "milkshake"]:
        assert word in text, text
