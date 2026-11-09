"""Runtime configuration (environment variables / ``.env``), validated at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    mistral_api_key: SecretStr | None = None  # SecretStr: never rendered in logs/reprs
    mistral_model: str = "mistral-small-latest"
    llm_temperature: float = Field(default=0.2, ge=0, le=1)
    llm_timeout_s: float = Field(default=30, gt=0)
    llm_max_output_tokens: int = Field(default=600, gt=0)

    # --- MCP order server ---
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    applicant_email: str | None = None
    mcp_timeout_s: float = Field(default=15, gt=0)
    mcp_max_attempts: int = Field(default=3, ge=1, le=5)
    mcp_force_failure: Literal["kitchen_busy", "server_error"] | None = None

    # --- Agent loop budgets ---
    max_tool_iterations: int = Field(default=8, ge=1)
    max_history_messages: int = Field(default=40, ge=4)
    max_user_message_chars: int = Field(default=1000, ge=10)

    # --- Replies ---
    # "voice": short, speakable replies, and the model is told input is a transcript.
    reply_style: Literal["text", "voice"] = "text"

    # --- Voice (optional: `uv sync --extra voice`, then `python main.py --voice`) ---
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # "George", a premade voice
    elevenlabs_tts_model: str = "eleven_flash_v2_5"  # lowest-latency TTS
    elevenlabs_stt_model: str = "scribe_v2"
    voice_language: str = "en"
    voice_stt_keyterms: bool = True  # bias STT towards menu words (billed extra by ElevenLabs)
    voice_end_silence_ms: int = Field(default=900, ge=200, le=5000)  # pause that ends a turn
    voice_vad_margin_db: float = Field(default=12, ge=3, le=40)  # how far above room noise
    voice_max_utterance_s: float = Field(default=30, gt=0, le=300)
    voice_timeout_s: float = Field(default=30, gt=0)

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")
    log_to_stderr: bool = False
