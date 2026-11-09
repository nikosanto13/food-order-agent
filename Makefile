.PHONY: install install-voice fmt lint typecheck test test-live check audit mock-server run run-voice evals

install:
	uv sync

install-voice:
	uv sync --extra voice

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

# --all-extras: the optional voice modules import elevenlabs/sounddevice.
typecheck:
	uv run --all-extras ty check

test:
	uv run pytest

# Hits real services (LLM / prod MCP / ElevenLabs) and spends credits.
test-live:
	uv run --all-extras pytest -m live

check: lint typecheck test

# Scan locked dependencies for known vulnerabilities (OSV). Needs network, so not part of `check`.
audit:
	uv audit --locked --preview-features audit-command

mock-server:
	uv run python -m food_agent.mock_server

run:
	uv run python main.py

run-voice:
	uv run --extra voice python main.py --voice

evals:
	uv run python -m evals.run
