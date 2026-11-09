.PHONY: install fmt lint typecheck test check audit mock-server run evals

install:
	uv sync

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest

check: lint typecheck test

# Scan locked dependencies for known vulnerabilities (OSV). Needs network, so not part of `check`.
audit:
	uv audit --locked --preview-features audit-command

mock-server:
	uv run python -m food_agent.mock_server

run:
	uv run python main.py

evals:
	uv run python -m evals.run
