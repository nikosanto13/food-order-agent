"""Interactive entry point: ``python main.py [--local] [--voice] [--verbose] ...``."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Protocol, Self

from food_agent.agent import FoodOrderAgent
from food_agent.config import Settings
from food_agent.local_server import BackgroundMockServer
from food_agent.menu import Menu
from food_agent.observability import SessionUsage, TraceSection, configure_logging

HELP = "Commands: /cart  /reset  /trace  /help  /quit"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with the food-ordering agent.")
    p.add_argument(
        "--local",
        action="store_true",
        help="start the local mock MCP server in-process and submit orders to it",
    )
    p.add_argument(
        "--voice",
        action="store_true",
        help="hands-free voice mode (needs `uv sync --extra voice` and ELEVENLABS_API_KEY)",
    )
    p.add_argument("--busy-rate", type=float, default=0.1, help="mock kitchen-busy rate (--local)")
    p.add_argument("--verbose", action="store_true", help="stream JSON logs to stderr")
    p.add_argument(
        "--force-failure",
        choices=["kitchen_busy", "server_error"],
        help="send X-Demo-Force-Failure on every submit (failure drills)",
    )
    return p.parse_args()


class Channel(Protocol):
    """Where user messages come from and replies go: the keyboard, or voice.

    A context manager, so a voice channel can own its microphone and live display.
    """

    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> None: ...
    def read(self) -> str: ...
    def reply(self, message: str) -> None: ...
    def trace_sections(self) -> tuple[TraceSection, ...]:
        """What this channel adds to the turn's trace record (and cost)."""
        ...


class TextChannel:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> str:
        return input("You: ").strip()

    def reply(self, message: str) -> None:
        print(f"Agent: {message}")

    def trace_sections(self) -> tuple[TraceSection, ...]:
        return ()


def _open_voice_channel(settings: Settings, menu: Menu, stack: contextlib.ExitStack) -> Channel:
    # Imported here so text mode never depends on the optional voice extra.
    from food_agent.voice import VoiceUnavailableError, build_voice_channel  # noqa: PLC0415

    try:
        channel = stack.enter_context(build_voice_channel(settings, menu))
    except VoiceUnavailableError as exc:
        raise SystemExit(f"Voice mode unavailable: {exc}") from exc
    print("Hands-free voice mode: just talk, and pause when you're done. Ctrl+C to quit.")
    return channel


def _repl(agent: FoodOrderAgent, channel: Channel) -> None:
    commands = f" {HELP}" if isinstance(channel, TextChannel) else ""
    print(f"Food ordering assistant ({agent.llm.model}).{commands}")
    print(f"Conversation: {agent.conversation_id}\n")
    while True:
        try:
            user = channel.read()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user in {"/quit", "/exit"}:
            return
        if user == "/help":
            print(HELP)
            continue
        if user == "/cart":
            print(agent.session.cart.summary_text())
            continue
        if user == "/reset":
            agent.reset()
            print(f"New conversation: {agent.conversation_id}")
            continue
        if user == "/trace":
            print(agent.settings.log_dir / "conversations" / f"{agent.conversation_id}.jsonl")
            continue

        # The turn stays open while the reply is delivered, so the channel's own
        # timings and cost (e.g. speech) land in the same trace record.
        response = agent.send(user, keep_turn_open=True)
        try:
            channel.reply(response["message"])
            for call in response.get("tool_calls", []):
                result = json.dumps(call["result"], ensure_ascii=False)
                print(f"  [{call['name']} -> {result}]")
            print()
        finally:
            agent.close_turn(*channel.trace_sections())


def _dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m" if sys.stdout.isatty() else text


def _usd(cost: float | None) -> str:
    return "n/a (unpriced model)" if cost is None else f"${cost:.6f}"


def session_cost_summary(usage: SessionUsage) -> str:
    """Estimated USD spend of the session, by source (LLM, speech-to-text, text-to-speech)."""
    turns = f"{usage.turns} turn{'' if usage.turns == 1 else 's'}"
    lines = [f"Session cost (estimated, USD) over {turns}:"]
    details = {
        "llm": (
            f"{usage.llm_calls} calls, {usage.prompt_tokens:,} in "
            f"({usage.cached_prompt_tokens:,} cached) / {usage.completion_tokens:,} out tokens"
        )
    }
    for source, cost in usage.costs.items():
        lines.append(f"  {source:<10} {_usd(cost):>12}  {details.get(source, '')}".rstrip())
    lines.append(f"  {'total':<10} {_usd(usage.total_cost_usd):>12}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    overrides: dict[str, object] = {}
    if args.verbose:
        overrides["log_to_stderr"] = True
    if args.force_failure:
        overrides["mcp_force_failure"] = args.force_failure
    if args.voice:
        overrides["reply_style"] = "voice"

    base = Settings()
    # Configure logging before anything (e.g. the mock server) starts emitting.
    configure_logging(
        level=base.log_level, log_dir=base.log_dir, to_stderr=args.verbose or base.log_to_stderr
    )

    agent: FoodOrderAgent | None = None
    try:
        with contextlib.ExitStack() as stack:
            if args.local:
                server = stack.enter_context(BackgroundMockServer(busy_rate=args.busy_rate))
                overrides["mcp_url"] = server.url
                overrides.setdefault("applicant_email", "local@example.com")
            settings = base.model_copy(update=overrides)
            agent = FoodOrderAgent(settings=settings)
            if args.voice:
                channel = _open_voice_channel(settings, agent.menu, stack)
            else:
                channel = stack.enter_context(TextChannel())
            _repl(agent, channel)
    except KeyboardInterrupt:
        print()  # Ctrl+C outside the prompt (e.g. mid-reply): still show the summary
    finally:
        # After the stack has closed, so the voice display is gone and this stays on screen.
        if agent is not None:
            agent.close_turn()
            if agent.usage.turns:
                print(_dim(session_cost_summary(agent.usage)))


if __name__ == "__main__":
    main()
