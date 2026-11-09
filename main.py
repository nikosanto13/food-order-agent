"""Interactive entry point: ``python main.py [--local] [--voice] [--verbose] ...``."""

from __future__ import annotations

import argparse
import contextlib
import json
from typing import Protocol

from food_agent.agent import FoodOrderAgent
from food_agent.config import Settings
from food_agent.local_server import BackgroundMockServer
from food_agent.menu import Menu
from food_agent.observability import configure_logging

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
        help="talk instead of type (needs `uv sync --extra voice` and ELEVENLABS_API_KEY)",
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
    """Where user messages come from and replies go: the keyboard, or voice."""

    def read(self) -> str: ...
    def reply(self, message: str) -> None: ...


class TextChannel:
    def read(self) -> str:
        return input("You: ").strip()

    def reply(self, message: str) -> None:
        print(f"Agent: {message}")


def _voice_channel(settings: Settings, menu: Menu) -> Channel:
    # Imported here so text mode never depends on the optional voice extra.
    from food_agent.voice import VoiceUnavailableError, build_voice_channel  # noqa: PLC0415

    try:
        return build_voice_channel(settings, menu)
    except VoiceUnavailableError as exc:
        raise SystemExit(f"Voice mode unavailable: {exc}") from exc


def _repl(agent: FoodOrderAgent, channel: Channel) -> None:
    print(f"Food ordering assistant ({agent.llm.model}). {HELP}")
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

        response = agent.send(user)
        channel.reply(response["message"])
        for call in response.get("tool_calls", []):
            result = json.dumps(call["result"], ensure_ascii=False)
            print(f"  [{call['name']} -> {result}]")
        print()


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

    with contextlib.ExitStack() as stack:
        if args.local:
            server = stack.enter_context(BackgroundMockServer(busy_rate=args.busy_rate))
            overrides["mcp_url"] = server.url
            overrides.setdefault("applicant_email", "local@example.com")
        settings = base.model_copy(update=overrides)
        agent = FoodOrderAgent(settings=settings)
        channel = _voice_channel(settings, agent.menu) if args.voice else TextChannel()
        _repl(agent, channel)


if __name__ == "__main__":
    main()
