"""Terminal UI for hands-free mode: a live status line and two level bars.

    ● Listening      you    ▁▁▂▅▇▆▅▃▂▁▁▁▁▁▁▁
                     agent

Rendered with ``rich`` (part of the ``voice`` extra) in a live region at the
bottom of the terminal; transcript lines and ordinary ``print()`` output scroll
above it.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from food_agent.voice.errors import INSTALL_HINT, VoiceUnavailableError
from food_agent.voice.meter import Lane, LevelMeter, bars

Status = Literal["listening", "hearing", "transcribing", "thinking", "speaking"]
Role = Literal["you", "agent", "note"]

_STATUS: dict[Status, tuple[str, str]] = {  # label, style
    "listening": ("Listening", "green"),
    "hearing": ("Hearing you", "bold cyan"),
    "transcribing": ("Transcribing", "yellow"),
    "thinking": ("Thinking", "yellow"),
    "speaking": ("Speaking", "bold magenta"),
}
_BUSY: set[Status] = {"transcribing", "thinking"}
_ROLE_STYLE: dict[Role, tuple[str, str]] = {
    "you": ("You: ", "bold cyan"),
    "agent": ("Agent: ", "bold magenta"),
    "note": ("  ", "dim"),
}


class RichDisplay:
    def __init__(self, meter: LevelMeter | None = None, console: Any | None = None) -> None:
        try:
            from rich.console import Console  # noqa: PLC0415 - optional extra
            from rich.live import Live  # noqa: PLC0415
        except ImportError as exc:
            raise VoiceUnavailableError(INSTALL_HINT) from exc
        self.meter = meter or LevelMeter()
        self.status: Status = "listening"
        self._console = console or Console()
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=20,
            transient=True,
            redirect_stderr=False,
        )

    def __enter__(self) -> Self:
        self._live.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._live.stop()

    # ------------------------------------------------------------------ Display protocol

    def set_status(self, status: Status) -> None:
        self.status = status

    def level(self, lane: Lane, level: float) -> None:
        self.meter.push(lane, level)

    def say(self, role: Role, text: str) -> None:
        from rich.text import Text  # noqa: PLC0415

        prefix, style = _ROLE_STYLE[role]
        line = Text(prefix, style=style)
        line.append(text, style="dim" if role == "note" else "")
        self._console.print(line)

    # ------------------------------------------------------------------ rendering

    def __rich__(self) -> Any:
        from rich.spinner import Spinner  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        label, style = _STATUS[self.status]
        status: Any = (
            Spinner("dots", text=Text(label, style=style), style=style)
            if self.status in _BUSY
            else Text(f"● {label}", style=style)
        )
        levels = self.meter.snapshot()
        you_live = self.status in {"listening", "hearing"}
        agent_live = self.status == "speaking"
        grid = Table.grid(padding=(0, 2))
        grid.add_column(width=16)
        grid.add_column(width=5)
        grid.add_column()
        grid.add_row(
            status,
            Text("you", style="cyan"),
            Text(bars(levels["you"]), "cyan" if you_live else "dim"),
        )
        grid.add_row(
            Text("Ctrl+C to quit", style="dim"),
            Text("agent", style="magenta"),
            Text(bars(levels["agent"]), "magenta" if agent_live else "dim"),
        )
        return grid
