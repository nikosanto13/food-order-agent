"""Scrolling level history for the "you" and "agent" bars (pure, thread-safe).

Audio threads ``push()`` levels whenever they have them (every 20 ms frame);
the renderer calls ``snapshot()`` at its own pace. Time advances in fixed ticks,
and a tick with no audio records silence, so the bars fall flat when nobody is
talking instead of freezing on the last frame.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Literal

Lane = Literal["you", "agent"]
LANES: tuple[Lane, ...] = ("you", "agent")
BAR_CHARS = " ▁▂▃▄▅▆▇█"


class LevelMeter:
    def __init__(
        self,
        width: int = 32,
        tick_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.width = width
        self._tick_s = tick_s
        self._clock = clock
        self._lock = threading.Lock()
        self._history = {lane: deque([0.0] * width, maxlen=width) for lane in LANES}
        self._pending = dict.fromkeys(LANES, 0.0)
        self._last_tick = clock()

    def push(self, lane: Lane, level: float) -> None:
        with self._lock:
            self._pending[lane] = max(self._pending[lane], min(1.0, max(0.0, level)))

    def snapshot(self) -> dict[Lane, list[float]]:
        with self._lock:
            ticks = int((self._clock() - self._last_tick) / self._tick_s + 1e-9)  # float slack
            if ticks:
                self._last_tick += ticks * self._tick_s
                for lane in LANES:
                    history = self._history[lane]
                    history.append(self._pending[lane])
                    history.extend([0.0] * min(ticks - 1, self.width))
                    self._pending[lane] = 0.0
            return {lane: list(self._history[lane]) for lane in LANES}


def bars(levels: Iterable[float]) -> str:
    top = len(BAR_CHARS) - 1
    return "".join(BAR_CHARS[round(level * top)] for level in levels)
