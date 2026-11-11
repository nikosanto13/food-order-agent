"""Run the mock MCP server in a background thread (for ``main.py --local``)."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn

from food_agent.mock_server import create_app


class BackgroundMockServer:
    def __init__(self, *, busy_rate: float = 0.0, seed: int | None = None) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port: int = s.getsockname()[1]
        config = uvicorn.Config(
            create_app(busy_rate=busy_rate, seed=seed), port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def __enter__(self) -> BackgroundMockServer:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("mock MCP server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *_: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
