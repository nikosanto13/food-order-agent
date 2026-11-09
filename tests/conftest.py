import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

from food_agent.menu import Menu, load_menu
from food_agent.mock_server import create_app


@pytest.fixture(scope="session")
def menu() -> Menu:
    return load_menu()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server_url() -> Iterator[str]:
    """Real Streamable HTTP mock server on a random port (exercises headers + transport)."""
    port = _free_port()
    config = uvicorn.Config(create_app(busy_rate=0.0), port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("mock MCP server did not start")
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)
