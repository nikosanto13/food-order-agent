"""Local stand-in for the Moveo ``submit_order`` MCP server.

Mirrors the documented contract so the agent can be developed and tested
offline, deterministically, and without spending the shared demo server's
capacity:

* same tool name, input schema (``quantity >= 1``) and declared output schema;
* requires ``X-Applicant-Email`` (like prod);
* business refusals (``success: false``) for off-menu orders and totals > $50;
* ``X-Demo-Force-Failure: kitchen_busy | server_error`` for failure drills;
* a configurable random "kitchen busy" rate (seedable) for realism.

Note: it validates with the agent's own menu module. That is fine for a mock
(the prod server is the real oracle for contract tests), and the duplication
would only add a second place to keep in sync.

Run: ``python -m food_agent.mock_server [--port 8765] [--busy-rate 0.1] [--seed 1]``
"""

from __future__ import annotations

import argparse
import itertools
import random
from typing import Annotated, Any

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from food_agent.cart import ORDER_TOTAL_LIMIT, Cart, OrderError
from food_agent.menu import Menu, load_menu

FORCE_FAILURE_HEADER = "x-demo-force-failure"
APPLICANT_HEADER = "x-applicant-email"


class OrderItem(BaseModel):
    item_id: str = Field(description="Menu item ID (e.g., classic_burger, margherita, fries)")
    quantity: int = Field(ge=1, description="Number of this item (1 or more)")
    options: dict[str, str] | None = Field(
        default=None, description="Item options (e.g., size, patty, crust, flavor)"
    )
    extras: list[str] | None = Field(default=None, description="List of extra IDs for this item")


class SubmitOrderResult(BaseModel):
    success: bool
    order_id: str | None = None
    total: float | None = None
    estimated_time: str | None = None
    error: str | None = None


def _result(body: SubmitOrderResult, *, is_error: bool = False) -> CallToolResult:
    payload = body.model_dump(exclude_none=True)
    return CallToolResult(
        content=[TextContent(type="text", text=body.model_dump_json(exclude_none=True))],
        structured_content=payload,
        is_error=is_error,
    )


def create_server(
    menu: Menu | None = None, *, busy_rate: float = 0.0, seed: int | None = None
) -> MCPServer:
    menu = menu or load_menu()
    rng = random.Random(seed)  # noqa: S311 - simulation, not crypto
    order_ids = itertools.count(10_000)
    server = MCPServer(name="mock-food-orders", version="0.1.0")

    @server.tool(name="submit_order", description="Submit the final food order")
    def submit_order(
        items: Annotated[list[OrderItem], Field(description="List of order items")],
        ctx: Context,
        special_instructions: Annotated[
            str, Field(description="Any special instructions for the order")
        ] = "",
    ) -> Annotated[CallToolResult, SubmitOrderResult]:
        headers: dict[str, Any] = {k.lower(): v for k, v in (ctx.headers or {}).items()}
        if ctx.headers is not None and not headers.get(APPLICANT_HEADER):
            raise ToolError("Missing required header X-Applicant-Email.")

        forced = headers.get(FORCE_FAILURE_HEADER)
        if forced == "server_error":
            return _result(
                SubmitOrderResult(success=False, error="Internal error while placing order."),
                is_error=True,
            )
        if forced == "kitchen_busy" or rng.random() < busy_rate:
            return _result(
                SubmitOrderResult(success=False, error="Kitchen is currently busy. Please retry.")
            )

        cart = Cart(menu)
        try:
            for it in items:
                cart.add(it.item_id, it.quantity, it.options, it.extras)
        except OrderError as exc:
            return _result(SubmitOrderResult(success=False, error=str(exc)))

        total = cart.total
        if total > ORDER_TOTAL_LIMIT:
            return _result(
                SubmitOrderResult(
                    success=False,
                    error=f"Order total (${total:.2f}) exceeds ${ORDER_TOTAL_LIMIT:.0f} limit.",
                )
            )
        _ = special_instructions  # accepted and ignored, like a real kitchen ticket
        return _result(
            SubmitOrderResult(
                success=True,
                order_id=f"ORD-{next(order_ids)}",
                total=float(total),
                estimated_time="15-20 minutes",
            )
        )

    return server


def create_app(*, busy_rate: float = 0.0, seed: int | None = None) -> Starlette:
    return create_server(busy_rate=busy_rate, seed=seed).streamable_http_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--busy-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    uvicorn.run(
        create_app(busy_rate=args.busy_rate, seed=args.seed), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
