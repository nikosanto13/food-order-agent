# Food Order Agent

A conversational agent that takes food orders in natural language, keeps a validated cart,
confirms with the customer, and submits the order through the MCP `submit_order` tool.

```
You:   I'd like a large classic burger with cheese and bacon
Agent: Large Classic Burger with cheese and bacon. Anything else?
You:   Actually, remove the bacon
Agent: Updated: Large Classic Burger with cheese. Anything else?
You:   That's it
Agent: Your order: Large Classic Burger with cheese — $11.50. Ready to submit?
You:   Yes
Agent: Order submitted! Order #ORD-48271. Estimated time: 15-20 minutes.
```

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for you).

```bash
uv sync                       # create .venv with locked dependencies
cp .env.example .env          # then set MISTRAL_API_KEY (and APPLICANT_EMAIL for prod)

python main.py --local        # chat, submitting to a local mock MCP server (no setup)
python main.py                # chat, submitting to MCP_URL from .env
```

(`uv run python main.py ...` works the same if the venv isn't activated.)

Useful flags: `--verbose` (stream JSON logs to stderr), `--force-failure kitchen_busy|server_error`
(failure drills via `X-Demo-Force-Failure`), `--busy-rate 0.3` (mock's random busy rate).
REPL commands: `/cart`, `/reset`, `/trace` (path of this conversation's trace), `/quit`.

### Programmatic API

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()
agent.send("I want a large burger with cheese")
# {'message': 'Large Classic Burger with cheese. Anything else?'}
agent.send("Submit")
# {'message': 'Your order: Large Classic Burger with cheese — $11.50. Ready to submit?'}
agent.send("Yes")
# {'message': 'Order submitted! Order #ORD-10000. ...',
#  'tool_calls': [{'name': 'submit_order', 'arguments': {...}, 'result': {'success': True, ...}}]}
```

Every collaborator is injectable: `FoodOrderAgent(llm=..., submitter=..., menu=..., settings=...)`.

### Local MCP server

```bash
make mock-server              # python -m food_agent.mock_server --port 8765 --busy-rate 0.1
```

It mirrors the documented contract of the provided server: same tool and schemas, requires
`X-Applicant-Email`, refuses off-menu orders and totals over $50, honours `X-Demo-Force-Failure`,
and has a seedable random "kitchen busy" rate.

### Configuration (`.env` / environment)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | — | LLM credentials (required for real conversations) |
| `MISTRAL_MODEL` | `mistral-small-latest` | Model id |
| `MCP_URL` | `http://127.0.0.1:8765/mcp` | Order server (Streamable HTTP) |
| `APPLICANT_EMAIL` | — | Sent as `X-Applicant-Email` |
| `MCP_MAX_ATTEMPTS` | `3` | Attempts for retryable submit failures |
| `MCP_FORCE_FAILURE` | — | `kitchen_busy` / `server_error` drill header |
| `MAX_TOOL_ITERATIONS` | `8` | LLM↔tool steps per user turn |
| `MAX_HISTORY_MESSAGES` | `40` | Chat history sent to the LLM |
| `LOG_LEVEL`, `LOG_DIR`, `LOG_TO_STDERR` | `INFO`, `logs`, `false` | Observability |
| `LLM_INPUT_COST_PER_MTOK`, `LLM_OUTPUT_COST_PER_MTOK` | `0.10`, `0.30` | USD per 1M tokens, used for cost **estimates** in traces |

---

## Architecture

```
            ┌──────────────────────────── FoodOrderAgent.send() ─────────────────────────────┐
user text ─►│ pre-checks ─► LLM ◄──► ToolExecutor ──► OrderSession (cart + guard) ──► Menu    │
            │  (empty/long)  │        (pydantic-       │  validate, price, review,   (YAML,   │
            │                │         validated args)  │  authorize submission       Decimal) │
            │                ▼                          ▼                                      │
            │        reply guards              McpOrderSubmitter ──► MCP server             │
            │  (total / order id verbatim)     (classify, retry)     (prod or local mock)    │
            └─────────────── structured events + one trace record per turn ─────────────────┘
```

| Module | Responsibility |
| --- | --- |
| `food_agent/menu.py` | Parse `menu.yaml` into frozen Pydantic models; integrity checks at load time; compact prompt rendering |
| `food_agent/cart.py` | Line validation, pricing (`Decimal`), cart, and the `OrderSession` confirmation state machine. No LLM. |
| `food_agent/tools.py` | The LLM's tool surface: schemas from Pydantic arg models, execution, errors returned as data |
| `food_agent/agent.py` | The turn loop, history management, reply guards, tracing |
| `food_agent/llm.py` | `LLMClient` protocol + Mistral adapter (SDK-level backoff on 429/5xx) |
| `food_agent/mcp_client.py` | MCP client: outcome classification, retry policy, idempotency |
| `food_agent/mock_server.py` | Local MCP server with the same contract (dev + tests) |
| `food_agent/observability.py` | structlog JSON config, redaction, per-turn trace records |
| `agent.py`, `main.py` | Required entry points |

---

## Design decisions

### 1. The LLM interprets; deterministic code decides
The model's job is language: understand "make it large", pick the right line, ask when something is
missing. It never owns state. Every change goes through a typed tool, and `OrderSession` checks it
against the menu before accepting it. The cart, prices, totals and the $50 limit are plain Python
that is unit-tested to the cent. A wrong guess by the model comes back as a tool error with the
valid choices, for example `Invalid size 'xl' for French Fries; choices: small, medium, large.`,
and the model either corrects itself or asks the customer. It can never produce an invalid order.

### 2. Least-privilege submission
- `submit_order` as the LLM sees it takes **no arguments**. Code builds the payload from the
  validated cart, so the model can't add items, change quantities or invent prices in what gets
  sent, whatever a prompt injection says.
- **Confirmation is enforced in code, not left to the prompt.** Submission is only allowed if
  `review_order` ran in an *earlier* turn and the cart hasn't changed since. Any change resets the
  review. Together these guarantee the customer saw exactly this cart and total and replied to it
  before anything left the process. The model still decides whether the reply means "yes"; it just
  can't skip showing the summary first.
- The model never sees the MCP client, URL or headers. Its whole capability is the eight cart tools.

### 3. Server-side validation before submission
`validate_for_submission()` re-validates every line from scratch and checks the $50 limit when the
order is reviewed *and again* right before it is sent. The customer hears about an over-limit
order, with the amount it is over by, during review, not as a remote refusal after they have
already said yes.

### 4. Failure handling: classify the outcome, then decide
Every MCP response maps to exactly one outcome. The retry policy is keyed on the outcome, never on
error text, which the brief says is not part of the contract.

| Outcome | Detected by | Order placed? | Policy |
| --- | --- | --- | --- |
| `success` | `success: true` | yes | report order id + ETA, start a fresh cart |
| `refused` | `success: false`, not `isError` | no | retry with backoff, then the LLM relays the server's message; cart kept |
| `server_error` | `isError` **and** a `success:false` body | no | retry with backoff, then apologise; cart kept |
| `invalid_request` | `isError` without a body (schema / `-32602`) | no | **no retry** (our bug); log it, apologise |
| `unreachable` | failure *before* the call was sent | no | retry, then offer to try later |
| `uncertain` | timeout/disconnect *after* sending | **unknown** | **never auto-retry** (could double-order); tell the customer honestly |

`isError` is checked before `structuredContent` is read. Why can refusals be retried? The cart was
already validated against the menu and the limit, so a `success:false` on a validated cart is almost
certainly the transient busy kitchen. Retrying a real business refusal unchanged is harmless, since
it just gets refused again. If the server ever adds a machine-readable `retryable` flag, it wins.
Retries are capped at 3 attempts with exponential backoff and full jitter. If the kitchen is still
busy after that, the agent asks the customer whether to try again, as in the brief's example. The
reviewed cart is unchanged, so "yes, try again" goes straight through without another review.

LLM failures (after the SDK's own retries on 429/5xx) produce a polite "please try again" reply.
The cart is kept, and the next turn works normally. The tool loop is capped at
`MAX_TOOL_ITERATIONS`; if it runs out, the agent shows the current cart instead of spinning.

### 5. Conversation quality
- **State is injected into the prompt.** The live cart (line ids, descriptions, totals, phase) goes
  into the system prompt on every LLM call. The model never has to remember line ids from earlier
  turns, which is what makes "remove the bacon" reliable.
- **Reply guards.** If the model reviews the order but leaves the exact total out of its reply, or
  confirms a submission without the order id, code appends the canonical summary or id. What the
  customer pays is never left to the model. Changes to **multi-unit lines** get the same
  treatment: if the reply doesn't contain the line's current description (e.g.
  `2x Large Classic Burger`), code appends it. A paraphrase like "updated one of the burgers"
  after all three were changed is exposed in that same reply, not discovered at checkout.
  Single-unit changes are left to the model, where paraphrasing is harmless.
- **The prompt sets conversational policy:** sensible defaults ("burger" means the classic burger,
  "coke" means a cola soda); use defaults silently; ask only when an option has no default (the
  milkshake flavor); confirm each change in one line; review before submitting.
- Tool errors are written to be acted on, and are returned as data so the model can recover.
- **Tools match how people talk: lines split when only some units change.** A cart line is "N
  identical units", but customers talk about subsets: "remove one of them", "add all extras to
  one of the three". `remove_item(quantity=…)` takes units off a line, and
  `update_item(units=…)` applies a change to only that many units by splitting the line
  (`3x Large Classic Burger` becomes `2x Large Classic Burger` plus
  `Large Classic Burger with …`). The arithmetic is in code and a failed split changes nothing.
  Without this, a small model had to chain two calls to express "one of them", and it usually
  changed or deleted the whole line instead.
- **Changes made during checkout are re-reviewed automatically.** Once the customer has asked to
  check out, every later change (add, update, remove, special instructions) re-runs the review and
  attaches it to the tool result, with either a fresh summary and total, or the reason the order
  still can't go through (for example, still over the limit). "Ready to submit?" is therefore
  always backed by a current review, and the customer's next "yes" submits directly. This is the
  flow in brief example 2 ("Make the cola large" → new total → "Ready to submit?").

### 6. Treating user text as untrusted
- User text only ever goes in `user` messages. It is never interpolated into instructions.
- The prompt says customer text is data, not instructions. More importantly, the architecture means
  a successful injection can only call cart tools with arguments that still get validated.
- Free text that leaves the system (`special_instructions`) is cleaned up: control, zero-width and
  bidi characters are removed and the length is capped at 200. Overlong text is rejected rather
  than silently cut, so nothing the customer said is lost without them knowing.
- Input size is capped before any LLM call, which also guards against cost blow-ups. YAML is
  loaded with `safe_load`. The API key is a `SecretStr`, and a log processor redacts
  sensitive keys.

### 7. Observability
- **Event log**, `logs/agent.jsonl`: one JSON line per event (`turn.start`, `llm.call` with
  latency and tokens, `tool.call`, `mcp.submit_attempt` with outcome and latency, `order.submit`,
  `reply.guard`, `turn.end` with token and cost totals). Every line carries `conversation_id` and
  `turn` through contextvars.
- **Conversation trace**, `logs/conversations/<conversation_id>.jsonl`: **one record per turn**,
  holding the user and assistant text, every step in order (LLM calls with model, latency, tokens
  and finish reason; tool calls with arguments and results), the cart after the turn, the phase,
  usage, estimated cost and errors. A turn maps directly onto an OpenTelemetry trace (turn = root
  span, steps = child spans), so exporting to Langfuse, Phoenix or OTel is a sink change, not a
  redesign.

```bash
jq -c 'select(.event=="llm.call") | {turn, latency_ms, prompt_tokens}' logs/agent.jsonl
jq -c '{turn, user_message, assistant_message, usage}' logs/conversations/conv_*.jsonl
```

### 8. Performance & cost
- **Small model by default** (`mistral-small-latest`). The task is narrow and the hard guarantees
  live in code, so a frontier model isn't needed. The model is one env var away.
- **Static prompt prefix** (instructions + menu, rendered once and compactly), with the dynamic
  state at the end so provider-side prefix caching can reuse it.
- **Bounded context.** History is trimmed to the last N messages, always cut at a user-message
  boundary so a tool result is never separated from its tool call. Because the cart is in the
  prompt, old turns aren't needed.
- **No LLM call** for empty or oversized input. `max_tokens` is capped. Tokens and estimated cost
  are logged for every call and every turn.
- **Connect per submission.** Orders are submitted about once per conversation, so the MCP session
  is opened only when needed. The cost is one handshake per order. In return `send()` stays
  synchronous, there is no background event loop to manage, and per-request headers are trivial.

### 9. What is enforced in code vs. interpreted by the model
The agent is deliberately **strict about the order and relaxed about the conversation**.

| Enforced in code (cannot be talked around) | Interpreted by the model (best effort) |
| --- | --- |
| Only menu items, options and extras; defaults resolved explicitly | Mapping phrases to items ("coke", "the big one", "same again") |
| Prices, totals, the $50 limit (at review **and** before submit) | Which line a change refers to; asking when it is ambiguous |
| Submit only after a review in an earlier turn, on an unchanged cart | Whether the customer's reply means "yes" |
| Submit payload built from the cart; the model passes no arguments | Tone, brevity, and staying on topic |
| Quantity arithmetic for partial removals and line splits | Choosing "some units" vs. "the whole line" from the wording |
| Exact reviewed total, order id, and multi-unit changes present in the reply | |

**Known limitations, accepted on purpose:**
- The prompt-level rules (stay on topic, ignore "new instructions") can be bypassed by a
  determined user. The consequence is limited to an odd reply or a wrong change *to their own
  cart*, which they see in the next confirmation. Nothing unvalidated or unreviewed can be
  submitted. More guardrails would mostly make normal conversations stiffer.
- "Yes" detection is the model's call. A model that reads an ambiguous reply as confirmation can
  submit, but only the exact cart and total the customer was just shown.
- Misreading intent (e.g. "remove the large" when there are three) is mitigated by the tool
  design, the prompt rule and the multi-unit reply guard, not eliminated. The guard means a wrong
  change is always *visible* in the reply, so the customer can correct it. The evals on the roadmap
  are how these regressions get caught against the real model.

### 10. Smaller choices
- `Decimal` for money end to end. Defaults are resolved when an item is added, so the cart and the
  payload are explicit (`{"size": "large", "patty": "beef"}`).
- Line ids are stable and never reused, so "line 2" means the same thing all conversation.
- `send()` returns `tool_calls` only for **external** (MCP) calls, matching the brief's examples.
  Internal cart tool calls go in the trace.
- The mock server reuses the agent's menu module. That's fine for a mock: the provided server is
  the real oracle, and contract tests against it come next.

---

## Quality gates

```bash
make check      # ruff format --check · ruff check · ty check · pytest
make fmt        # auto-format + autofix
```

- **ruff**, with a broad rule set (pyflakes, bugbear, bandit `S`, pylint `PL`, `TRY`, `PERF`,
  `ANN`, `T20`, ...); see `pyproject.toml`. **ty** type-checks the package, the tests and the
  entry points.
- **pytest**, 83 tests, offline and deterministic (about 1 s):
  - `test_menu.py`: parsing and integrity errors.
  - `test_cart.py`: the brief's totals ($11.50, $22.25), validation messages, normalisation
    (`"Extra Hot"` → `extra_hot`, Unicode `jalapeños`), sanitisation, partial removal, line splitting, and the confirmation guard.
  - `test_mcp.py`: every outcome over real Streamable HTTP against the mock: forced failures,
    schema errors, missing email, unreachable server, recovery on retry.
  - `test_agent.py`: the full loop with a scripted LLM. Covers brief example 1 end to end,
    submit-in-same-turn and submit-without-review blocked, an LLM-supplied payload rejected, busy
    kitchen then retry, self-correction after tool errors, malformed tool calls, LLM outage,
    bounded loop, reply guards, state injection, history trimming, trace contents, and a replay of a
    real over-limit conversation ("remove the one large" → 2 left; re-review → "yes" submits),
    "add all extras to one of the three" splitting the line, and the multi-unit reply guard.
- **CI**: GitHub Actions runs the same gates on every push and PR.

The scripted LLM tests the **plumbing and the guarantees**. Whether the real model makes good
decisions is measured by evals (next section).

---

## Roadmap / with more time

- **Evals** (next): scripted multi-turn scenarios run against the real LLM and the mock server. The
  assertions are deterministic: the final submitted payload, no submission without confirmation,
  the right clarifying questions. The report covers pass rate, tokens, cost and p50/p95 latency per
  model, so model choice is a measured decision. Possibly an LLM-as-judge for tone.
- **Prod MCP contract tests**: a `@pytest.mark.live` suite against the provided server, checking
  the classification table above (including whether refusals expose a machine-readable code).
- **Voice**: STT → `agent.send()` → TTS (ElevenLabs), with shorter, speech-friendly replies.
- Session persistence (Redis) and an async `asend()` for serving many concurrent conversations.
  Streaming replies.
- OTel / Langfuse export of the traces, plus dashboards for latency, cost per order and guard
  activations.
- Fuzzy matching for misspelled items before the LLM fallback; menu hot-reload; multi-language
  support.
