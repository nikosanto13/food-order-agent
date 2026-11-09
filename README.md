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

### Voice mode (optional)

Hands-free: just talk. ElevenLabs Scribe transcribes you, the same agent handles the order,
and ElevenLabs Flash speaks the reply, with live level bars for you and the agent.

```bash
uv sync --extra voice         # optional extra: elevenlabs + sounddevice + rich
# set ELEVENLABS_API_KEY in .env
python main.py --voice --local
```

```
You: a large margherita with olives
Agent: Large Margherita Pizza with olives. Anything else?
● Speaking        you
Ctrl+C to quit    agent   ▃▅▆▆▅▃▁▂▄▆▆▆▄▁▂▄▆▆▆▄▂▁▃▅▆▆▅▂ ▃▅
```

Pause for about a second to end your turn. The microphone is off while the agent speaks, so
it never hears itself, but headphones still work best. Ctrl+C while the agent speaks stops the
audio; Ctrl+C while listening quits. Without the extra, everything else installs and runs as
before; `--voice` just tells you what to install. See
[design notes](docs/ARCHITECTURE.md#11-voice-an-optional-front-end).

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
| `ELEVENLABS_API_KEY` | — | Voice mode credentials |
| `ELEVENLABS_VOICE_ID` | `JBFqnCBsd6RMkjVDRZzb` | TTS voice |
| `ELEVENLABS_TTS_MODEL`, `ELEVENLABS_STT_MODEL` | `eleven_flash_v2_5`, `scribe_v2` | Speech models |
| `VOICE_STT_KEYTERMS` | `true` | Bias transcription towards menu words (billed extra) |
| `VOICE_END_SILENCE_MS` | `900` | Pause that ends your turn |
| `VOICE_VAD_MARGIN_DB` | `12` | How much louder than the room counts as speech (raise in noisy rooms) |
| `VOICE_MAX_UTTERANCE_S` | `30` | Longest turn kept |
| `LOG_LEVEL`, `LOG_DIR`, `LOG_TO_STDERR` | `INFO`, `logs`, `false` | Observability |

---

## How it works

The LLM interprets the conversation; deterministic code owns the order. The model edits the cart
only through typed tools that are validated against the menu, prices and the $50 limit are plain
Python, and `submit_order` takes no arguments from the model, so it can only send the cart the
customer has already reviewed and confirmed. MCP failures are classified (refused, server error,
unreachable, uncertain, ...) and retried only when that is safe.

For the details, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: module layout, the
failure-handling policy, prompt-injection defences, observability, cost, and what is enforced in
code vs. left to the model.

---

## Quality gates

```bash
make check      # format, linting, typechecking, pytest
make fmt        # auto-format + autofix
make audit      # scan locked dependencies for known vulnerabilities (needs network)
make test-live  # tests marked `live`: real LLM / MCP / ElevenLabs (spends credits)
```

Linting is **ruff** with a broad rule set (see `pyproject.toml`), type-checking is **ty**, and the
**pytest** suite is offline and deterministic (about 1 s). Dependencies are audited with
**uv audit** against OSV; accepted findings go in `[tool.uv.audit]`. GitHub Actions runs the same
gates on every push and PR (once with all extras, once without, to keep voice optional), plus a
weekly audit.

---

## Future steps

- **Voice, full duplex**: let the customer interrupt the agent (barge-in, which needs echo
  cancellation instead of muting the mic), stream the LLM reply into TTS sentence by sentence,
  and swap the energy detector for a model-based VAD (e.g. Silero) or realtime STT with
  server-side turn detection if noisy rooms turn out to matter.
- **Evals**: scripted multi-turn scenarios run against the real LLM and the mock server. The
  assertions are deterministic: the final submitted payload, no submission without confirmation,
  the right clarifying questions. The report covers pass rate, tokens, cost and p50/p95 latency per
  model, so model choice is a measured decision. Possibly an LLM-as-judge for tone.
- **Prod MCP contract tests**: a `@pytest.mark.live` suite against the provided server, checking
  the [outcome classification table](docs/ARCHITECTURE.md#4-failure-handling-classify-the-outcome-then-decide) (including whether refusals expose a machine-readable code).
- Extension to provide additional model providers.
- Interactive mode for text agent / streaming replies.
- OTel / Langfuse export of the traces, plus dashboards for latency, cost per order and guard activations.
- Fuzzy matching for misspelled items before the LLM fallback.
