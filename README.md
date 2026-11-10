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

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for you). Run commands from the repository root.

```bash
uv sync                          # create .venv with locked dependencies
cp .env.example .env             # then set MISTRAL_API_KEY
uv run python main.py --local    # chat, submitting to an in-process mock MCP server
```

The mock server mirrors the production contract and returns "kitchen busy" on 10% of submissions,
to exercise the retry path. Pass `--busy-rate 0` for a clean run.

To submit to the production server instead, set these in `.env` and run `uv run python main.py`:

```bash
MCP_URL=https://webviews.moveo.ai/api/demo/mcp
APPLICANT_EMAIL=you@example.com
```

REPL commands: `/cart`, `/reset`, `/trace`, `/help`, `/quit`. Useful flags: `--verbose` (JSON logs
to stderr) and `--force-failure kitchen_busy|server_error` (fail every submit, for drills).

### Programmatic API

`FoodOrderAgent` submits to `MCP_URL`, so it needs a server: either the production URL above, or the
mock started with `make mock-server` (listens on `127.0.0.1:8765`, the default `MCP_URL`).

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

<details>
<summary>Voice mode</summary>

Talk instead of typing - the agent speaks its replies and prints the transcript.

```bash
# elevenlabs + sounddevice + rich
uv sync --extra voice
```

Then set `ELEVENLABS_API_KEY` in `.env`. Voice also needs PortAudio (macOS: `brew install portaudio`,
Debian/Ubuntu: `sudo apt install libportaudio2`) and microphone access for your terminal (macOS asks
on first run).

```bash
uv run python main.py --voice --local
```

```text
You: a large margherita with olives
Agent: Large Margherita Pizza with olives. Anything else?
● Speaking        you
Ctrl+C to quit    agent   ▃▅▆▆▅▃▁▂▄▆▆▆▄▁▂▄▆▆▆▄▂▁▃▅▆▆▅▂ ▃▅
```

> **⚠️ Warning:**
> Voice mode is early-stage. See the [design notes](docs/ARCHITECTURE.md#11-voice-an-optional-front-end)
> and [future steps](#future-steps).

</details>

<details>
<summary>Configuration</summary>

Set any of these in `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | — | LLM credentials (required for real conversations) |
| `MISTRAL_MODEL` | `mistral-small-latest` | Model id |
| `MCP_URL` | `http://127.0.0.1:8765/mcp` | Order server (Streamable HTTP) |
| `APPLICANT_EMAIL` | — | Sent as `X-Applicant-Email` |
| `MCP_MAX_ATTEMPTS` | `3` | Attempts for retryable submit failures |
| `MCP_FORCE_FAILURE` | — | `kitchen_busy` / `server_error` drill header (same as `--force-failure`) |
| `MAX_TOOL_ITERATIONS` | `8` | LLM↔tool steps per user turn |
| `MAX_HISTORY_MESSAGES` | `40` | Chat history sent to the LLM |
| `ELEVENLABS_API_KEY` | — | Voice mode credentials |
| `ELEVENLABS_VOICE_ID` | `JBFqnCBsd6RMkjVDRZzb` | TTS voice |
| `ELEVENLABS_TTS_MODEL`, `ELEVENLABS_STT_MODEL` | `eleven_flash_v2_5`, `scribe_v2` | Speech models |
| `VOICE_STT_KEYTERMS` | `true` | Bias transcription towards menu words (billed extra) |
| `VOICE_END_SILENCE_MS` | `900` | Pause that ends your turn |
| `VOICE_VAD_MARGIN_DB` | `12` | How much louder than the room counts as speech (raise in noisy rooms) |
| `VOICE_MAX_UTTERANCE_S` | `30` | Longest turn kept |
| `LOG_LEVEL`, `LOG_DIR`, `LOG_TO_STDERR` | `INFO`, `logs`, `false` | Observability (`LOG_TO_STDERR` = `--verbose`) |

</details>

---

## High-level Design

The LLM interprets the conversation, while deterministic code owns the order. The model edits the cart
only through typed tools that are validated against the menu, prices and the $50 limit are plain
Python, and `submit_order` takes no arguments from the model, so it can only send the cart the
customer has already reviewed and confirmed. MCP failures are classified (refused, server error,
unreachable, uncertain, ...) and retried only when that is safe.

For more details, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: module layout, the
failure-handling policy, prompt-injection defences, observability, cost, and what is enforced in
code vs. left to the model.

---

## Quality gates

The project uses [ruff](https://docs.astral.sh/ruff/) (formatting and linting), [ty](https://docs.astral.sh/ty/) (typechecking),
[pytest](https://docs.pytest.org/en/stable/) (unit tests) and [uv](https://docs.astral.sh/uv/) (vulnerability scanning).

Run the checks with:

```bash
make check      # format check, lint, typecheck, tests
make fmt        # auto-format + autofix
make audit      # scan locked dependencies for known vulnerabilities (needs network)
make test-live  # tests marked `live`: real LLM / MCP / ElevenLabs (spends credits)
```

---

## Future steps

Below is a non-exhaustive list of potential improvements to the agent:

- **Voice, full duplex**: let the customer interrupt the agent (barge-in, which needs echo
  cancellation instead of muting the mic), stream the LLM reply into TTS sentence by sentence,
  and swap the energy detector for a model-based VAD (e.g. Silero) or realtime STT with
  server-side turn detection if noisy rooms turn out to matter.
- **Voice**: Expose the agent over a network transport (WebSocket/WebRTC) so voice isn’t tied
  to the local REPL’s Channel
- **Evals**: scripted multi-turn scenarios run against the real LLM and the mock server. The
  assertions are deterministic: the final submitted payload, no submission without confirmation,
  the right clarifying questions. The report covers pass rate, tokens, cost and p50/p95 latency per
  model, so model choice is a measured decision. Possibly an LLM-as-judge for tone.
- Extension to provide additional model providers.
- Interactive mode for text agent / streaming replies.
- OTel / Langfuse export of the traces, plus dashboards for latency, cost per order and guard activations.
- Fuzzy matching for misspelled items before the LLM fallback.
