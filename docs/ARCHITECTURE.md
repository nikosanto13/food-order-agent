# Architecture & Design Decisions

How the Food Order Agent is put together and why. For setup and usage, see the
[README](../README.md).

**Contents**

- [Architecture](#architecture)
- [Design decisions](#design-decisions)
  - [The LLM interprets; deterministic code decides](#1-the-llm-interprets-deterministic-code-decides)
  - [Least-privilege submission](#2-least-privilege-submission)
  - [Server-side validation before submission](#3-server-side-validation-before-submission)
  - [Failure handling: classify the outcome, then decide](#4-failure-handling-classify-the-outcome-then-decide)
  - [Conversation quality](#5-conversation-quality)
  - [Treating user text as untrusted](#6-treating-user-text-as-untrusted)
  - [Observability](#7-observability)
  - [Performance & cost](#8-performance--cost)
  - [What is enforced in code vs. interpreted by the model](#9-what-is-enforced-in-code-vs-interpreted-by-the-model)
  - [Smaller choices](#10-smaller-choices)
  - [Voice: an optional front-end](#11-voice-an-optional-front-end)

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
| `food_agent/pricing.py` | Per-model token price registry (input / cached input / output) for cost estimates |
| `food_agent/mcp_client.py` | MCP client: outcome classification, retry policy, idempotency |
| `food_agent/mock_server.py` | Local MCP server with the same contract (dev + tests) |
| `food_agent/observability.py` | structlog JSON config, redaction, per-turn trace records |
| `food_agent/voice/` | Optional voice front-end (`--extra voice`): hands-free channel, turn detection, live level bars, ElevenLabs STT/TTS, audio devices, speakable text |
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
  live in code, so a frontier model probably isn't necessary.
- **Static prompt prefix** (instructions + menu, rendered once and compactly), with the dynamic
  state at the end so provider-side prefix caching can reuse it (input vs. cache input tokens
  are shown in logs).
- **Bounded context.** History is trimmed to the last N messages, always cut at a user-message
  boundary so a tool result is never separated from its tool call. Because the cart is in the
  prompt, old turns aren't needed.
- **No LLM call** for empty or oversized input. `max_tokens` is capped. Tokens (including cached
  prompt tokens) and estimated cost are logged for every call and every turn.
- **Cache-aware cost estimates.** Prices live in a per-model registry in code
  (`food_agent/pricing.py`), not in configuration. Cached prompt tokens (reported by the provider
  as a subset of prompt tokens) are billed at the cached-input rate, the rest at the input rate.
  A model without a registered price logs `llm.pricing_missing` at startup and its cost is
  reported as `null` rather than a misleading partial number.
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
  change is always *visible* in the reply, so the customer can correct it. The evals on the [roadmap](../README.md#future-steps)
  are how these regressions get caught against the real model.

### 10. Smaller choices
- `Decimal` for money end to end. Defaults are resolved when an item is added, so the cart and the
  payload are explicit (`{"size": "large", "patty": "beef"}`).
- Line ids are stable and never reused, so "line 2" means the same thing all conversation.
- `send()` returns `tool_calls` only for **external** (MCP) calls, matching the brief's examples.
  Internal cart tool calls go in the trace.
- The mock server reuses the agent's menu module. That's fine for a mock: the provided server is
  the real oracle, and contract tests against it come next.

### 11. Voice: an optional front-end
```
mic (always on, 16 kHz) ─► endpointer ─► Scribe v2 STT ─► agent.send() ─► speakable text ─► Flash v2.5 TTS ─► speaker
                           speech…pause    + menu keyterms  (unchanged)    ($11.50 → "11 dollars 50")    streamed PCM
          └──────────── live "you" / "agent" level bars + status (rich) ────────────┘
```
- **A channel, not a new agent.** `main.py` reads from and replies to a channel (`read()` /
  `reply()`, used as a context manager); hands-free voice is a second implementation next to
  the keyboard. Cart, guards, confirmation and MCP handling are the same code, so every
  guarantee above holds when speaking. The only agent-side change is `REPLY_STYLE=voice`,
  which appends a short prompt section: input is a transcript (match misheard words to the menu
  by sound, ask when it matters), and replies are one or two speakable sentences that still
  state every item and the exact total at review.
- **Hands-free turn taking, locally.** The microphone streams 20 ms frames into an energy-based
  endpointer (`voice/vad.py`): a turn starts after 100 ms of audio clearly above the room
  (`VOICE_VAD_MARGIN_DB`) and ends after a pause (`VOICE_END_SILENCE_MS`); 300 ms of pre-roll
  keeps the first syllable, and blips too short to be words are dropped. The noise floor is the
  quietest frame of the last few seconds: gaps between words keep it low while you talk, and a
  steady noise that starts later (a fan) becomes the new floor instead of an endless turn.
  Anything that slips through as noise transcribes to nothing and is ignored. It needs no model
  and no extra dependency, and is unit-tested with synthetic audio. A model-based VAD is the
  upgrade path if noisy rooms matter.
- **Half-duplex, on purpose.** The microphone is ignored while the agent thinks and speaks, and
  for 300 ms afterwards, so the agent never transcribes its own voice from the speakers. The
  cost is no barge-in: interrupting the agent needs acoustic echo cancellation, which the
  roadmap covers.
- **Live feedback.** A `rich` live region shows the state (listening, hearing you,
  transcribing, thinking, speaking) and two scrolling level bars: the mic level while it is your
  turn, the level of each frame actually sent to the speaker while the agent talks. Levels are
  pushed from the audio threads into a small time-ticked meter (`voice/meter.py`), so the bars
  fall flat in silence instead of freezing. Transcript lines and ordinary output scroll above.
- **Optional by construction.** `elevenlabs`, `sounddevice` and `rich` live in the `voice`
  extra and are imported only when a speech client, an audio device or the display is created.
  A missing extra, PortAudio, microphone permission or API key becomes `VoiceUnavailableError`
  with the fix in the message. `main.py` imports the voice package only under `--voice`. CI runs
  the suite with and without the extras.
- **Recognition is biased towards the menu.** Item names, option choices and extras are sent as
  Scribe *keyterms* (derived from `menu.yaml`, so never stale), which is what turns
  "margarita" into "margherita". `no_verbatim` drops fillers and false starts; audio-event tags
  are off, so the LLM sees only words. Keyterms cost extra, so `VOICE_STT_KEYTERMS=false` turns
  them off.
- **Speakable text is deterministic.** Flash models read text literally, and the reply guards
  need the exact `$11.50` in the reply, so the model keeps writing for the screen and code
  rewrites a copy for the ear: money in words, order ids spelled out, `15-20` → "15 to 20",
  `2x` → "2", bullets and markup flattened into sentences. The screen shows the exact text.
- **Low latency where it is cheap.** Flash v2.5 (the lowest-latency TTS) streams raw PCM, which
  is played as it arrives (no MP3 decoding, no ffmpeg), re-chunked on sample boundaries so a
  network split never garbles the audio. Logs record STT latency, time to first audio and total
  playback time per turn (`voice.stt`, `voice.tts`).
- **Speech failures never end the order.** A failed transcription asks the customer to say it
  again; a failed synthesis leaves the printed reply; Ctrl+C stops the audio, not the session.
