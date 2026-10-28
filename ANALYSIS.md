# Current project analysis — completion-cooldown release

**Review date:** 2026-10-28 (Asia/Colombo)  
**Scope:** Current source tree with a bounded FIFO client queue and completion-based cooldowns. This replaces the older RPM-pacing design.

## Executive summary

This is a local, browser-session relay for DeepSeek Web and Qwen Web. It exposes OpenAI Chat Completions and a minimal Anthropic Messages adapter, asks for a prompt-format mode and a delay in seconds at interactive startup, and serializes generation requests through a bounded FIFO queue. It is not a local model or the official provider API.

The selected delay begins after the browser-backed provider stream completes; the provider gate applies it before the next stream, including tool-repair attempts. The client-facing gate separately waits until the complete API response has been sent—including the final SSE body—before it starts the cooldown before admitting the next queued generation. This is completion-based spacing, not a requests-per-minute quota. The default delay is 3 seconds and can be changed at startup.

The queue has explicit backpressure, structured queue-full 429 responses, status visibility, and cancellation-safe release. The default is one active request plus 100 pending requests per process. There is no live account/browser test, so the project remains experimental rather than production-ready.

## Queue and cooldown behavior

- `POST /v1/chat/completions` and `POST /v1/messages` are admitted FIFO, with one active generation per server process.
- The provider slot is held for the complete underlying stream. After that stream ends or is cancelled, the next provider submission waits the selected delay.
- The HTTP request slot is held through the full response stream. Its cooldown starts after the final response body is sent; a queued request cannot get to the provider early.
- Default pending capacity is 100. When full, the bridge returns a structured OpenAI- or Anthropic-shaped 429 instead of growing an unbounded in-memory backlog.
- `/queue/status` reports queue depth, active duration, provider-stream state, remaining cooldowns, and local counters without request text.
- Queue/cooldown state is in-memory and per process; it is not durable and does not coordinate separate processes or multiple workers.

Detailed design, provider-limit caveats, startup variables, and research sources are in `QUEUE_RATE_LIMIT_RESEARCH.md`. Installation and operation are in `CLAUDE_CODE_SETUP.md`.

## Security and privacy

- The APIs have no authentication and should stay bound to `127.0.0.1`; do not expose them to a LAN or public network.
- Prompts and conversation history are sent to provider websites using the logged-in browser session. Browser profiles hold sensitive session data and must not be committed or shared.
- Tool calls are parsed and schema-validated against the current request but are returned to the calling agent; the bridge does not execute them. Keep the downstream tool approval policy enabled.
- The code does not execute parsed model-generated JavaScript or Python and adds no evasion, account rotation, stealth, or provider risk-control bypass logic.
- Provider website automation may break, may be restricted by provider terms, and may trigger account controls. This implementation cannot guarantee account safety.

## Correctness hardening in this release

- Browser response replay now removes compression, content-length, digest, and hop-by-hop headers that no longer match the decompressed body; route outcomes are logged with metadata only.
- Validation failures no longer echo Pydantic's submitted `input` values.
- Unsupported text/image request content is rejected with HTTP 400 rather than an uncaught 500.
- Non-streaming provider failures return an upstream error response rather than a false successful empty completion.
- Regular OpenAI streaming chunks use a consistent completion ID and send error events without a false success finish marker.
- Queue overflow is handled as a controlled 429; cancelled waiters are removed; provider and client cooldowns are anchored to stream/response completion.

Known compatibility limits remain documented in `CLAUDE_CODE_SETUP.md`: text-only inputs, estimated token counts, incomplete generation-control support, no OpenAI Responses API, and probabilistic tool syntax because the model is prompted through a web interface rather than native function calling.

## Verification

Offline checks in the workspace passed:

- `test_parser.py` — DeepSeek and Qwen parser assertions.
- `test_tool_bridge.py` — 13 parser/validation/SSE tests.
- `test_anthropic_compat.py` — 13 adapter tests.
- `test_api_smoke.py` — both APIs, sanitized errors, unsupported input, fake provider failures, and stable streaming IDs.
- `test_browser_response.py` — 2 tests for compressed SSE replay headers, `[DONE]` preservation, and safe abort diagnostics on both browser adapters.
- `test_request_queue.py` — 6 FIFO, capacity, cancellation, completion-cooldown, and HTTP backpressure tests.
- `test_startup_config.py` — 5 prompt-order, validation, environment override, and child-process configuration tests.
- `python -m compileall -q .`, Ruff `check --select F,E9`, and `git diff --check`.

All provider streams in these tests are fake. No live Playwright login, DeepSeek/Qwen web session, Claude Code call, account-specific web-limit validation, or current provider website behavior has been tested. Do not treat offline tests as proof of live service reliability.
