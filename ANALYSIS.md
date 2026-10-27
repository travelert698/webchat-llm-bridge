# Current project analysis — queue-enabled release

**Review date:** 2026-10-27 (Asia/Colombo)  
**Scope:** Current source tree after the FIFO request queue and RPM limiter changes. This replaces the older pre-patch baseline review.

## Executive summary

This is a local, browser-session relay for DeepSeek Web and Qwen Web. It exposes OpenAI Chat Completions and a minimal Anthropic Messages adapter, asks for a prompt-format mode and a local provider-request pacing value at interactive startup, and serializes generation requests through a bounded FIFO queue. It is not a local model and it is not the official provider API.

The bridge now has explicit request backpressure, rate pacing for every actual browser-backed model submission (including tool-repair attempts), protocol-shaped queue-full 429 responses, queue status visibility, and cancellation-safe queue release. The rate setting does not determine, discover, or bypass a provider's actual web-chat limits. There is no live account/browser test, so the project should still be treated as experimental rather than production-ready.

## Current queue behavior

- `POST /v1/chat/completions` and `POST /v1/messages` are admitted FIFO, with one active generation per server process.
- Default pending capacity is 100. When full, the bridge returns a structured OpenAI- or Anthropic-shaped 429; it does not grow an unbounded in-memory backlog.
- The startup RPM value counts each call to the browser-backed model stream, including tool-format repair submissions, and uses smooth pacing plus a rolling 60-second count.
- `/queue/status` exposes pending depth, current active duration, recent provider starts, and local counters without request text.
- The queue is in-memory and per process. It is not durable and does not coordinate separate processes or multiple workers.

Detailed research, provider-limit caveats, configuration, and sources are in `QUEUE_RATE_LIMIT_RESEARCH.md`. Installation and operation are in `CLAUDE_CODE_SETUP.md`.

## Security and privacy

- The APIs have no authentication and should stay bound to `127.0.0.1`; do not expose them to a LAN or public network.
- Prompts and conversation history are sent to provider websites using the logged-in browser session. Browser profiles hold sensitive session data and must not be committed or shared.
- Tool calls are parsed and schema-validated against the current request but are returned to the calling agent; the bridge does not execute them. Keep the downstream tool approval policy enabled.
- The code does not execute parsed model-generated JavaScript or Python and adds no evasion, account rotation, stealth, or provider risk-control bypass logic.
- Provider website automation may break, may be restricted by provider terms, and may trigger account controls. This implementation cannot guarantee account safety.

## Correctness hardening in this release

- Validation failures no longer echo Pydantic's submitted `input` values.
- Unsupported text/image request content is rejected with HTTP 400 rather than an uncaught 500.
- Non-streaming provider failures return an upstream error response rather than a false successful empty completion.
- Regular OpenAI streaming chunks now use a consistent completion ID and send error events without a false success finish marker.
- Queue overflow is handled as a controlled 429; queued cancellations remove their tickets, and active tickets release when responses complete, fail, or disconnect.

Known compatibility limits remain documented in `CLAUDE_CODE_SETUP.md`: text-only inputs, estimated token counts, incomplete generation-control support, no OpenAI Responses API, and probabilistic tool syntax because the model is prompted through a web interface rather than native function calling.

## Verification

Offline checks in the workspace passed:

- `test_parser.py` — DeepSeek and Qwen parser assertions.
- `test_tool_bridge.py` — 13 parser/validation/SSE tests.
- `test_anthropic_compat.py` — 13 adapter tests.
- `test_api_smoke.py` — both APIs, sanitized errors, unsupported input, fake provider failures, and stable streaming IDs.
- `test_request_queue.py` — 4 FIFO, capacity, cancellation, rate pacing, and HTTP backpressure tests.
- `test_startup_config.py` — 5 prompt-order, validation, environment override, and child-process configuration tests.
- `python -m compileall -q .`, Ruff `check --select F,E9`, and `git diff --check`.

All provider streams in these tests are fake. No live Playwright login, DeepSeek/Qwen web session, Claude Code call, account-specific rate-limit validation, or current provider website behavior has been tested. Do not treat offline tests as proof of live service reliability.
