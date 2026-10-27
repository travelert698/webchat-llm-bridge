# Queue and rate-limit design research

**Prepared:** 2026-10-27 (Asia/Colombo)  
**Scope:** The existing DeepSeek/Qwen browser-session bridge, preserving its visible browser/UI workflow and OpenAI Chat Completions plus Anthropic Messages routes.

## Executive summary

The bridge needs two different controls, not just a rate counter:

1. **A bounded FIFO admission queue for client generation requests.** This smooths concurrent client arrivals, preserves order, and limits how many requests can wait in memory.
2. **A provider-submission rate limiter.** This paces each actual model submission, including a tool-call repair attempt, rather than counting only the outer client request.

The two existing browser backends are stateful: each uses one persistent page and shared per-turn interception state, and each `stream_response()` already has a process-level lock. Running multiple generations against the same browser session concurrently is therefore not a safe throughput optimization. The implementation deliberately admits **one active generation per backend process**, queues later requests FIFO, and enforces a smooth RPM ceiling for outbound submissions.

No finite queue can guarantee acceptance of an unlimited arrival rate or prevent every client-side timeout. This implementation caps the default backlog at 100 waiting requests. If that cap is reached, it responds with an explicit, protocol-shaped HTTP 429 rather than leaking memory, dropping an already-admitted request, or crashing. Clients should use backoff when they receive `queue_full`.

## What provider documentation does and does not establish

- DeepSeek's current **API** rate/isolation page describes account-level concurrency limits for API models and states that exceeding a limit can produce HTTP 429. That is useful evidence that concurrency and rate enforcement are provider-side concerns, but it is **not a published RPM guarantee for the DeepSeek web-chat UI** used by this project. [1](https://api-docs.deepseek.com/quick_start/rate_limit/)
- Alibaba Cloud Model Studio publishes quotas that vary by model and service scope, and its best-practices guide recommends layered request pacing and concurrency control. Those are **Model Studio API** rules, not a transferable quota for the consumer Qwen web-chat UI or this logged-in browser session. [1](https://help.aliyun.com/zh/model-studio/rate-limit) [3](https://help.aliyun.com/en/model-studio/rate-limiting-best-practices)
- Therefore there is no evidence-based universal number that this bridge can safely choose for every web account. Startup asks the operator. The default of 5 provider submissions/minute is only a conservative starting point, not an official provider allowance, an account guarantee, or a way to bypass provider enforcement.

The bridge does not call provider APIs directly, rotate accounts, evade rate controls, spoof browser identity, or retry provider throttling to defeat a limit.

## Design choices

### 1. FIFO queue for inbound generation requests

`POST /v1/chat/completions` and `POST /v1/messages` enter a per-process FIFO gate. One request owns the browser-backed generation slot at a time. Later requests wait in arrival order, and the slot remains held through the full response stream, completion, error, or client disconnect. Cancellation while waiting removes the waiter so it does not block later work.

The queue stores admission tickets, not whole copied prompt payloads. Its default is **100 pending requests** (plus one active request); `DEEPSEEK_MAX_QUEUE`, `QWEN_MAX_QUEUE`, or `BRIDGE_MAX_QUEUE` can change the cap. The status endpoint is `GET /queue/status`.

Health checks, model listing, connectivity probes, and Anthropic's local token-count estimate do not consume provider quota and do not wait behind generation work.

### 2. Provider-start rate limiting

The selected RPM counts starts of `bridge.stream_response()`—the point that submits a prompt to the logged-in provider session. That includes tool-call repair attempts because repairs make another model request. It does not count one outer request as one when that request causes several provider submissions.

Starts are spaced by at least `60 / RPM` seconds and checked against a rolling 60-second history. This avoids a large token-bucket burst immediately after startup, while keeping the actual number of provider starts within the configured rolling-minute cap. Separate DeepSeek and Qwen processes have independent rate histories.

### 3. Bounded overload behavior

The local pending queue is finite by design. A FIFO queue with no maximum is effectively an unbounded resource commitment; Python's async queue documentation explicitly distinguishes finite `maxsize` from the default infinite queue. [1](https://docs.python.org/3/library/asyncio-queue.html)

When the configured pending capacity is full:

- OpenAI callers receive HTTP 429 with `error.code = "queue_full"`.
- Anthropic Messages callers receive HTTP 429 with an Anthropic-shaped `rate_limit_error` body.
- The body explains that the local queue is full and recommends retrying with backoff. Request contents are not included.

RFC 6585 defines 429 for rate limiting and says the response should explain the condition; `Retry-After` is optional. This queue-full case has no dependable wait estimate because an active browser response can take a variable amount of time, so the bridge does not publish a misleading fixed `Retry-After`. [3](https://www.rfc-editor.org/info/rfc6585/)

This is controlled backpressure, not a promise that clients can never see a 429. It is safer than accepting unlimited queued work or silently losing accepted work.

### 4. Startup configuration

Standalone launches ask in this order:

1. prompt method `[1] FULL passthrough` or `[2] USER message only`;
2. maximum provider model submissions per minute (positive integer from 1 to 600; Enter selects 5).

Non-interactive launches use environment overrides or safe defaults. `run_both.py` gathers the shared prompt mode and a separate RPM choice for each provider in its own console, then passes the settings to both processes so the two child processes cannot race to read the same standard input.

Environment variables:

| Purpose | DeepSeek | Qwen | Shared fallback |
|---|---|---|---|
| Prompt mode | `PROMPT_MODE=full` or `user_only` | same | same |
| RPM | `DEEPSEEK_RPM` | `QWEN_RPM` | `BRIDGE_RPM`, then `RATE_LIMIT_RPM` |
| Pending queue capacity | `DEEPSEEK_MAX_QUEUE` | `QWEN_MAX_QUEUE` | `BRIDGE_MAX_QUEUE` |

## Safety and operational limits

- The queue and rate counter are **in-memory and per server process**. They do not coordinate two processes, two machines, or multiple Uvicorn workers. The normal `python server.py` / `python qwen_server.py` launch is one process; do not increase workers if you expect this configured rate to be a global limit.
- A server restart ends its in-memory queue. Clients with a long wait may time out on their own; the bridge cannot control their timeout settings.
- A request queue is not a durable job system. It does not persist requests across process or machine failure, and it does not automatically retry a provider failure.
- Queueing does not make the browser/UI integration native or officially supported. Provider website changes, login expiry, account restrictions, network failures, and web-chat quotas can still prevent responses. Use non-sensitive test work and keep the service bound to loopback.
- Rate settings are local caps chosen by the user. A low number is sensible when the applicable web-chat allowance is unknown; do not interpret API quota tables as the limit for a web session.

## Correctness hardening included with the queue

The new release also closes issues found during the preceding repository audit:

- validation failures return generic, sanitized 422 error bodies rather than echoing Pydantic's submitted `input` value;
- unsupported text/image message content returns a clean 400 response;
- provider failures in non-streaming OpenAI requests return 502 instead of a false 200 empty answer;
- regular OpenAI stream chunks now keep a stable completion ID and report stream failures as error events rather than appending a success finish chunk.

## Verification plan and results

The implementation adds tests for FIFO order, bounded queue capacity, cancellation removal, rolling-window pacing, both 429 response shapes, startup question order/validation, safe environment overrides, sanitized validation errors, unsupported content, provider errors, and stable stream IDs.

Offline tests use fake provider streams and an ASGI test transport. They do **not** sign in to DeepSeek/Qwen or perform a live browser/provider round trip. Live UI behavior, current account-specific web limits, and actual client timeout settings remain unverified.
