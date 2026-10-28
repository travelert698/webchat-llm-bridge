# FIFO queue and completion-cooldown design research

**Prepared:** 2026-10-28 (Asia/Colombo)  
**Scope:** The existing DeepSeek/Qwen browser-session bridge, preserving its visible browser/UI workflow and OpenAI Chat Completions plus Anthropic Messages routes.

## Executive summary

The bridge now uses two coordinated controls:

1. **A bounded FIFO admission queue** for client generation requests. It smooths client bursts, preserves arrival order, and limits waiting work in memory.
2. **A completion-based cooldown**, configurable in seconds. It starts after the provider stream has fully ended, and the next provider stream waits until that interval has elapsed. Separately, the client-request gate starts the same cooldown only after the complete response—including the final SSE body—has been sent. This is the key guarantee for the scenario where a coding agent sends request A, receives its complete stream, and only then may request B start after the selected pause.

One generation request per backend process is active at a time. The browser modules use one persistent page and mutable per-turn interception state, so overlapping generations are unsafe. The queue remains useful for concurrent clients, but queue order alone is not the timing policy: the configured delay is anchored to **completion**, not to the earlier request start.

This is a **cooldown**, not an RPM quota. A 3-second value does not mean exactly 20 requests per minute: long provider responses make the actual throughput lower. No finite queue can accept an unlimited arrival rate or guarantee every client's timeout. The default queue allows 100 waiting requests; when full it sends a structured HTTP 429 rather than growing without bound or crashing.

## What provider documentation does and does not establish

- DeepSeek's current **API** rate/isolation page describes account-level concurrency limits for API models and says exceeding a limit can produce HTTP 429. It is not a published requests-per-minute or cooldown guarantee for the DeepSeek web-chat UI used here. [1](https://api-docs.deepseek.com/quick_start/rate_limit/)
- Alibaba Cloud Model Studio publishes limits that vary by model and service scope, and its API guidance discusses layered request pacing and concurrency control. These are **Model Studio API** rules, not a transferable quota for a logged-in Qwen web-chat session. [1](https://help.aliyun.com/zh/model-studio/rate-limit) [3](https://help.aliyun.com/en/model-studio/rate-limiting-best-practices)
- Thus no provider-specific universal number can be justified from those API documents. Startup asks the operator how long to pause. The default 3 seconds is an adjustable local spacing choice, not an official allowance, account guarantee, or way to bypass provider controls.

The bridge does not call provider APIs directly, rotate accounts, evade throttling, spoof browser identity, or retry provider restrictions to defeat them.

## Scheduling model

### 1. Client requests: FIFO until the entire response completes

`POST /v1/chat/completions` and `POST /v1/messages` enter the per-process FIFO request gate. One generation request owns the slot at a time. A streaming response holds the slot until the final body has been sent; errors and disconnects also release the slot. A cancelled waiter is removed so it cannot block later callers.

At request release, the gate records `next_request_at = response_complete_time + selected_delay`. The head of the FIFO queue is not promoted until that time. If there are no queued callers, the next request arriving during the cooldown waits only for the remaining time. Health checks, model listing, connectivity probes, and the local token-count estimate do not enter this generation queue.

The queue stores admission tickets, not copied prompt bodies. Default capacity is **100 pending requests** plus one active request. Configure it with `DEEPSEEK_MAX_QUEUE`, `QWEN_MAX_QUEUE`, or shared `BRIDGE_MAX_QUEUE`. `GET /queue/status` reports pending depth, active duration, cooldown remaining, and non-sensitive counters.

### 2. Provider streams: cooldown after completion, including internal retries

Every plain chat, OpenAI tool, and Anthropic adapter call uses the same wrapped `bridge.stream_response()` path. The provider slot remains owned for the full underlying browser response. On completion, error, or cancellation, the provider gate records a cooldown from that end time; the next provider call waits for it. Tool-call repair submissions pass through the same gate, so a repair model request cannot race the prior browser stream or skip the configured pause.

This prevents the earlier start-to-start interpretation: if request A takes 40 seconds and the selected delay is 3 seconds, the next provider submission is allowed no earlier than 3 seconds after A's stream completes, not immediately at A's start plus 3 seconds. At the client layer, the separate gate ensures the agent has received the complete HTTP/SSE response before another client generation begins.

### 3. Bounded overload behavior

A finite queue prevents an unbounded resource commitment. Python's async queue documentation likewise distinguishes a finite maximum from an infinite default queue. [1](https://docs.python.org/3/library/asyncio-queue.html)

When the pending queue is full:

- OpenAI callers receive HTTP 429 with `error.code = "queue_full"`.
- Anthropic Messages callers receive HTTP 429 with an Anthropic-shaped `rate_limit_error` body.
- Request content is not echoed. The body explains the queue is full and recommends backoff.

RFC 6585 defines 429 for rate limiting and says the response should explain the condition; `Retry-After` is optional. Since browser response duration is variable, the bridge does not give a misleading fixed wait estimate for queue saturation. [3](https://www.rfc-editor.org/info/rfc6585/)

This is controlled backpressure, not a promise that callers can never see a 429. Accepted requests wait in FIFO order while the process stays up; process failure/restart can interrupt this in-memory queue.

## Startup settings

On an interactive standalone launch, the questions are asked in this order:

1. prompt mode `[1] FULL passthrough` or `[2] USER message only`;
2. seconds to wait before another request (default **3 seconds**, accepted range **0–300**, decimals allowed).

`run_both.py` collects the prompt mode and separate DeepSeek/Qwen cooldown values in its own console, then passes them to each child so they do not race to read the same terminal. For non-interactive launches, environment settings apply directly; on an interactive terminal they prefill the prompt.

| Purpose | DeepSeek | Qwen | Shared fallback |
|---|---|---|---|
| Prompt mode | `PROMPT_MODE=full` or `user_only` | same | same |
| Completion cooldown seconds | `DEEPSEEK_DELAY_SECONDS` | `QWEN_DELAY_SECONDS` | `BRIDGE_DELAY_SECONDS`, then `REQUEST_DELAY_SECONDS` |
| Pending queue capacity | `DEEPSEEK_MAX_QUEUE` | `QWEN_MAX_QUEUE` | `BRIDGE_MAX_QUEUE` |

## Safety and operational limits

- Cooldown and queue state are **in-memory and per server process**. They do not coordinate separate processes, machines, or multiple Uvicorn workers. The normal Python launch uses one process; do not add workers if you require one shared scheduler.
- The queue is not durable. Restarting a server can interrupt waiting requests. Callers may also impose their own shorter network timeout while waiting.
- The delay does not discover a provider's web-session limit and cannot prevent every provider error, login failure, selector change, or site outage. Use non-sensitive test work and keep the local API bound to loopback.
- The delay applies to actual provider streams as well as the client-facing turn boundary. Tool-repair requests can therefore add cooldown time within a single client generation, intentionally favoring stable spacing over minimum latency.

## Other correctness hardening in the project

- Validation failures return generic, sanitized 422 bodies instead of echoing Pydantic's submitted `input` value.
- Unsupported text/image message content returns a clean 400 response.
- Non-streaming provider failures return an upstream error instead of a false successful empty completion.
- Regular OpenAI stream chunks use a stable completion ID and do not add a success finish marker after a stream error.

## Verification

Offline tests cover FIFO ordering, bounded capacity and 429 bodies, queue waiter cancellation, provider cooldown after stream completion, client cooldown after response-slot release, startup question order and validation, non-interactive overrides, sanitized validation errors, unsupported content, provider errors, and stable streaming IDs.

Tests use fake provider streams and an ASGI test transport. They do **not** sign into DeepSeek/Qwen or perform a live browser/provider round trip. Actual web-session behavior, account-specific limits, and client timeout settings remain unverified.
