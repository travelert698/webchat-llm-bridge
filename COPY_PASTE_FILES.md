# Complete copy/paste bundle — webchat-llm-bridge

Complete runtime, test, and documentation files for the queue-enabled project. Copy each file to the same relative path in the project root. The ZIP also contains the unchanged browser/UI automation modules `app.py` and `qwen_app.py`, which are intentionally not duplicated in this bundle.

## `.gitignore`

````gitignore
# ==================== Python ====================
# Virtual environments
venv/
.venv/
env/
ENV/

# Python bytecode / caches
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environment files (secrets / local config)
.env
.env.*
!.env.example

# ==================== This project (bridges) ====================
# Browser login profiles (contain live session cookies - NEVER commit)
browser_data/
.browser_data/
qwen_browser_data/
.qwen_browser_data/

# Raw SSE debug dumps (contain prompt + AI reply content - NEVER commit)
deepseek_stream_dump.jsonl
qwen_stream_dump.jsonl
*_stream_dump.jsonl

# Logs
*.log

# ==================== IDE / OS ====================
.vscode/
.idea/
.DS_Store
Thumbs.db
````

## `README.md`

````markdown
# WebChat LLM Bridge

A local OpenAI Chat Completions / minimal Anthropic Messages adapter for the existing **visible-browser** DeepSeek Web and Qwen Web automation. It preserves the browser/UI workflow; it is not a direct-provider-API client or a local model.

## Quick start

1. Install Python 3.10+ and the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   python -m playwright install chromium
   ```

2. Start one backend:

   ```powershell
   python server.py       # DeepSeek, default port 8000
   # or
   python qwen_server.py  # Qwen, default port 8001
   ```

3. In the console, choose prompt method **1** (full conversation) or **2** (last user message only). Then enter the local provider submissions-per-minute limit. Press Enter to use the conservative default of **5**. A visible browser opens; log in manually and confirm in the server console.

4. Configure your client:

   - DeepSeek OpenAI base URL: `http://127.0.0.1:8000/v1`
   - Qwen OpenAI base URL: `http://127.0.0.1:8001/v1`
   - Anthropic Messages base URL: the corresponding server root, without `/v1`

For both backends, run `python run_both.py`; it asks for one prompt mode and separate DeepSeek/Qwen RPM values. Full setup, environment overrides, compatibility limits, and safety notes are in [`CLAUDE_CODE_SETUP.md`](CLAUDE_CODE_SETUP.md).

## Queue and local rate limit

The generation endpoints are FIFO-queued, one active browser-backed generation at a time. Every outbound model submission—including tool-repair attempts—is paced against the selected RPM. Up to 100 requests may wait by default; when full, the API returns a structured 429 instead of accumulating an unbounded backlog. Check `GET /queue/status` on either server.

The value you enter is a **local pacing cap**, not an official provider quota. DeepSeek/Qwen web-chat limits can differ from their developer API quotas and may be unpublished or account-specific. The queue does not bypass throttling or provider controls. See [`QUEUE_RATE_LIMIT_RESEARCH.md`](QUEUE_RATE_LIMIT_RESEARCH.md).

## Privacy and use

The local APIs have no authentication and bind to loopback by default; do not expose them to a network. Prompts are sent through the provider website using your logged-in browser session. Keep browser profiles private, use non-sensitive test data, and retain the downstream tool approval policy. Provider compatibility and account-safety cannot be guaranteed.

## Offline checks

```powershell
python test_parser.py
python test_tool_bridge.py
python test_anthropic_compat.py
python test_api_smoke.py
python test_request_queue.py
python test_startup_config.py
```

The tests use fake streams and do not verify live logins, provider website behavior, or actual account-specific web limits.
````

## `server.py`

````python
"""
server.py – OpenAI-compatible API server for the DeepSeek Bridge

Runs on port 8000 by default. To change: set DEEPSEEK_PORT env var, e.g.
    DEEPSEEK_PORT=8010 python server.py
Interactive startup asks for prompt mode, then a provider-submission RPM limit.
Generation requests are queued FIFO; see /queue/status for queue statistics.

Exposes the existing OpenAI-compatible Chat Completions API and a minimum
Anthropic Messages adapter for Claude Code/custom Anthropic-format clients.
Tool calls are returned to the client for approval/execution; this server does
not run client tools.
"""
import json
import os
import time
import uuid
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn
import app          # the DeepSeek bridge module (app.py)
import tool_bridge  # the shared OpenAI/tool-calling bridge
from anthropic_compat import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    handle_count_tokens,
    handle_messages,
)
from request_queue import RequestQueue, RequestQueueMiddleware
from startup_config import choose_startup_config

# ==================== Port and startup configuration ====================
PORT = int(os.environ.get("DEEPSEEK_PORT", "8000"))
PROMPT_MODE, RATE_LIMIT_RPM, MAX_PENDING_REQUESTS = choose_startup_config("deepseek")
tool_bridge.PROMPT_MODE = PROMPT_MODE   # sync the global tool-bridge mode

# ==================== Bounded client queue + provider pacing ====================
REQUEST_QUEUE = RequestQueue(
    requests_per_minute=RATE_LIMIT_RPM,
    max_pending=MAX_PENDING_REQUESTS,
)
_original_stream_response = app.stream_response


async def _rate_limited_stream_response(prompt: str):
    """Gate every underlying web-model submission, including tool retries."""
    await REQUEST_QUEUE.wait_for_provider_start()
    async for chunk in _original_stream_response(prompt):
        yield chunk


# All plain chat, tool-bridge, and Anthropic adapter calls share this wrapper.
app.stream_response = _rate_limited_stream_response

# ==================== FastAPI app ====================
api = FastAPI(title="DeepSeek Bridge API")
api.add_middleware(RequestQueueMiddleware, queue=REQUEST_QUEUE)

def extract_text(content: Any) -> str:
    """Extract plain text from OpenAI-style content (str or list of parts),
    exactly as the client sent it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                elif "image_url" in part:
                    parts.append("[image]")
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)

def render_messages(messages: list) -> str:
    """Render the conversation to one prompt according to PROMPT_MODE.

    Delegates to tool_bridge.render_messages so the chosen mode (full or
    user_only) applies to BOTH plain chat and tool requests consistently.
    """
    return tool_bridge.render_messages(messages)

class Message(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_calls: Optional[list] = None

class StreamOptions(BaseModel):
    include_usage: bool = False

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: Optional[str] = "deepseek-chat"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream_options: Optional[StreamOptions] = None
    tools: Optional[list] = None          # OpenAI tools array (hybrid path)
    tool_choice: Optional[Any] = None     # "auto" | "none" | "required" | {"function": {...}}
    parallel_tool_calls: Optional[bool] = None

# ==================== Lifespan ====================
@api.on_event("startup")
async def startup():
    await app.startup()
    print("\n" + "=" * 60)
    print(f"  DeepSeek Bridge API running at http://127.0.0.1:{PORT}")
    print("  OpenAI endpoint: POST /v1/chat/completions")
    print("  Anthropic endpoint: POST /v1/messages")
    print(f"  OpenAI base URL: http://127.0.0.1:{PORT}/v1")
    print(f"  Claude Code base URL: http://127.0.0.1:{PORT}")
    print("  Mode: full passthrough + tool calling (tool_bridge.py)")
    print(f"  Prompt mode: {PROMPT_MODE}")
    print(f"  Provider submission limit: {RATE_LIMIT_RPM} per rolling minute (smoothed)")
    print(f"  FIFO client queue: 1 active + {MAX_PENDING_REQUESTS} waiting")
    print("  Queue status: GET /queue/status")
    print("=" * 60 + "\n")

@api.on_event("shutdown")
async def shutdown():
    await app.shutdown()

# ==================== Models endpoint ====================
@api.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "deepseek-chat",
            "object": "model",
            "created": 1700000000,
            "owned_by": "deepseek-bridge"
        }]
    }

# ==================== Validation error logger ====================
@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Never log or echo the submitted prompt/tool payload on validation errors.
    print(f"[VALIDATION ERROR] {request.url.path}: {len(exc.errors())} field error(s)")
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(
            status_code=422,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid Anthropic Messages request.",
                },
            },
        )
    return JSONResponse(
        status_code=422,
        content={"error": {
            "message": "Invalid request body.",
            "type": "invalid_request_error",
            "param": None,
            "code": "validation_error",
        }},
    )

# ==================== Chat completions ====================
@api.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # ---- Hybrid tool path: when tools or tool_choice are supplied ----
    if req.tools is not None or req.tool_choice is not None:
        result = await tool_bridge.handle_chat(req, bridge=app)
        if result is not None:
            return result
        # fall through for tool_choice=none, or auto with no tools

    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided."})
    try:
        prompt = render_messages(req.messages)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": {
            "message": "Invalid or unsupported message content; this bridge accepts text-only input.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "unsupported_content",
        }})
    if not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    if not req.stream:
        full_text = ""
        full_reasoning = ""
        try:
            async for chunk in app.stream_response(prompt):
                parsed = app._parse_sse(chunk)
                if parsed and "error" in parsed:
                    return JSONResponse(status_code=502, content={"error": {
                        "message": str(parsed["error"]),
                        "type": "upstream_error",
                        "param": None,
                        "code": "provider_error",
                    }})
                if parsed:
                    full_text += parsed.get("content", "") or ""
                    full_reasoning += parsed.get("reasoning", "") or ""
        except Exception as e:
            print(f"[SERVER] upstream request failed: {e}")
            return JSONResponse(status_code=502, content={"error": {
                "message": str(e),
                "type": "upstream_error",
                "param": None,
                "code": "provider_error",
            }})

        message = {"role": "assistant", "content": full_text}
        if full_reasoning:
            message["reasoning_content"] = full_reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "deepseek-chat",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]
        }

    async def event_stream():
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = req.model or "deepseek-chat"
        yield ": connected\n\n"
        init = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(init)}\n\n"

        try:
            async for chunk in app.stream_response(prompt):
                parsed = app._parse_sse(chunk)
                if not parsed:
                    if chunk.startswith(":"):
                        yield chunk
                    continue
                if "error" in parsed:
                    error = {"error": {
                        "message": str(parsed["error"]),
                        "type": "upstream_error",
                        "code": "provider_error",
                    }}
                    yield f"data: {json.dumps(error)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                delta = {}
                if parsed.get("content"):
                    delta["content"] = parsed["content"]
                if parsed.get("reasoning"):
                    delta["reasoning_content"] = parsed["reasoning"]
                if delta:
                    item = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(item)}\n\n"

            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"[SERVER] upstream stream failed: {e}")
            err = {"error": {
                "message": str(e),
                "type": "upstream_error",
                "code": "provider_error",
            }}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ==================== Anthropic Messages API ====================
@api.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    response = await handle_messages(req, bridge=app, default_model="deepseek-chat")
    if isinstance(response, JSONResponse) and response.status_code == 400:
        # The adapter returns an Anthropic error body; log its reason without the request.
        try:
            message = json.loads(response.body.decode("utf-8"))["error"]["message"]
            print(f"[ANTHROPIC 400] request rejected: {message}")
        except Exception:
            pass
    return response


@api.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: AnthropicCountTokensRequest):
    return handle_count_tokens(req)


# ==================== Client connectivity probes ====================
@api.get("/api/hello")
async def api_hello():
    return {"status": "ok"}


@api.head("/api/hello")
async def api_hello_head():
    return Response(status_code=200)


# ==================== Queue status and health ====================
@api.get("/queue/status")
async def queue_status():
    return await REQUEST_QUEUE.snapshot()


@api.get("/health")
async def health():
    return {"status": "ok"}

# ==================== Entry point ====================
if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=PORT)
````

## `qwen_server.py`

````python
"""
qwen_server.py – OpenAI-compatible API server for the Qwen Bridge

Runs on port 8001 by default (DeepSeek server.py keeps port 8000).
To change: set QWEN_PORT env var, e.g.
    QWEN_PORT=8020 python qwen_server.py
Interactive startup asks for prompt mode, then a provider-submission RPM limit.
Generation requests are queued FIFO; see /queue/status for queue statistics.

Exposes the existing OpenAI-compatible Chat Completions API and a minimum
Anthropic Messages adapter, wired to the Qwen bridge module (qwen_app.py).
Tool calls are returned to the client; this server does not run client tools.
"""
import json
import os
import time
import uuid
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn
import qwen_app as app  # the Qwen bridge module (qwen_app.py)
import tool_bridge      # the shared OpenAI/tool-calling bridge
from anthropic_compat import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    handle_count_tokens,
    handle_messages,
)
from request_queue import RequestQueue, RequestQueueMiddleware
from startup_config import choose_startup_config

# ==================== Port and startup configuration ====================
PORT = int(os.environ.get("QWEN_PORT", "8001"))
PROMPT_MODE, RATE_LIMIT_RPM, MAX_PENDING_REQUESTS = choose_startup_config("qwen")
tool_bridge.PROMPT_MODE = PROMPT_MODE   # sync the global tool-bridge mode

# ==================== Bounded client queue + provider pacing ====================
REQUEST_QUEUE = RequestQueue(
    requests_per_minute=RATE_LIMIT_RPM,
    max_pending=MAX_PENDING_REQUESTS,
)
_original_stream_response = app.stream_response


async def _rate_limited_stream_response(prompt: str):
    """Gate every underlying web-model submission, including tool retries."""
    await REQUEST_QUEUE.wait_for_provider_start()
    async for chunk in _original_stream_response(prompt):
        yield chunk


# All plain chat, tool-bridge, and Anthropic adapter calls share this wrapper.
app.stream_response = _rate_limited_stream_response

# ==================== FastAPI app ====================
api = FastAPI(title="Qwen Bridge API")
api.add_middleware(RequestQueueMiddleware, queue=REQUEST_QUEUE)

def extract_text(content: Any) -> str:
    """Extract plain text from OpenAI-style content (str or list of parts),
    exactly as the client sent it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                elif "image_url" in part:
                    parts.append("[image]")
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)

def render_messages(messages: list) -> str:
    """Render the conversation to one prompt according to PROMPT_MODE.

    Delegates to tool_bridge.render_messages so the chosen mode (full or
    user_only) applies to BOTH plain chat and tool requests consistently.
    """
    return tool_bridge.render_messages(messages)

class Message(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_calls: Optional[list] = None

class StreamOptions(BaseModel):
    include_usage: bool = False

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: Optional[str] = "qwen-auto"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream_options: Optional[StreamOptions] = None
    tools: Optional[list] = None          # OpenAI tools array (hybrid path)
    tool_choice: Optional[Any] = None     # "auto" | "none" | "required" | {"function": {...}}
    parallel_tool_calls: Optional[bool] = None

# ==================== Lifespan ====================
@api.on_event("startup")
async def startup():
    await app.startup()
    print("\n" + "=" * 60)
    print(f"  Qwen Bridge API running at http://127.0.0.1:{PORT}")
    print("  OpenAI endpoint: POST /v1/chat/completions")
    print("  Anthropic endpoint: POST /v1/messages")
    print(f"  OpenAI base URL: http://127.0.0.1:{PORT}/v1")
    print(f"  Claude Code base URL: http://127.0.0.1:{PORT}")
    print("  Model: qwen-auto")
    print("  Mode: full passthrough + tool calling (tool_bridge.py)")
    print(f"  Prompt mode: {PROMPT_MODE}")
    print(f"  Provider submission limit: {RATE_LIMIT_RPM} per rolling minute (smoothed)")
    print(f"  FIFO client queue: 1 active + {MAX_PENDING_REQUESTS} waiting")
    print("  Queue status: GET /queue/status")
    print("=" * 60 + "\n")

@api.on_event("shutdown")
async def shutdown():
    await app.shutdown()

# ==================== Models endpoint ====================
@api.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen-auto",
            "object": "model",
            "created": 1700000000,
            "owned_by": "qwen-bridge"
        }]
    }

# ==================== Validation error logger ====================
@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Never log or echo the submitted prompt/tool payload on validation errors.
    print(f"[VALIDATION ERROR] {request.url.path}: {len(exc.errors())} field error(s)")
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(
            status_code=422,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid Anthropic Messages request.",
                },
            },
        )
    return JSONResponse(
        status_code=422,
        content={"error": {
            "message": "Invalid request body.",
            "type": "invalid_request_error",
            "param": None,
            "code": "validation_error",
        }},
    )

# ==================== Chat completions ====================
@api.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # ---- Hybrid tool path: when tools or tool_choice are supplied ----
    if req.tools is not None or req.tool_choice is not None:
        result = await tool_bridge.handle_chat(req, bridge=app)
        if result is not None:
            return result
        # fall through for tool_choice=none, or auto with no tools

    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided."})
    try:
        prompt = render_messages(req.messages)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": {
            "message": "Invalid or unsupported message content; this bridge accepts text-only input.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "unsupported_content",
        }})
    if not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    if not req.stream:
        full_text = ""
        full_reasoning = ""
        try:
            async for chunk in app.stream_response(prompt):
                parsed = app._parse_sse(chunk)
                if parsed and "error" in parsed:
                    return JSONResponse(status_code=502, content={"error": {
                        "message": str(parsed["error"]),
                        "type": "upstream_error",
                        "param": None,
                        "code": "provider_error",
                    }})
                if parsed:
                    full_text += parsed.get("content", "") or ""
                    full_reasoning += parsed.get("reasoning", "") or ""
        except Exception as e:
            print(f"[SERVER] upstream request failed: {e}")
            return JSONResponse(status_code=502, content={"error": {
                "message": str(e),
                "type": "upstream_error",
                "param": None,
                "code": "provider_error",
            }})

        message = {"role": "assistant", "content": full_text}
        if full_reasoning:
            message["reasoning_content"] = full_reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "qwen-auto",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]
        }

    async def event_stream():
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = req.model or "qwen-auto"
        yield ": connected\n\n"
        init = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(init)}\n\n"

        try:
            async for chunk in app.stream_response(prompt):
                parsed = app._parse_sse(chunk)
                if not parsed:
                    if chunk.startswith(":"):
                        yield chunk
                    continue
                if "error" in parsed:
                    error = {"error": {
                        "message": str(parsed["error"]),
                        "type": "upstream_error",
                        "code": "provider_error",
                    }}
                    yield f"data: {json.dumps(error)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                delta = {}
                if parsed.get("content"):
                    delta["content"] = parsed["content"]
                if parsed.get("reasoning"):
                    delta["reasoning_content"] = parsed["reasoning"]
                if delta:
                    item = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(item)}\n\n"

            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"[SERVER] upstream stream failed: {e}")
            err = {"error": {
                "message": str(e),
                "type": "upstream_error",
                "code": "provider_error",
            }}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ==================== Anthropic Messages API ====================
@api.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    response = await handle_messages(req, bridge=app, default_model="qwen-auto")
    if isinstance(response, JSONResponse) and response.status_code == 400:
        try:
            message = json.loads(response.body.decode("utf-8"))["error"]["message"]
            print(f"[ANTHROPIC 400] request rejected: {message}")
        except Exception:
            pass
    return response


@api.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: AnthropicCountTokensRequest):
    return handle_count_tokens(req)


# ==================== Client connectivity probes ====================
@api.get("/api/hello")
async def api_hello():
    return {"status": "ok"}


@api.head("/api/hello")
async def api_hello_head():
    return Response(status_code=200)


# ==================== Queue status and health ====================
@api.get("/queue/status")
async def queue_status():
    return await REQUEST_QUEUE.snapshot()


@api.get("/health")
async def health():
    return {"status": "ok"}

# ==================== Entry point ====================
if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=PORT)
````

## `request_queue.py`

````python
"""Bounded FIFO admission and per-provider request pacing for the local bridge.

The bridge's browser modules have single-session state. This scheduler therefore
admits one model-generation HTTP request per server at a time, queues later
OpenAI/Anthropic generation requests in FIFO order, and limits every underlying
provider submission (including tool-repair attempts) to the configured RPM.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestQueueFull(Exception):
    """Raised when the bounded number of waiting client requests is reached."""

    def __init__(self, pending: int, capacity: int) -> None:
        self.pending = pending
        self.capacity = capacity
        super().__init__(f"request queue full ({pending}/{capacity} waiting)")


@dataclass
class RequestLease:
    """A granted turn in the FIFO request queue."""

    _owner: "RequestQueue"
    _token: object
    _released: bool = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            # Keep admission state consistent if the client disconnect cancels
            # the ASGI task while its response is being finalized.
            await asyncio.shield(self._owner._release_request(self._token))


class RequestQueue:
    """Fair in-process request gate plus smooth rolling-window RPM limiter."""

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        requests_per_minute: int,
        max_pending: int = 100,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if isinstance(requests_per_minute, bool) or requests_per_minute < 1:
            raise ValueError("requests_per_minute must be a positive integer")
        if isinstance(max_pending, bool) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.requests_per_minute = int(requests_per_minute)
        self.max_pending = int(max_pending)
        self._clock = clock
        self._sleep = sleep

        self._condition = asyncio.Condition()
        self._waiting: deque[object] = deque()
        self._active_token: object | None = None
        self._active_since: float | None = None
        self._accepted = 0
        self._completed = 0
        self._rejected = 0
        self._cancelled_while_waiting = 0

        self._rate_lock = asyncio.Lock()
        self._provider_starts: deque[float] = deque()
        self._next_provider_start: float | None = None
        self._provider_requests_started = 0
        self._provider_interval = self.WINDOW_SECONDS / self.requests_per_minute

    async def acquire_request(self) -> RequestLease:
        """Wait for a FIFO turn; fail fast only when the pending queue is full."""
        token = object()
        async with self._condition:
            if len(self._waiting) >= self.max_pending:
                self._rejected += 1
                raise RequestQueueFull(len(self._waiting), self.max_pending)
            self._waiting.append(token)
            self._accepted += 1
            try:
                await self._condition.wait_for(
                    lambda: (
                        self._active_token is None
                        and bool(self._waiting)
                        and self._waiting[0] is token
                    )
                )
                self._waiting.popleft()
                self._active_token = token
                self._active_since = self._clock()
                return RequestLease(self, token)
            except BaseException:
                try:
                    self._waiting.remove(token)
                    self._cancelled_while_waiting += 1
                except ValueError:
                    # The ticket may already have been promoted when a task is
                    # cancelled; middleware's finally block releases its lease.
                    pass
                self._condition.notify_all()
                raise

    async def _release_request(self, token: object) -> None:
        async with self._condition:
            if self._active_token is token:
                self._active_token = None
                self._active_since = None
                self._completed += 1
                self._condition.notify_all()

    async def wait_for_provider_start(self) -> float:
        """Charge one provider submission and wait until it may safely begin.

        Combines smooth pacing (no burst at process start) with a rolling
        60-second count. Returns the monotonic start time for observability.
        A cancelled wait does not consume quota.
        """
        while True:
            async with self._rate_lock:
                now = self._clock()
                while self._provider_starts and now - self._provider_starts[0] >= self.WINDOW_SECONDS:
                    self._provider_starts.popleft()

                earliest = self._next_provider_start
                if len(self._provider_starts) >= self.requests_per_minute:
                    window_expiry = self._provider_starts[0] + self.WINDOW_SECONDS
                    earliest = window_expiry if earliest is None else max(earliest, window_expiry)

                delay = 0.0 if earliest is None else earliest - now
                if delay <= 1e-9:
                    started_at = now
                    self._provider_starts.append(started_at)
                    self._next_provider_start = started_at + self._provider_interval
                    self._provider_requests_started += 1
                    return started_at

            await self._sleep(delay)

    async def snapshot(self) -> dict[str, int | bool | float | None]:
        """Return a small, prompt-free queue/rate-limit status snapshot."""
        async with self._condition:
            pending = len(self._waiting)
            active = self._active_token is not None
            active_seconds = (
                max(0.0, self._clock() - self._active_since)
                if self._active_since is not None
                else None
            )
            accepted = self._accepted
            completed = self._completed
            rejected = self._rejected
            cancelled = self._cancelled_while_waiting

        async with self._rate_lock:
            now = self._clock()
            while self._provider_starts and now - self._provider_starts[0] >= self.WINDOW_SECONDS:
                self._provider_starts.popleft()
            next_start = self._next_provider_start
            if len(self._provider_starts) >= self.requests_per_minute:
                window_expiry = self._provider_starts[0] + self.WINDOW_SECONDS
                next_start = window_expiry if next_start is None else max(next_start, window_expiry)
            wait = 0.0 if next_start is None else max(0.0, next_start - now)
            starts_last_60 = len(self._provider_starts)
            total_provider_starts = self._provider_requests_started

        return {
            "requests_per_minute": self.requests_per_minute,
            "provider_requests_started_last_60_seconds": starts_last_60,
            "total_provider_requests_started": total_provider_starts,
            "minimum_spacing_seconds": round(self._provider_interval, 3),
            "next_provider_start_in_seconds": round(wait, 3),
            "max_pending": self.max_pending,
            "pending": pending,
            "active": active,
            "active_seconds": round(active_seconds, 3) if active_seconds is not None else None,
            "accepted_requests": accepted,
            "completed_requests": completed,
            "queue_full_rejections": rejected,
            "cancelled_while_waiting": cancelled,
        }


class RequestQueueMiddleware:
    """FIFO-gate model-generation requests for the full ASGI response lifetime."""

    QUEUED_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})

    def __init__(self, app: ASGIApp, queue: RequestQueue) -> None:
        self.app = app
        self.queue = queue

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method", "").upper() != "POST"
            or scope.get("path") not in self.QUEUED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        try:
            lease = await self.queue.acquire_request()
        except RequestQueueFull as exc:
            message = (
                f"The local request queue is full ({exc.pending} of "
                f"{exc.capacity} pending). Retry later with backoff."
            )
            headers = {"Cache-Control": "no-store", "X-Bridge-Queue": "full"}
            if scope.get("path") == "/v1/messages":
                response = JSONResponse(
                    status_code=429,
                    headers=headers,
                    content={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": message,
                        },
                    },
                )
            else:
                response = JSONResponse(
                    status_code=429,
                    headers=headers,
                    content={
                        "error": {
                            "message": message,
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "queue_full",
                        }
                    },
                )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            # Starlette fully consumes streaming iterators inside the ASGI call.
            # Releasing here therefore keeps the browser slot through stream end,
            # and also releases it on exceptions or client disconnects.
            await lease.release()
````

## `startup_config.py`

````python
"""Interactive startup choices shared by the DeepSeek and Qwen servers.

The chosen RPM is a *local outbound provider-submission pace*, not a promise
about either provider's unpublished web-chat limits. Environment variables are
supported for defaults, non-interactive runs, and supervised child processes.
"""
from __future__ import annotations

import os
import sys

DEFAULT_PROMPT_MODE = "full"
DEFAULT_RPM = 5
DEFAULT_MAX_PENDING = 100
MIN_RPM = 1
MAX_RPM = 600
MIN_MAX_PENDING = 1
MAX_MAX_PENDING = 10_000


def _is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def _skip_prompts() -> bool:
    return os.environ.get("BRIDGE_STARTUP_CONFIGURED", "").strip() == "1"


def _read_choice(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return ""


def _environment_prompt_mode() -> str | None:
    env = os.environ.get("PROMPT_MODE", "").strip().lower()
    aliases = {"1": "full", "2": "user_only"}
    if env in ("full", "user_only"):
        return env
    if env in aliases:
        return aliases[env]
    if env:
        print(f"[config] Ignoring invalid PROMPT_MODE={env!r}; choose a mode below.")
    return None


def choose_prompt_mode() -> str:
    """Return ``full`` or ``user_only``; interactive terminals always ask."""
    env_mode = _environment_prompt_mode()
    if not _is_interactive() or _skip_prompts():
        mode = env_mode or DEFAULT_PROMPT_MODE
        if not _is_interactive() and env_mode is None:
            print(f"[config] Non-interactive input; using prompt mode {mode!r}.")
        return mode

    default_mode = env_mode or DEFAULT_PROMPT_MODE
    default_choice = "2" if default_mode == "user_only" else "1"
    print("\n=== Prompt mode selection ===")
    print("  [1] FULL passthrough (system + all messages)")
    print("  [2] USER message only")
    print("  Both modes work with Continue.dev and DeepSeek Harness.")
    while True:
        choice = _read_choice(f"Choose method 1 or 2 [default {default_choice}]: ")
        if choice == "":
            return default_mode
        if choice == "1":
            return "full"
        if choice == "2":
            return "user_only"
        print("Please enter 1 or 2 (or press Enter for the default).")


def _provider_env_names(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        return ("DEEPSEEK_RPM", "BRIDGE_RPM", "RATE_LIMIT_RPM")
    if normalized in ("qwen", "qw"):
        return ("QWEN_RPM", "BRIDGE_RPM", "RATE_LIMIT_RPM")
    raise ValueError(f"Unknown bridge provider: {provider!r}")


def _environment_rpm(provider: str) -> int | None:
    for name in _provider_env_names(provider):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            rpm = int(value.strip())
            if MIN_RPM <= rpm <= MAX_RPM:
                return rpm
        except (TypeError, ValueError):
            pass
        print(f"[config] Ignoring invalid {name}={value!r}; expected {MIN_RPM}..{MAX_RPM}.")
    return None


def choose_rpm(provider: str) -> int:
    """Select a conservative positive requests-per-minute value.

    The limit counts each outbound browser-backed model submission, including
    a tool-call repair attempt, rather than merely counting inbound HTTP calls.
    """
    env_rpm = _environment_rpm(provider)
    default_rpm = env_rpm or DEFAULT_RPM
    interactive = _is_interactive() and not _skip_prompts()
    if not interactive:
        if env_rpm is None:
            print(
                f"[config] Non-interactive input; using conservative local pace "
                f"of {default_rpm} provider requests/minute for {provider.title()}."
            )
        return default_rpm

    print(f"\n=== {provider.title()} provider request pacing ===")
    print("This is a local cap on model submissions, including tool-repair attempts.")
    print("It does not reveal or override the provider's own web-chat limits.")
    while True:
        value = _read_choice(
            f"Maximum provider requests per minute [{default_rpm}]: "
        )
        if value == "":
            return default_rpm
        try:
            rpm = int(value)
        except ValueError:
            rpm = 0
        if MIN_RPM <= rpm <= MAX_RPM:
            return rpm
        print(f"Please enter a whole number from {MIN_RPM} to {MAX_RPM}.")


def _provider_queue_env_names(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        return ("DEEPSEEK_MAX_QUEUE", "BRIDGE_MAX_QUEUE")
    if normalized in ("qwen", "qw"):
        return ("QWEN_MAX_QUEUE", "BRIDGE_MAX_QUEUE")
    raise ValueError(f"Unknown bridge provider: {provider!r}")


def choose_max_pending(provider: str) -> int:
    """Return the bounded count of requests allowed to wait in the FIFO queue."""
    for name in _provider_queue_env_names(provider):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value.strip())
            if MIN_MAX_PENDING <= size <= MAX_MAX_PENDING:
                return size
        except (TypeError, ValueError):
            pass
        print(
            f"[config] Ignoring invalid {name}={value!r}; expected "
            f"{MIN_MAX_PENDING}..{MAX_MAX_PENDING}."
        )

    return DEFAULT_MAX_PENDING


def choose_startup_config(provider: str) -> tuple[str, int, int]:
    """Ask the startup questions in the requested order and return settings."""
    mode = choose_prompt_mode()
    rpm = choose_rpm(provider)
    max_pending = choose_max_pending(provider)
    return mode, rpm, max_pending


def environment_for_child(
    provider: str,
    prompt_mode: str,
    rpm: int,
    max_pending: int,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create child env with the selected per-provider settings made explicit."""
    env = dict(os.environ if base is None else base)
    normalized = provider.strip().lower()
    if normalized in ("deepseek", "ds"):
        rpm_name, queue_name = "DEEPSEEK_RPM", "DEEPSEEK_MAX_QUEUE"
    elif normalized in ("qwen", "qw"):
        rpm_name, queue_name = "QWEN_RPM", "QWEN_MAX_QUEUE"
    else:
        raise ValueError(f"Unknown bridge provider: {provider!r}")
    env["PROMPT_MODE"] = prompt_mode
    env[rpm_name] = str(rpm)
    env[queue_name] = str(max_pending)
    env["BRIDGE_STARTUP_CONFIGURED"] = "1"
    return env
````

## `tool_bridge.py`

````python
"""
tool_bridge.py — Hybrid Tool-Calling Bridge Module (shared by server.py and qwen_server.py)

DESIGN:
  - "full" mode renders every message into a single role-labelled prompt; it
    preserves the conversation content but is not a native role-by-role
    provider pass-through. "user_only" mode sends only the latest user turn.
  - Only when the request carries client-side `tools`, their definitions are
    included in the prompt. Tagged/OpenAI-like calls are parsed, then each name
    and argument object is checked against this request's JSON Schemas before
    returning OpenAI-compatible `tool_calls`. The bridge never executes tools;
    permission prompts and execution remain with the calling client.

The parser accepts the bridge's `<tool_call>` format, DeepSeek native call
markers and DSML wrappers (including V3.2/V4/V4.1 forms), a conservative
exact-response JSON fallback, Anthropic-style `<invoke>` blocks, and the
existing `<run_code>` wrapper. It never evaluates model-generated code.
Malformed outputs may trigger a bounded repair retry.

Tool definitions are request-scoped and capped at MAX_TOOLS_INJECTED; returned
calls are capped at MAX_TOOL_CALLS. Protocol compatibility is tested with fake
provider streams; live agent/provider support is not guaranteed.

PROMPT MODE (global, set by server at startup):
  "full"      -> system + ALL messages rendered with role labels
  "user_only" -> ONLY the last user message
"""
from __future__ import annotations

import json
import re
import secrets
import time
from html import unescape
from html.parser import HTMLParser
from typing import Any, Optional

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import SchemaError as JSONSchemaError
from jsonschema import validate as validate_json_schema
from jsonschema.validators import validator_for

from fastapi.responses import JSONResponse, StreamingResponse

# ==================== Config ====================
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
DEEPSEEK_TOOL_CALL_START = "<｜tool▁calls▁begin｜>"
DEEPSEEK_TOOL_CALL_END = "<｜tool▁calls▁end｜>"
RUN_CODE_START = "<run_code"
RUN_CODE_END = "</run_code>"
MAX_REPAIR_RETRIES = 2
MAX_TOOL_BLOCK_CHARS = 400_000
MAX_TOOL_CALLS = 32
MAX_TOTAL_CHARS = 300_000

# Prompt mode — GLOBAL toggle, set at startup by the server (question + env).
PROMPT_MODE = "full"

# Tool access is request-scoped: only names present in the caller's `tools`
# array are described to the model, and returned calls are validated against
# that exact request and its JSON Schema. Never execute a returned tool here.
TOOL_WHITELIST: set[str] | None = None
MAX_TOOLS_INJECTED = 64
MAX_TOOL_SCHEMA_CHARS = 24_000


# ==================== Text helpers ====================
def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" or "image_url" in part:
                    raise ValueError("image content is not supported by the web-chat bridge (text-only)")
                if "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _g(m: Any, key: str, default: Any = None) -> Any:
    if isinstance(m, dict):
        return m.get(key, default)
    try:
        return getattr(m, key, default)
    except Exception:
        return default


def render_messages(messages: list) -> str:
    """Render the conversation to one prompt according to PROMPT_MODE.

    PROMPT_MODE "full":      system + ALL messages in order, roles labelled,
                             tool_calls preserved — exact original passthrough.
    PROMPT_MODE "user_only": ONLY the last user message (classic chat).
    """
    if PROMPT_MODE == "user_only":
        last_user = None
        for m in messages:
            if str(_g(m, "role", "") or "").lower() == "user":
                last_user = m
        if last_user is None:
            return ""
        raw = _g(last_user, "content", None)
        return extract_text(raw) if raw is not None else ""

    blocks = []
    for m in messages:
        role = str(_g(m, "role", "") or "user").lower()
        raw_content = _g(m, "content", None)
        content = extract_text(raw_content) if raw_content is not None else ""

        if role == "system":
            blocks.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            blocks.append(f"[USER]\n{content}")
        elif role == "assistant":
            text = f"[ASSISTANT]\n{content}".rstrip()
            tool_calls = _g(m, "tool_calls", None)
            if tool_calls:
                try:
                    text += "\n[tool_calls]\n" + json.dumps(
                        [tc for tc in tool_calls], ensure_ascii=False, default=str
                    )
                except Exception:
                    pass
            blocks.append(text)
        elif role == "tool":
            name = _g(m, "name", None) or "tool"
            blocks.append(f"[TOOL RESULT ({name})]\n{content}")
        elif role == "function":
            blocks.append(f"[FUNCTION]\n{content}")
        else:
            blocks.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(blocks)


# ==================== Tool injection ====================
def _tool_function(tool: Any) -> dict:
    """Accept OpenAI function tools and the normalized Anthropic shape."""
    if not isinstance(tool, dict):
        return {}
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn
    return tool


def _tool_schema(tool: Any) -> dict:
    fn = _tool_function(tool)
    schema = fn.get("parameters")
    if schema is None:
        schema = fn.get("input_schema")
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        raise ValueError("tool input schema must be a JSON object")
    return schema


def _validate_local_schema_refs(schema: Any) -> None:
    """Disallow remote JSON Schema references; only request-local refs are used."""
    if isinstance(schema, dict):
        for key in ("$ref", "$dynamicRef", "$recursiveRef"):
            ref = schema.get(key)
            if ref is not None and (not isinstance(ref, str) or not ref.startswith("#")):
                raise ValueError("external JSON Schema references are unsupported")
        for value in schema.values():
            _validate_local_schema_refs(value)
    elif isinstance(schema, list):
        for value in schema:
            _validate_local_schema_refs(value)


def tool_specs(tools: Any) -> dict[str, dict]:
    """Return request-scoped tool name -> validated JSON Schema definitions."""
    specs: dict[str, dict] = {}
    if not isinstance(tools, list):
        return specs
    for tool in tools:
        fn = _tool_function(tool)
        raw_name = fn.get("name")
        if raw_name is None:
            continue
        if not isinstance(raw_name, str):
            raise ValueError("tool function name must be a string")
        name = raw_name.strip()
        if not name or (TOOL_WHITELIST is not None and name not in TOOL_WHITELIST):
            continue
        if name in specs:
            raise ValueError(f"duplicate tool name: {name}")
        schema = _tool_schema(tool)
        _validate_local_schema_refs(schema)
        validator_for(schema).check_schema(schema)
        specs[name] = schema
    return specs


def serialize_tools(tools: Any) -> str:
    lines: list[str] = []
    if not isinstance(tools, list):
        return ""
    seen: set[str] = set()
    for tool in tools:
        fn = _tool_function(tool)
        name = str(fn.get("name") or "").strip()
        if not name or name in seen:
            continue
        if TOOL_WHITELIST is not None and name not in TOOL_WHITELIST:
            continue
        seen.add(name)
        description = str(fn.get("description") or "").strip().replace("\n", " ")
        schema = _tool_schema(tool)
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        if len(schema_text) > MAX_TOOL_SCHEMA_CHARS:
            raise ValueError(f"schema for tool {name!r} is too large")
        lines.append(
            f"Tool name: {name}\n"
            f"Description: {description or '(no description)'}\n"
            f"Input JSON Schema: {schema_text}"
        )
        if len(seen) >= MAX_TOOLS_INJECTED:
            break
    return "\n\n".join(lines)


def build_tool_block(
    tools_text: str,
    forced_tool: Optional[str],
    parallel_tool_calls: Optional[bool] = None,
) -> str:
    parts = [
        "[SYSTEM RULE — TOOL CALLING]",
        "You may use only the tools listed below. Never invent a tool name or argument.",
        "Preferred tool-call format (no explanatory text around a call):",
        '<tool_call><name>TOOL_NAME</name><args>{"key": "value"}</args></tool_call>',
        "Use valid JSON for args and follow the listed input schema.",
        "A response may contain multiple separate <tool_call> blocks when useful.",
        "Follow a compatible alternate wrapper only if the caller's system prompt explicitly requires it (for example DeepSeek-native markers, DeepSeek DSML, or <run_code> with tools.NAME({...})). The bridge extracts calls but never executes code.",
        "If no tool is needed, answer normally. The calling client controls tool permissions.",
    ]
    if forced_tool == "__required__":
        parts.append("You MUST call one or more of the listed tools.")
    elif forced_tool:
        parts.append(f"You MUST call the listed tool named: {forced_tool}")
    if parallel_tool_calls is False:
        parts.append("Call at most one tool in this response.")
    parts.extend(["", "[TOOLS AVAILABLE]", tools_text])
    return "\n".join(parts)


def resolve_tool_choice(tool_choice: Any) -> Optional[str]:
    """Return a required tool name, ``__required__``, or None for auto/none.

    Invalid choices are rejected instead of silently turning into automatic
    tool selection. Named tools use the standard function-choice object shape.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice in ("auto", "none"):
            return None
        if choice in ("required", "any"):
            return "__required__"
        raise ValueError("tool_choice string must be auto, none, or required")
    if isinstance(tool_choice, dict):
        kind = str(tool_choice.get("type") or "").lower()
        if kind in ("auto", "none"):
            return None
        if kind in ("required", "any"):
            return "__required__"
        # OpenAI: {"type":"function","function":{"name":"..."}}
        # Anthropic requests are normalized to this same function shape.
        function = tool_choice.get("function")
        if kind in ("", "function") and isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if name:
                return name
        raise ValueError("tool_choice must be auto, none, required, or a named function tool")
    raise ValueError("tool_choice must be a string or object")


# ==================== Tool block parsing (multi-format) ====================
def _strip_fences(block: str) -> str:
    """Unwrap a code fence only when it encloses the whole response."""
    match = re.fullmatch(
        r"\s*```[A-Za-z0-9_-]*[ \t]*(?:\r?\n)?(.*?)\r?\n?```\s*",
        block,
        re.S,
    )
    return match.group(1).strip() if match else block.strip()


def _decode_escaped_text(raw: str) -> str:
    """Decode common JavaScript/JSON string escapes without evaluating code."""
    simple = {
        "n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
        "v": "\v", "0": "\0", "\\": "\\", '"': '"', "'": "'",
        "`": "`", "$": "$", "/": "/",
    }
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] != "\\":
            out.append(raw[i])
            i += 1
            continue
        if i + 1 >= len(raw):
            out.append("\\")
            break
        nxt = raw[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "u" and i + 5 < len(raw):
            try:
                codepoint = int(raw[i + 2:i + 6], 16)
            except ValueError:
                out.extend(("\\", nxt))
                i += 2
                continue
            i += 6
            if 0xD800 <= codepoint <= 0xDBFF and raw[i:i + 2] == "\\u" and i + 5 < len(raw):
                try:
                    low = int(raw[i + 2:i + 6], 16)
                except ValueError:
                    low = -1
                if 0xDC00 <= low <= 0xDFFF:
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                    i += 6
            out.append(chr(codepoint) if not 0xD800 <= codepoint <= 0xDFFF else f"\\u{codepoint:04x}")
        elif nxt == "x" and i + 3 < len(raw):
            digits = raw[i + 2:i + 4]
            try:
                out.append(chr(int(digits, 16)))
                i += 4
            except ValueError:
                out.extend(("\\", nxt))
                i += 2
        else:
            # Preserve unknown escapes literally; json.dumps will escape them.
            out.extend(("\\", nxt))
            i += 2
    return "".join(out)


class _RelaxedJSONParser:
    """Small, non-evaluating parser for JSON and common JS object literals.

    It accepts quoted or identifier keys, single/double/backtick strings,
    trailing commas, JSON literals, and identifier values used by run_code.
    It intentionally rejects executable expressions and never calls eval().
    """

    _NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
    _IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")

    def __init__(self, text: str, variables: Optional[dict[str, Any]] = None):
        self.text = text
        self.variables = variables or {}
        self.pos = 0

    def parse(self) -> Any:
        value = self._value()
        self._space()
        if self.pos != len(self.text):
            raise ValueError("unexpected trailing characters")
        return value

    def _space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _string(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        raw: list[str] = []
        while self.pos < len(self.text):
            char = self.text[self.pos]
            self.pos += 1
            if char == quote:
                return _decode_escaped_text("".join(raw))
            if char == "\\" and self.pos < len(self.text):
                raw.append(char)
                raw.append(self.text[self.pos])
                self.pos += 1
            else:
                raw.append(char)
        raise ValueError("unterminated string")

    def _identifier(self) -> str:
        match = self._IDENTIFIER.match(self.text, self.pos)
        if not match:
            raise ValueError("expected an identifier")
        self.pos = match.end()
        return match.group(0)

    def _object(self) -> dict[str, Any]:
        self.pos += 1  # {
        result: dict[str, Any] = {}
        self._space()
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
            return result
        while True:
            self._space()
            if self.pos >= len(self.text):
                raise ValueError("unterminated object")
            quoted_key = self.text[self.pos] in "\"'`"
            if quoted_key:
                key = self._string()
            else:
                key = self._identifier()
            self._space()
            if self.pos < len(self.text) and self.text[self.pos] == ":":
                self.pos += 1
                value = self._value()
            elif not quoted_key:
                # JavaScript object shorthand, e.g. {path, count: 2}.
                value = self.variables.get(key, key)
            else:
                raise ValueError("expected ':' after object key")
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
            self._space()
            if self.pos >= len(self.text):
                raise ValueError("unterminated object")
            if self.text[self.pos] == "}":
                self.pos += 1
                return result
            if self.text[self.pos] != ",":
                raise ValueError("expected ',' or '}'")
            self.pos += 1
            self._space()
            if self.pos < len(self.text) and self.text[self.pos] == "}":
                self.pos += 1  # allow a trailing comma
                return result

    def _array(self) -> list[Any]:
        self.pos += 1  # [
        result: list[Any] = []
        self._space()
        if self.pos < len(self.text) and self.text[self.pos] == "]":
            self.pos += 1
            return result
        while True:
            result.append(self._value())
            self._space()
            if self.pos >= len(self.text):
                raise ValueError("unterminated array")
            if self.text[self.pos] == "]":
                self.pos += 1
                return result
            if self.text[self.pos] != ",":
                raise ValueError("expected ',' or ']' ")
            self.pos += 1
            self._space()
            if self.pos < len(self.text) and self.text[self.pos] == "]":
                self.pos += 1  # allow a trailing comma
                return result

    def _value(self) -> Any:
        self._space()
        if self.pos >= len(self.text):
            raise ValueError("expected a value")
        char = self.text[self.pos]
        if char in "\"'`":
            return self._string()
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        number = self._NUMBER.match(self.text, self.pos)
        if number:
            token = number.group(0)
            self.pos = number.end()
            return float(token) if any(c in token for c in ".eE") else int(token)
        token = self._identifier()
        if token == "true":
            return True
        if token == "false":
            return False
        if token in ("null", "undefined"):
            return None
        return self.variables.get(token, token)


def _load_jsonish(raw: str, variables: Optional[dict[str, Any]] = None) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _RelaxedJSONParser(raw, variables).parse()


def _parse_invoke_calls(block: str) -> list[dict[str, Any]]:
    """Parse Anthropic-style <invoke>/<parameter> blocks without executing them."""
    class InvokeParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.calls: list[dict[str, Any]] = []
            self.current: Optional[dict[str, Any]] = None
            self.parameter: Optional[str] = None
            self.parameter_text: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
            values = dict(attrs)
            if tag == "invoke":
                name = str(values.get("name") or "").strip()
                self.current = {"name": name, "arguments": {}}
            elif tag == "parameter" and self.current is not None:
                self.parameter = str(values.get("name") or "").strip()
                self.parameter_text = []

        def handle_data(self, data: str) -> None:
            if self.parameter is not None:
                self.parameter_text.append(data)

        def handle_endtag(self, tag: str) -> None:
            if tag == "parameter" and self.current is not None and self.parameter:
                raw_value = "".join(self.parameter_text).strip()
                try:
                    value = _load_jsonish(raw_value)
                except (ValueError, TypeError, RecursionError):
                    value = raw_value
                self.current["arguments"][self.parameter] = value
                self.parameter = None
                self.parameter_text = []
            elif tag == "invoke" and self.current is not None:
                if self.current.get("name"):
                    self.calls.append(self.current)
                self.current = None
                self.parameter = None
                self.parameter_text = []

    if "<invoke" not in block.lower():
        return []
    parser = InvokeParser()
    try:
        parser.feed(block)
        parser.close()
    except Exception:
        return []
    return parser.calls


_DSML_TAG_PREFIX = r"[｜|]+\s*DSML\s*[｜|]+"
_DSML_INVOKE_START = re.compile(r"<\s*" + _DSML_TAG_PREFIX + r"\s*invoke\b([^>]*)>", re.I)
_DSML_INVOKE_END = re.compile(r"</\s*" + _DSML_TAG_PREFIX + r"\s*invoke\s*>", re.I)
_DSML_PARAMETER_START = re.compile(r"<\s*" + _DSML_TAG_PREFIX + r"\s*parameter\b([^>]*)>", re.I)
_DSML_PARAMETER_END = re.compile(r"</\s*" + _DSML_TAG_PREFIX + r"\s*parameter\s*>", re.I)
_DSML_ATTRIBUTE = re.compile(
    r"([A-Za-z_][\w.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.S,
)


def _dsml_attributes(raw: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in _DSML_ATTRIBUTE.finditer(raw):
        value = next((part for part in match.groups()[1:] if part is not None), "")
        attributes[match.group(1).lower()] = unescape(value)
    return attributes


def _parse_dsml_calls(block: str) -> list[dict[str, Any]]:
    """Parse DeepSeek DSML invoke blocks (V3.2/V4/V4.1 spellings)."""
    calls: list[dict[str, Any]] = []
    position = 0
    while (start := _DSML_INVOKE_START.search(block, position)) is not None:
        end = _DSML_INVOKE_END.search(block, start.end())
        if end is None:
            break
        name = _dsml_attributes(start.group(1)).get("name", "").strip()
        if name:
            body = block[start.end():end.start()]
            arguments: dict[str, Any] = {}
            parameter_position = 0
            found_parameters = False
            while (parameter := _DSML_PARAMETER_START.search(body, parameter_position)) is not None:
                parameter_end = _DSML_PARAMETER_END.search(body, parameter.end())
                if parameter_end is None:
                    found_parameters = False
                    arguments = {}
                    break
                attrs = _dsml_attributes(parameter.group(1))
                parameter_name = attrs.get("name", "").strip()
                if not parameter_name or parameter_name in arguments:
                    arguments = {}
                    found_parameters = False
                    break
                raw_value = unescape(body[parameter.end():parameter_end.start()])
                string_flag = attrs.get("string", "").lower()
                if string_flag == "true":
                    value: Any = raw_value
                else:
                    candidate = raw_value.strip()
                    try:
                        value = _load_jsonish(candidate)
                    except (ValueError, TypeError, RecursionError):
                        if string_flag == "false":
                            arguments = {}
                            found_parameters = False
                            break
                        value = raw_value
                arguments[parameter_name] = value
                found_parameters = True
                parameter_position = parameter_end.end()

            if found_parameters:
                calls.append({"name": name, "arguments": arguments})
            elif not _DSML_PARAMETER_START.search(body):
                # V3.2-style emitters may put one JSON object directly inside invoke.
                try:
                    direct = _load_jsonish(unescape(body).strip())
                except (ValueError, TypeError, RecursionError):
                    direct = None
                if isinstance(direct, dict):
                    calls.append({"name": name, "arguments": direct})
        position = end.end()
    return calls


def _parse_tool_call_xml(block: str) -> Optional[dict]:
    block = _strip_fences(block)
    name: Optional[str] = None
    args: Any = None

    name_match = re.search(r"<name>(.*?)</name>", block, re.S | re.I)
    args_match = re.search(r"<args>(.*?)</args>", block, re.S | re.I)
    if name_match:
        name = name_match.group(1).strip()
    if args_match:
        raw = args_match.group(1).strip()
        try:
            args = _load_jsonish(raw)
        except (ValueError, TypeError, RecursionError):
            args = raw

    if not name:
        return None
    if isinstance(args, str):
        try:
            args = _load_jsonish(args)
        except (ValueError, TypeError, RecursionError):
            pass
    if not isinstance(args, dict):
        args = {"value": args} if args is not None else {}
    return {"name": name, "arguments": args}


def _parse_js_object(raw: str, templates: dict[str, Any]) -> Optional[dict]:
    try:
        obj = _load_jsonish(raw, templates)
    except (ValueError, TypeError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _skip_quoted(source: str, start: int) -> int:
    quote = source[start]
    i = start + 1
    while i < len(source):
        if source[i] == "\\":
            i += 2
        elif source[i] == quote:
            return i + 1
        else:
            i += 1
    return len(source)


def _skip_comment(source: str, start: int) -> int:
    if source.startswith("//", start):
        end = source.find("\n", start + 2)
        return len(source) if end < 0 else end + 1
    if source.startswith("/*", start):
        end = source.find("*/", start + 2)
        return len(source) if end < 0 else end + 2
    return start


def _remove_js_comments(source: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(source):
        if source[i] in "\"'`":
            end = _skip_quoted(source, i)
            out.append(source[i:end])
            i = end
        elif source.startswith("//", i) or source.startswith("/*", i):
            end = _skip_comment(source, i)
            out.append("\n" if "\n" in source[i:end] else " ")
            i = end
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def _balanced_object_end(source: str, start: int) -> Optional[int]:
    if start >= len(source) or source[start] != "{":
        return None
    stack: list[str] = []
    closing = {"{": "}", "[": "]", "(": ")"}
    i = start
    while i < len(source):
        char = source[i]
        if char in "\"'`":
            i = _skip_quoted(source, i)
            continue
        if source.startswith("//", i) or source.startswith("/*", i):
            i = _skip_comment(source, i)
            continue
        if char in closing:
            stack.append(closing[char])
        elif char in "}])":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    return None


def _parse_run_code(block: str) -> list[dict[str, Any]]:
    gt = block.find(">")
    if gt != -1 and block.lstrip().startswith("<run_code"):
        block = block[gt + 1:]
    block = re.sub(r"</run_code\s*>\s*$", "", block, flags=re.I)

    templates: dict[str, Any] = {}
    literal = r"(?:`(?:\\.|[^`])*`|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    assignment = re.compile(r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*(" + literal + r")\s*;?", re.S)
    for match in assignment.finditer(block):
        try:
            templates[match.group(1)] = _load_jsonish(match.group(2), templates)
        except (ValueError, TypeError):
            continue

    calls: list[dict[str, Any]] = []
    i = 0
    while i < len(block):
        if block[i] in "\"'`":
            i = _skip_quoted(block, i)
            continue
        if block.startswith("//", i) or block.startswith("/*", i):
            i = _skip_comment(block, i)
            continue
        if block.startswith("tools.", i) and (i == 0 or not (block[i - 1].isalnum() or block[i - 1] in "_$")):
            name_match = re.match(r"tools\.([A-Za-z_$][\w$]*)\s*\(", block[i:])
            if name_match:
                name = name_match.group(1)
                open_paren = i + name_match.end() - 1
                arg_start = open_paren + 1
                while arg_start < len(block) and block[arg_start].isspace():
                    arg_start += 1
                arg_end = _balanced_object_end(block, arg_start)
                if arg_end is not None:
                    args = _parse_js_object(_remove_js_comments(block[arg_start:arg_end]), templates)
                    if args is not None:
                        calls.append({"name": name, "arguments": args})
                    i = arg_end
                    continue
        i += 1
    return calls


def _normalized_json_call(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = str(item.get("name") or function.get("name") or "").strip()
    if not name:
        return None
    missing = object()
    args = item.get("arguments", item.get("args", item.get("parameters", item.get("input", missing))))
    if args is missing:
        args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = _load_jsonish(args)
        except (ValueError, TypeError, RecursionError):
            args = {"value": args}
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        args = {"value": args}
    return {"name": name, "arguments": args}


def parse_tool_blocks(kind: str, block: str) -> list[dict[str, Any]]:
    if kind == "dsml":
        return _parse_dsml_calls(block)
    if kind == "tool_call":
        invoked = _parse_invoke_calls(block)
        if invoked:
            return invoked
        parsed = _parse_tool_call_xml(block)
        if parsed:
            return [parsed]
        # Some emitters put a JSON object/array directly inside the wrapper.
        return parse_tool_blocks("json", block)
    if kind == "run_code":
        return _parse_run_code(block)
    if kind == "json":
        candidate = _strip_fences(block)
        try:
            parsed = _load_jsonish(candidate)
        except (ValueError, TypeError, RecursionError):
            return []
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            items = parsed["tool_calls"]
        else:
            items = parsed if isinstance(parsed, list) else [parsed]
        return [call for item in items if (call := _normalized_json_call(item)) is not None]
    return []


# ==================== Stream sieve (multi-format) ====================
class Sieve:
    """Extract tagged tool blocks without losing markers split across chunks."""

    _MARKERS = (
        (TOOL_CALL_START, TOOL_CALL_END, "tool_call"),
        # DeepSeek-family native text markers (Unicode and common ASCII renderings).
        (DEEPSEEK_TOOL_CALL_START, DEEPSEEK_TOOL_CALL_END, "json"),
        ("<|tool▁calls▁begin|>", "<|tool▁calls▁end|>", "json"),
        ("<｜tool_calls_begin｜>", "<｜tool_calls_end｜>", "json"),
        ("<|tool_calls_begin|>", "<|tool_calls_end|>", "json"),
        # DeepSeek DSML wrappers used by V3.2/V4 and V4.1 (space/no-space variants).
        ("<｜DSML｜ calls>", "</｜DSML｜ calls>", "dsml"),
        ("<｜｜DSML｜｜ calls>", "</｜｜DSML｜｜ calls>", "dsml"),
        ("<|DSML| calls>", "</|DSML| calls>", "dsml"),
        ("<｜DSML｜calls>", "</｜DSML｜calls>", "dsml"),
        ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", "dsml"),
        ("<｜DSML｜function_calls>", "</｜DSML｜function_calls>", "dsml"),
        ("<｜DSML｜ tool_calls>", "</｜DSML｜ tool_calls>", "dsml"),
        ("<｜DSML｜ function_calls>", "</｜DSML｜ function_calls>", "dsml"),
        # Some versions omit the enclosing calls container.
        ("<｜DSML｜ invoke", "</｜DSML｜ invoke>", "dsml"),
        ("<｜｜DSML｜｜ invoke", "</｜｜DSML｜｜ invoke>", "dsml"),
        ("<|DSML| invoke", "</|DSML| invoke>", "dsml"),
        # Generic structured wrapper used by some chat models.
        ("<tool_calls>", "</tool_calls>", "json"),
        (RUN_CODE_START, RUN_CODE_END, "run_code"),
    )

    def __init__(self) -> None:
        self._carry = ""
        self._in_block = False
        self._block = ""
        self._current_start: Optional[str] = None
        self._current_end: Optional[str] = None
        self._current_kind = "tool_call"

    @staticmethod
    def _suffix_prefix_length(text: str, markers: tuple[str, ...]) -> int:
        """Length of the longest suffix that could be the start of a marker."""
        for size in range(min(len(text), max(map(len, markers)) - 1), 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in markers):
                return size
        return 0

    def _reset_block(self) -> None:
        self._in_block = False
        self._block = ""
        self._current_start = None
        self._current_end = None
        self._current_kind = "tool_call"

    def feed(self, text: str):
        out: list = []
        self._carry += text
        while self._carry:
            if not self._in_block:
                found = []
                for start, end, kind in self._MARKERS:
                    idx = self._carry.find(start)
                    if idx >= 0:
                        found.append((idx, start, end, kind))
                if not found:
                    keep = self._suffix_prefix_length(
                        self._carry, tuple(item[0] for item in self._MARKERS)
                    )
                    safe_len = len(self._carry) - keep
                    if safe_len:
                        out.append(("content", self._carry[:safe_len]))
                        self._carry = self._carry[safe_len:]
                    break
                idx, start, end, kind = min(found, key=lambda item: item[0])
                if idx:
                    out.append(("content", self._carry[:idx]))
                self._carry = self._carry[idx + len(start):]
                self._in_block = True
                self._current_start = start
                self._current_end = end
                self._current_kind = kind
                self._block = ""
                continue

            assert self._current_end is not None
            idx = self._carry.find(self._current_end)
            if idx < 0:
                keep = self._suffix_prefix_length(self._carry, (self._current_end,))
                safe_len = len(self._carry) - keep
                if safe_len:
                    self._block += self._carry[:safe_len]
                    self._carry = self._carry[safe_len:]
                if len(self._block) > MAX_TOOL_BLOCK_CHARS:
                    out.append(("invalid_block", self._current_kind, self._block))
                    self._reset_block()
                break

            self._block += self._carry[:idx]
            self._carry = self._carry[idx + len(self._current_end):]
            out.append(("tool_block", self._current_kind, self._block))
            self._reset_block()

        return out

    def finish(self):
        """Flush buffered text and report an unclosed tag as invalid, not a call."""
        out: list = []
        if self._in_block:
            raw = (self._current_start or "") + self._block + self._carry
            out.append(("invalid_block", self._current_kind, raw))
        elif self._carry:
            starts = tuple(item[0] for item in self._MARKERS)
            if self._suffix_prefix_length(self._carry, starts):
                out.append(("invalid_block", "tool_call", self._carry))
            else:
                out.append(("content", self._carry))
        self._reset_block()
        self._carry = ""
        return out


# ==================== OpenAI-compatible response helpers ====================
def _openai_error(message: str, status_code: int = 400) -> JSONResponse:
    error_type = "invalid_request_error" if status_code < 500 else "server_error"
    return JSONResponse(
        status_code=status_code,
        content={"error": {
            "message": str(message),
            "type": error_type,
            "param": None,
            "code": None,
        }},
    )


# ==================== SSE wire helpers ====================
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_base(model: str, response_id: str, created: int) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }


def _role_chunk(model: str, response_id: str, created: int) -> str:
    payload = _chunk_base(model, response_id, created)
    payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    return _sse(payload)


def _content_chunk(text: str, model: str, response_id: str, created: int) -> str:
    payload = _chunk_base(model, response_id, created)
    payload["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    return _sse(payload)


def _reasoning_chunk(text: str, model: str, response_id: str, created: int) -> str:
    payload = _chunk_base(model, response_id, created)
    payload["choices"] = [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]
    return _sse(payload)


def _tool_call_chunk(
    name: str,
    arguments: dict,
    call_id: str,
    index: int,
    model: str,
    response_id: str,
    created: int,
) -> str:
    payload = _chunk_base(model, response_id, created)
    payload["choices"] = [{
        "index": 0,
        "delta": {
            "tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }]
        },
        "finish_reason": None,
    }]
    return _sse(payload)


def _finish_chunk(reason: str, model: str, response_id: str, created: int) -> str:
    payload = _chunk_base(model, response_id, created)
    payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": reason}]
    return _sse(payload)


def _error_chunk(message: str, model: str) -> str:
    del model  # The OpenAI-compatible streaming error envelope has no completion chunk.
    return _sse({"error": {
        "message": message,
        "type": "server_error",
        "param": None,
        "code": None,
    }})


# ==================== Bridge interaction ====================
async def _run_bridge_turn(bridge, prompt: str) -> list:
    events: list = []
    try:
        async for chunk in bridge.stream_response(prompt):
            parsed = bridge._parse_sse(chunk) if hasattr(bridge, "_parse_sse") else None
            if not parsed:
                continue
            if "error" in parsed:
                events.append(("error", str(parsed["error"])))
                break
            if parsed.get("reasoning"):
                events.append(("reasoning", parsed["reasoning"]))
            if parsed.get("content"):
                events.append(("content", parsed["content"]))
    except Exception as e:
        events.append(("error", str(e)))
    return events


def _events_to_chunks(
    events: list,
    model: str,
    allowed_tools: Optional[dict[str, dict]] = None,
    response_id: Optional[str] = None,
    created: Optional[int] = None,
) -> tuple[list, Optional[tuple]]:
    response_id = response_id or f"chatcmpl-{secrets.token_hex(6)}"
    created = int(time.time()) if created is None else created
    sieve = Sieve()
    out: list = []
    pending: list[str] = []
    raw_blocks: list[tuple[str, str]] = []
    invalid_block: Optional[str] = None

    def consume(sieve_events: list) -> None:
        nonlocal invalid_block
        for event in sieve_events:
            if event[0] == "content":
                pending.append(event[1])
            elif event[0] == "tool_block":
                raw_blocks.append((event[1], event[2]))
            elif event[0] == "invalid_block":
                invalid_block = "incomplete or oversized tool block"

    for kind, value in events:
        if kind == "error":
            return [_error_chunk(value, model)], ("error", str(value))
        if kind == "content":
            consume(sieve.feed(value))
        elif kind == "reasoning":
            out.append(_reasoning_chunk(value, model, response_id, created))

    consume(sieve.finish())
    if invalid_block:
        return out, ("unparseable", invalid_block)

    if not raw_blocks:
        text = "".join(pending)
        # Some web models ignore the XML instruction and return a bare JSON
        # function call. Accept it only when it names request-scoped tools.
        json_calls = parse_tool_blocks("json", text) if text else []
        if json_calls and allowed_tools is not None:
            unknown = [call.get("name") for call in json_calls if call.get("name") not in allowed_tools]
            if unknown:
                return out, ("unparseable", f"unrequested tool: {unknown[0]}")
            raw_blocks.append(("json", text))
            pending.clear()
        else:
            if text:
                out.append(_content_chunk(text, model, response_id, created))
            out.append(_finish_chunk("stop", model, response_id, created))
            return out, None

    calls: list[dict] = []
    for block_kind, raw in raw_blocks:
        calls.extend(parse_tool_blocks(block_kind, raw))

    if not calls:
        return out, ("unparseable", "tool block")
    if len(calls) > MAX_TOOL_CALLS:
        return out, ("unparseable", "too many tool calls in one response")

    # A model may only call tools supplied in this exact request. Validate both
    # the name and arguments before returning calls to the client for execution.
    if allowed_tools is not None:
        for call in calls:
            name = call.get("name")
            if name not in allowed_tools:
                return out, ("unparseable", f"unrequested tool: {name}")
            schema = allowed_tools[name]
            try:
                validate_json_schema(instance=call.get("arguments", {}), schema=schema)
            except (JSONSchemaValidationError, JSONSchemaError, TypeError, ValueError, RecursionError) as exc:
                return out, ("unparseable", f"invalid arguments for {name}: {exc}")

    text = "".join(pending)
    if text:
        out.append(_content_chunk(text, model, response_id, created))
    for index, call in enumerate(calls):
        out.append(_tool_call_chunk(
            call["name"], call["arguments"], f"call_{secrets.token_hex(8)}", index,
            model, response_id, created,
        ))
    out.append(_finish_chunk("tool_calls", model, response_id, created))
    return out, ("parsed", calls)


# ==================== Main entry ====================
async def handle_chat(req, bridge) -> Any:
    tools = _g(req, "tools", None)
    tool_choice = _g(req, "tool_choice", None)
    parallel_tool_calls = _g(req, "parallel_tool_calls", None)
    has_tools = isinstance(tools, list) and len(tools) > 0

    choice_is_none = (
        isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"
    ) or (
        isinstance(tool_choice, dict)
        and str(tool_choice.get("type") or "").lower() == "none"
    )
    if choice_is_none:
        return None
    try:
        forced_tool = resolve_tool_choice(tool_choice)
    except ValueError as exc:
        return _openai_error(str(exc), 400)

    if not has_tools:
        if forced_tool is not None:
            return _openai_error(
                "tool_choice requires tools, but no callable tools were provided", 400
            )
        return None

    model = _g(req, "model") or "deepseek-chat"
    messages = list(_g(req, "messages") or [])
    stream = bool(_g(req, "stream", False))

    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        return _openai_error("parallel_tool_calls must be a boolean", 400)
    if not messages:
        return _openai_error("messages must be a non-empty list", 400)
    if len(tools) > MAX_TOOLS_INJECTED:
        return _openai_error("too many tools in one request", 413)

    try:
        allowed_tools = tool_specs(tools)
    except (JSONSchemaError, TypeError, ValueError, RecursionError) as exc:
        return _openai_error(f"invalid tool schema: {exc}", 400)
    if not allowed_tools:
        return _openai_error("no usable function tools were supplied", 400)
    if forced_tool and forced_tool != "__required__" and forced_tool not in allowed_tools:
        return _openai_error("tool_choice names a tool not present in tools", 400)

    total = 0
    for message in messages:
        content = _g(message, "content")
        if isinstance(content, str):
            total += len(content)
        calls = _g(message, "tool_calls", None)
        if calls:
            total += len(json.dumps(calls, ensure_ascii=False, default=str))
    if total > MAX_TOTAL_CHARS:
        return _openai_error("message payload too large", 413)

    try:
        base_prompt = render_messages(messages)
        tools_text = serialize_tools(tools)
    except (ValueError, TypeError, RecursionError) as exc:
        return _openai_error(str(exc), 400)
    if len(base_prompt) + len(tools_text) > MAX_TOTAL_CHARS:
        return _openai_error("prompt and tool schemas are too large", 413)

    tool_block = build_tool_block(tools_text, forced_tool, parallel_tool_calls)
    prompt = base_prompt + "\n\n" + tool_block
    chunks: list[str] = []
    retries = 0
    response_id = f"chatcmpl-{secrets.token_hex(6)}"
    created = int(time.time())

    while True:
        events = await _run_bridge_turn(bridge, prompt)
        turn_chunks, outcome = _events_to_chunks(
            events, model, allowed_tools, response_id, created
        )

        if outcome is not None and outcome[0] == "error":
            chunks.extend(turn_chunks)
            break
        if outcome is None and forced_tool == "__required__":
            outcome = ("unparseable", "tool_choice requires a tool call")
        if (
            outcome is not None
            and outcome[0] == "parsed"
            and parallel_tool_calls is False
            and len(outcome[1]) > 1
        ):
            outcome = ("unparseable", "parallel_tool_calls=false allows only one call")
        if outcome is not None and outcome[0] == "parsed" and forced_tool not in (None, "__required__"):
            calls = outcome[1]
            if not calls or any(call.get("name") != forced_tool for call in calls):
                outcome = ("unparseable", "model did not use the required tool")

        if outcome is None or outcome[0] == "parsed":
            chunks.extend(turn_chunks)
            break

        retries += 1
        if retries > MAX_REPAIR_RETRIES:
            chunks = [_error_chunk("model failed to produce a valid, allowed tool call", model)]
            break
        call_count_instruction = (
            "Reply with exactly one correctly formatted <tool_call> block. "
            if parallel_tool_calls is False
            else "Reply with exactly one or more correctly formatted <tool_call> blocks. "
        )
        prompt = (
            base_prompt + "\n\n" + tool_block + "\n\n"
            "[REPAIR] Your previous reply was not a valid call to an available tool. "
            + call_count_instruction
            + "Use a listed tool name and valid JSON arguments matching its schema. "
            "Do not include other text."
        )

    if stream:
        is_error = False
        if chunks:
            try:
                is_error = "error" in json.loads(chunks[0][len("data: "):].strip())
            except (json.JSONDecodeError, IndexError):
                pass
        if chunks and not is_error:
            chunks.insert(0, _role_chunk(model, response_id, created))

        async def gen():
            for chunk in chunks:
                yield chunk
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    content_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    for chunk in chunks:
        try:
            obj = json.loads(chunk[len("data: "):].strip())
        except Exception:
            continue
        if "error" in obj:
            error = obj["error"]
            message = error.get("message", "tool bridge error") if isinstance(error, dict) else str(error)
            return _openai_error(message, 502)
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                function = tc.get("function") or {}
                tool_calls.append({
                    "id": tc.get("id"), "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    },
                })
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]

    message: dict = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{secrets.token_hex(6)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
````

## `anthropic_compat.py`

````python
"""Minimal Anthropic Messages API adapter for the webchat bridges.

This module exposes a small Anthropic-compatible surface over the project's
existing DeepSeek/Qwen web-chat backends. It does not make those providers
Anthropic models. Unsupported media/Anthropic-specific features are rejected
or ignored explicitly rather than silently passed through as malformed data.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import tool_bridge

MAX_PROMPT_CHARS = 300_000
PING_INTERVAL_S = 15.0


class _MessagesBase(BaseModel):
    model: str = "webchat-bridge"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    system: Any = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Any = None
    model_config = ConfigDict(extra="allow")


class AnthropicMessagesRequest(_MessagesBase):
    max_tokens: int = Field(gt=0)
    stream: bool = False


class AnthropicCountTokensRequest(_MessagesBase):
    pass


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _estimate_tokens(text: str) -> int:
    """Rough character-based estimate; provider token usage is unavailable."""
    return max(1, (len(text) + 3) // 4) if text else 0


def _error_response(message: str, status_code: int = 400) -> JSONResponse:
    error_type = "invalid_request_error" if status_code < 500 else "api_error"
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": str(message)},
        },
    )


def _error_from_json_response(response: JSONResponse) -> JSONResponse:
    try:
        body = json.loads(response.body.decode("utf-8"))
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    error = body.get("error", {})
    if isinstance(error, dict):
        message = error.get("message") or error.get("error") or body.get("detail", "Bridge request failed")
    else:
        message = str(error or body.get("detail", "Bridge request failed"))
    status_code = getattr(response, "status_code", 502)
    return _error_response(message, status_code if status_code >= 400 else 502)


def _error_message(response: JSONResponse) -> str:
    converted = _error_from_json_response(response)
    try:
        return json.loads(converted.body.decode("utf-8"))["error"]["message"]
    except Exception:
        return "Bridge request failed"


def _block_text(content: Any) -> str:
    """Flatten supported Anthropic text content; reject images instead of lying."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("content must be text or a list of Anthropic content blocks")

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = str(block.get("type") or "")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind in ("thinking", "redacted_thinking", "signature_delta"):
            # The web backends cannot preserve Anthropic-signed thinking blocks.
            continue
        elif kind in ("image", "document", "image_url"):
            raise ValueError(f"Anthropic {kind} content is not supported by this text-only bridge")
        elif "text" in block:
            parts.append(str(block["text"]))
        else:
            raise ValueError(f"unsupported Anthropic content block type: {kind or 'unknown'}")
    return "\n".join(part for part in parts if part)


def normalize_messages(request: Any) -> list[dict[str, Any]]:
    """Convert Anthropic messages/content blocks to the bridge's role-labelled form."""
    normalized: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}

    system_text = _block_text(_get(request, "system"))
    if system_text:
        normalized.append({"role": "system", "content": system_text})

    raw_messages = _get(request, "messages", None)
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")

    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "").lower()
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"unsupported message role: {role or 'missing'}")

        content = message.get("content")
        if role == "system":
            normalized.append({"role": "system", "content": _block_text(content)})
            continue

        if isinstance(content, str) or content is None:
            normalized.append({"role": role, "content": content or ""})
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or a list of content blocks")

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            kind = str(block.get("type") or "")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "tool_use" and role == "assistant":
                name = str(block.get("name") or "").strip()
                call_id = str(block.get("id") or f"call_{uuid.uuid4().hex[:16]}")
                arguments = block.get("input", {})
                if arguments is None:
                    arguments = {}
                if not name:
                    raise ValueError("tool_use block is missing its tool name")
                if not isinstance(arguments, dict):
                    raise ValueError(f"tool_use input for {name!r} must be a JSON object")
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
                tool_names_by_id[call_id] = name
            elif kind == "tool_result" and role == "user":
                call_id = str(block.get("tool_use_id") or "")
                tool_name = tool_names_by_id.get(call_id, "tool")
                result_text = _block_text(block.get("content"))
                if block.get("is_error"):
                    result_text = "[tool reported an error]\n" + result_text
                text_parts.append(f"[TOOL RESULT ({tool_name})]\n{result_text}")
            elif kind in ("thinking", "redacted_thinking"):
                # This backend cannot validate or continue Anthropic signatures.
                continue
            elif kind in ("image", "document", "image_url"):
                raise ValueError(f"Anthropic {kind} content is not supported by this text-only bridge")
            else:
                raise ValueError(f"unsupported {role} content block type: {kind or 'unknown'}")

        normalized_message: dict[str, Any] = {
            "role": role,
            "content": "\n".join(part for part in text_parts if part) or None,
        }
        if tool_calls:
            normalized_message["tool_calls"] = tool_calls
        normalized.append(normalized_message)

    return normalized


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert client-defined Anthropic tools to OpenAI function definitions.

    Provider-managed/server tools (for example computer or web-search tools)
    are not executable by this browser-backed bridge and are rejected rather
    than silently converted into client functions.
    """
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    converted: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool definition must be an object")
        name = str(tool.get("name") or "").strip()
        tool_type = str(tool.get("type") or "").lower()
        schema = tool.get("input_schema")
        if tool_type and tool_type not in ("custom", "function"):
            raise ValueError("server-side Anthropic tools are unsupported; supply client-defined tools with input_schema")
        if schema is None:
            if tool_type:
                raise ValueError("server-side Anthropic tools are unsupported; supply client-defined tools with input_schema")
            raise ValueError("client-defined Anthropic tools must include input_schema")
        if not name:
            raise ValueError("each client-defined tool must include a name")
        if name in names:
            raise ValueError(f"duplicate tool name: {name}")
        names.add(name)
        if not isinstance(schema, dict):
            raise ValueError(f"input_schema for {name!r} must be a JSON object")
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": schema,
            },
        })
    return converted


def _disable_parallel_tool_use(choice: Any) -> bool:
    if not isinstance(choice, dict) or "disable_parallel_tool_use" not in choice:
        return False
    value = choice["disable_parallel_tool_use"]
    if not isinstance(value, bool):
        raise ValueError("tool_choice.disable_parallel_tool_use must be a boolean")
    return value


def normalize_tool_choice(choice: Any) -> Any:
    """Map Anthropic auto/any/tool/none choices to the bridge's tool choice."""
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        value = choice.lower()
        if value in ("auto", "none", "required"):
            return value
        raise ValueError(f"unsupported tool_choice: {choice}")
    if not isinstance(choice, dict):
        raise ValueError("tool_choice must be an object")
    _disable_parallel_tool_use(choice)
    kind = str(choice.get("type") or "").lower()
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        name = str(choice.get("name") or "").strip()
        if not name:
            raise ValueError("tool_choice type 'tool' requires a name")
        return {"type": "function", "function": {"name": name}}
    raise ValueError(f"unsupported Anthropic tool_choice type: {kind or 'missing'}")


def _make_prompt(request: Any, messages: Optional[list[dict[str, Any]]] = None) -> str:
    normalized = messages if messages is not None else normalize_messages(request)
    prompt = tool_bridge.render_messages(normalized)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("message payload is too large")
    return prompt


def _sse_event(event_type: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n"


def _message_start(message_id: str, model: str, input_tokens: int) -> str:
    return _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def _message_delta(stop_reason: str, output_tokens: int) -> str:
    return _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def _message_stop() -> str:
    return _sse_event("message_stop", {"type": "message_stop"})


def _stream_error(message: str) -> str:
    return _sse_event("error", {
        "type": "error",
        "error": {"type": "api_error", "message": str(message)},
    })


def _anthropic_message(
    model: str,
    content: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _anthropic_stop_reason(finish_reason: Any, has_tool_calls: bool = False) -> str:
    if has_tool_calls or finish_reason in ("tool_calls", "function_call", "tool_use"):
        return "tool_use"
    if finish_reason in ("length", "max_tokens"):
        return "max_tokens"
    if finish_reason in ("content_filter", "refusal"):
        return "refusal"
    return "end_turn"


def _parse_openai_sse(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return []
    objects: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _convert_tool_completion(result: Any, model: str, input_tokens: int) -> Any:
    if isinstance(result, JSONResponse):
        return _error_from_json_response(result)
    if not isinstance(result, dict):
        return _error_response("tool bridge returned an invalid response", 502)
    choices = result.get("choices") or []
    if not choices:
        return _error_response("tool bridge returned no completion choice", 502)
    message = choices[0].get("message") or {}
    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            return _error_response(f"invalid JSON arguments returned for tool {name!r}", 502)
        if not name or not isinstance(arguments, dict):
            return _error_response("invalid tool call returned by bridge", 502)
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:20]}",
            "name": name,
            "input": arguments,
        })
    stop_reason = _anthropic_stop_reason(choices[0].get("finish_reason"), bool(calls))
    output_chars = len(text or "") + sum(len(json.dumps(block, ensure_ascii=False)) for block in blocks if block["type"] == "tool_use")
    response = _anthropic_message(
        model=model,
        content=blocks,
        input_tokens=input_tokens,
        output_tokens=max(1, (output_chars + 3) // 4) if output_chars else 0,
        stop_reason=stop_reason,
    )
    return response


async def _stream_plain(bridge: Any, prompt: str, model: str, input_tokens: int):
    message_id = f"msg_{uuid.uuid4().hex}"
    yield _message_start(message_id, model, input_tokens)
    block_index = 0
    block_open = False
    output_text: list[str] = []
    iterator = bridge.stream_response(prompt).__aiter__()
    pending: Optional[asyncio.Task] = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(iterator.__anext__())
            try:
                chunk = await asyncio.wait_for(asyncio.shield(pending), timeout=PING_INTERVAL_S)
                pending = None
            except asyncio.TimeoutError:
                yield _sse_event("ping", {"type": "ping"})
                continue
            except StopAsyncIteration:
                pending = None
                break

            parsed = bridge._parse_sse(chunk) if hasattr(bridge, "_parse_sse") else None
            if not parsed:
                continue
            if parsed.get("error"):
                yield _stream_error(parsed["error"])
                return
            text = parsed.get("content") or ""
            if not text:
                # Provider reasoning is intentionally not exposed as an
                # Anthropic thinking block: it has no Anthropic signature.
                continue
            if not block_open:
                yield _sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                block_open = True
            output_text.append(text)
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        if block_open:
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop", "index": block_index
            })
        yield _message_delta("end_turn", _estimate_tokens("".join(output_text)))
        yield _message_stop()
    except asyncio.CancelledError:
        if pending is not None:
            pending.cancel()
        raise
    except Exception as exc:
        yield _stream_error(str(exc))
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                await pending
        close = getattr(iterator, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def _completion_blocks_from_stream(stream_objects: list[dict[str, Any]]) -> tuple[list[tuple[str, Any]], str]:
    """Collect already-buffered OpenAI chunks while preserving block order."""
    sequence: list[tuple[str, Any]] = []
    tool_by_index: dict[int, dict[str, Any]] = {}
    last_stop_reason = "stop"
    for obj in stream_objects:
        if "error" in obj:
            err = obj["error"]
            message = err.get("message", "bridge error") if isinstance(err, dict) else str(err)
            raise RuntimeError(message)
        choices = obj.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            if sequence and sequence[-1][0] == "text":
                sequence[-1] = ("text", sequence[-1][1] + text)
            else:
                sequence.append(("text", text))
        for call in delta.get("tool_calls") or []:
            index = int(call.get("index", 0))
            if index not in tool_by_index:
                tool_by_index[index] = {"id": call.get("id"), "name": "", "arguments": ""}
                sequence.append(("tool", index))
            item = tool_by_index[index]
            if call.get("id"):
                item["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                item["name"] += str(function["name"])
            if function.get("arguments"):
                value = function["arguments"]
                item["arguments"] += value if isinstance(value, str) else json.dumps(value)
        finish = choice.get("finish_reason")
        if finish:
            last_stop_reason = str(finish)
    return [(kind, tool_by_index[value] if kind == "tool" else value) for kind, value in sequence], last_stop_reason


async def _stream_tools(internal_request: dict[str, Any], bridge: Any, model: str, input_tokens: int):
    message_id = f"msg_{uuid.uuid4().hex}"
    yield _message_start(message_id, model, input_tokens)
    task: Optional[asyncio.Task] = None
    try:
        task = asyncio.create_task(tool_bridge.handle_chat(internal_request, bridge=bridge))
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(task), timeout=PING_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                yield _sse_event("ping", {"type": "ping"})

        if isinstance(response, JSONResponse):
            yield _stream_error(_error_message(response))
            return
        if not isinstance(response, StreamingResponse):
            converted = _convert_tool_completion(response, model, input_tokens)
            if isinstance(converted, JSONResponse):
                yield _stream_error(_error_message(converted))
                return
            for index, block in enumerate(converted.get("content", [])):
                if block.get("type") == "text":
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    })
                    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
                else:
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
                    })
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)},
                    })
                    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
            stop_reason = converted.get("stop_reason", "end_turn")
            output_tokens = converted.get("usage", {}).get("output_tokens", 0)
            yield _message_delta(stop_reason, output_tokens)
            yield _message_stop()
            return

        raw_chunks: list[Any] = []
        async for raw in response.body_iterator:
            raw_chunks.append(raw)
        objects: list[dict[str, Any]] = []
        for raw in raw_chunks:
            objects.extend(_parse_openai_sse(raw))
        sequence, finish_reason = _completion_blocks_from_stream(objects)

        block_index = 0
        output_text: list[str] = []
        output_json: list[str] = []
        for kind, value in sequence:
            if kind == "text":
                yield _sse_event("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "text_delta", "text": value},
                })
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
                output_text.append(value)
                block_index += 1
                continue

            name = str(value.get("name") or "")
            raw_arguments = value.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                raise RuntimeError(f"invalid JSON arguments for tool {name!r}")
            if not name or not isinstance(arguments, dict):
                raise RuntimeError("bridge returned an invalid tool call")
            tool_id = f"toolu_{uuid.uuid4().hex[:20]}"
            partial_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            yield _sse_event("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            })
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_index})
            output_json.append(partial_json)
            block_index += 1

        stop_reason = _anthropic_stop_reason(
            finish_reason,
            has_tool_calls=any(kind == "tool" for kind, _ in sequence),
        )
        yield _message_delta(stop_reason, _estimate_tokens("".join(output_text + output_json)))
        yield _message_stop()
    except asyncio.CancelledError:
        if task is not None and not task.done():
            task.cancel()
        raise
    except Exception as exc:
        yield _stream_error(str(exc))
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


async def handle_messages(request: Any, bridge: Any, default_model: str) -> Any:
    """Handle an Anthropic Messages request for either webchat bridge."""
    tool_prompt = ""
    try:
        normalized_messages = normalize_messages(request)
        prompt = _make_prompt(request, normalized_messages)
        raw_tools = _get(request, "tools", None)
        openai_tools = normalize_tools(raw_tools)
        raw_tool_choice = _get(request, "tool_choice", None)
        disable_parallel = _disable_parallel_tool_use(raw_tool_choice)
        tool_choice = normalize_tool_choice(raw_tool_choice)
        model = str(_get(request, "model", None) or default_model)
        stream = bool(_get(request, "stream", False))
        if not openai_tools and (
            tool_choice == "required" or isinstance(tool_choice, dict)
        ):
            raise ValueError("tool_choice requires a callable tool, but no tools were provided")
        if tool_choice != "none" and openai_tools:
            tool_prompt = tool_bridge.serialize_tools(openai_tools)
            if len(prompt) + len(tool_prompt) > MAX_PROMPT_CHARS:
                raise ValueError("messages and tool schemas are too large")
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception as exc:
        return _error_response(f"invalid Anthropic request: {exc}", 400)

    use_tools = bool(openai_tools) and tool_choice != "none"
    counted_prompt = prompt + ("\n\n" + tool_prompt if use_tools else "")
    input_tokens = _estimate_tokens(counted_prompt)

    if use_tools:
        internal_request = {
            "model": model,
            "messages": normalized_messages,
            "tools": openai_tools,
            "tool_choice": tool_choice,
            "stream": stream,
        }
        if disable_parallel:
            internal_request["parallel_tool_calls"] = False
        if stream:
            return StreamingResponse(
                _stream_tools(internal_request, bridge, model, input_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = await tool_bridge.handle_chat(internal_request, bridge=bridge)
        return _convert_tool_completion(result, model, input_tokens)

    if stream:
        return StreamingResponse(
            _stream_plain(bridge, prompt, model, input_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text_parts: list[str] = []
    try:
        async for chunk in bridge.stream_response(prompt):
            parsed = bridge._parse_sse(chunk) if hasattr(bridge, "_parse_sse") else None
            if not parsed:
                continue
            if parsed.get("error"):
                return _error_response(parsed["error"], 502)
            if parsed.get("content"):
                text_parts.append(parsed["content"])
    except Exception as exc:
        return _error_response(str(exc), 502)

    text = "".join(text_parts)
    content = [{"type": "text", "text": text}] if text else []
    return _anthropic_message(
        model=model,
        content=content,
        input_tokens=input_tokens,
        output_tokens=_estimate_tokens(text),
    )


def count_tokens(request: Any) -> dict[str, int]:
    """Best-effort token estimate for the optional Anthropic count endpoint."""
    normalized = normalize_messages(request)
    prompt = _make_prompt(request, normalized)
    tools = normalize_tools(_get(request, "tools", None))
    if tools:
        prompt += "\n\n" + tool_bridge.serialize_tools(tools)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("messages and tool schemas are too large")
    return {"input_tokens": _estimate_tokens(prompt)}


def handle_count_tokens(request: Any) -> Any:
    try:
        return count_tokens(request)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception as exc:
        return _error_response(f"invalid token-count request: {exc}", 400)
````

## `requirements.txt`

````text
fastapi
uvicorn[standard]
playwright
pydantic
httpx
jsonschema>=4.23,<5
````

## `run_both.py`

````python
"""Start both browser-backed bridge APIs from one console.

DeepSeek listens on 127.0.0.1:8000 and Qwen on 127.0.0.1:8001. The
supervisor collects the shared prompt mode and each backend's own RPM choice,
then passes the settings to child processes so prompts never race on stdin.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from startup_config import (
    choose_max_pending,
    choose_prompt_mode,
    choose_rpm,
    environment_for_child,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def start_server(name: str, script: str, port_env: str, port: int, env: dict[str, str]):
    child_env = dict(env)
    child_env[port_env] = str(port)
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=BASE,
        env=child_env,
        creationflags=flags,
    )
    print(f"[run_both] started {name}: {script} on port {port} (pid {proc.pid})")
    return proc


def _stop_processes(names: list[str], procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for name, proc in zip(names, procs):
        try:
            proc.wait(timeout=5)
            print(f"[run_both] {name} stopped.")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            print(f"[run_both] {name} killed after shutdown timeout.")


def main() -> None:
    prompt_mode = choose_prompt_mode()
    deepseek_rpm = choose_rpm("deepseek")
    qwen_rpm = choose_rpm("qwen")
    deepseek_pending = choose_max_pending("deepseek")
    qwen_pending = choose_max_pending("qwen")

    deepseek_env = environment_for_child(
        "deepseek", prompt_mode, deepseek_rpm, deepseek_pending
    )
    qwen_env = environment_for_child("qwen", prompt_mode, qwen_rpm, qwen_pending)

    print("\nStarting both bridges with independent provider queues:")
    print(f"  Prompt mode: {prompt_mode}")
    print(f"  DeepSeek: {deepseek_rpm} provider requests/min, {deepseek_pending} pending slots")
    print(f"  Qwen:     {qwen_rpm} provider requests/min, {qwen_pending} pending slots")

    names: list[str] = []
    procs: list[subprocess.Popen] = []
    try:
        deepseek = start_server(
            "DeepSeek", "server.py", "DEEPSEEK_PORT", 8000, deepseek_env
        )
        names.append("DeepSeek")
        procs.append(deepseek)
        qwen = start_server("Qwen", "qwen_server.py", "QWEN_PORT", 8001, qwen_env)
        names.append("Qwen")
        procs.append(qwen)

        print("\nOpenAI-compatible clients:")
        print("  DeepSeek: http://127.0.0.1:8000/v1  (model deepseek-chat)")
        print("  Qwen:     http://127.0.0.1:8001/v1  (model qwen-auto)")
        print("Claude Code / Anthropic Messages base URL (choose one backend per session):")
        print("  DeepSeek: http://127.0.0.1:8000")
        print("  Qwen:     http://127.0.0.1:8001")
        print("Queue stats: GET /queue/status on either server.")
        print("Both services remain loopback-only. Press Ctrl+C here to stop both.\n")

        while True:
            exited = [(name, proc.returncode) for name, proc in zip(names, procs)
                      if proc.poll() is not None]
            if exited:
                for name, code in exited:
                    print(f"[run_both] {name} exited with code {code}; stopping the other bridge.")
                _stop_processes(names, procs)
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run_both] shutting down both servers...")
        _stop_processes(names, procs)
    except Exception:
        _stop_processes(names, procs)
        raise


if __name__ == "__main__":
    main()
````

## `test_parser.py`

````python
"""
test_parser.py – Unified parser tests for BOTH bridges.

Tests the DeepSeek parser (app.py) and the Qwen parser (qwen_app.py) using their
real stream payload shapes. No browser needed (playwright is mocked).

DeepSeek scenarios:
  1. BATCH ops + relative paths + FINISHED status
  2. inline first fragment in opening payload
  3. status ops never become text
  4. direct {"v": "text"} payloads
  5. SEARCH fragments dropped

Qwen scenarios:
  1. thinking_summary (summary_title/thought) + answer + finished
  2. web_search / tool_calls ignored
  3. repeated summary deduped (incremental emission)
  4. error payload surfaced + stops processing
  5. SEND_THINKING=False drops thinking

Run:  python3 test_parser.py
"""
import asyncio
import json
import sys
import types

# ---- mock playwright so both app modules can be imported without a browser ----
fake_module = types.ModuleType("playwright")
fake_async_api = types.ModuleType("playwright.async_api")
fake_async_api.async_playwright = lambda: None
fake_async_api.TimeoutError = TimeoutError
sys.modules["playwright"] = fake_module
sys.modules["playwright.async_api"] = fake_async_api

import app        # DeepSeek bridge
import qwen_app   # Qwen bridge

# =====================================================================
# DeepSeek tests
# =====================================================================
def run_deepseek(payloads):
    app._fragment_types = []
    app._active_fragment_type = None
    queue = asyncio.Queue()
    for payload in payloads:
        app._handle_payload(payload, queue)
    content, reasoning = [], []
    while not queue.empty():
        item = queue.get_nowait()
        delta = json.loads(item[6:].strip())["choices"][0]["delta"]
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])
    return "".join(content), "".join(reasoning)


print("========== DEEPSEEK PARSER ==========")

# Scenario 1: BATCH ops (the common case)
s1 = [
    {"p": "response", "o": "BATCH", "v": [
        {"p": "fragments", "o": "APPEND", "v": [{"type": "THINK", "content": ""}]},
        {"p": "fragments/0/content", "o": "ADD", "v": "The user just said hi. "},
    ]},
    {"p": "response", "o": "BATCH", "v": [
        {"p": "fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": ""}]},
        {"p": "fragments/1/content", "o": "ADD", "v": "Hello! How can I help you today? "},
    ]},
    {"p": "response/status", "o": "SET", "v": "FINISHED"},
]
content, reasoning = run_deepseek(s1)
assert reasoning == "The user just said hi. ", f"THINK leaked: {reasoning!r}"
assert content == "Hello! How can I help you today? ", f"ANSWER wrong: {content!r}"
assert "FINISHED" not in content and "FINISHED" not in reasoning, "FINISHED leaked!"
print("✓ DeepSeek 1 (BATCH ops + relative paths + FINISHED status): OK")

# Scenario 2: inline first fragment in opening payload
s2 = [
    {"v": {"response": {"fragments": [{"type": "THINK", "content": "Let me reason."}]}}},
    {"p": "response/fragments/0/content", "o": "ADD", "v": "More thinking "},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": ""}]},
    {"p": "response/fragments/1/content", "o": "ADD", "v": "The answer."},
    {"p": "response/status", "o": "SET", "v": "FINISHED"},
]
content, reasoning = run_deepseek(s2)
assert reasoning == "Let me reason.More thinking ", f"THINK wrong: {reasoning!r}"
assert content == "The answer.", f"ANSWER wrong: {content!r}"
print("✓ DeepSeek 2 (inline first fragment + response/ prefixed paths): OK")

# Scenario 3: status ops must NEVER become text
s3 = [
    {"p": "response/status", "o": "SET", "v": "FINISHED"},
    {"p": "status", "o": "SET", "v": "CONTENT_FILTER"},
    {"p": "response/fragments/0/status", "o": "SET", "v": "FINISHED"},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "ok"}]},
]
content, reasoning = run_deepseek(s3)
assert content == "ok", f"content wrong: {content!r}"
assert "FINISHED" not in content and "CONTENT_FILTER" not in content, "status leaked!"
print("✓ DeepSeek 3 (statuses never leak): OK")

# Scenario 4: direct {"v": "text"} with active fragment
s4 = [
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": ""}]},
    {"v": "direct think text"},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": ""}]},
    {"v": "direct answer text"},
]
content, reasoning = run_deepseek(s4)
assert reasoning == "direct think text", f"THINK wrong: {reasoning!r}"
assert content == "direct answer text", f"ANSWER wrong: {content!r}"
print("✓ DeepSeek 4 (direct v-string payloads): OK")

# Scenario 5: SEARCH fragments dropped
s5 = [
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "SEARCH", "content": "search noise"}]},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "final"}]},
]
content, reasoning = run_deepseek(s5)
assert content == "final" and "search noise" not in content, f"SEARCH leaked: {content!r}"
print("✓ DeepSeek 5 (SEARCH fragments dropped): OK")


# =====================================================================
# Qwen tests
# =====================================================================
def run_qwen(payloads):
    qwen_app._thinking_emitted = ""
    qwen_app._stream_error = False
    queue = asyncio.Queue()
    for payload in payloads:
        qwen_app._handle_payload(payload, queue)
    content, reasoning, finishes = [], [], []
    while not queue.empty():
        item = queue.get_nowait()
        obj = json.loads(item[6:].strip())
        if "error" in obj:
            content.append("ERROR:" + obj["error"]["message"])
            continue
        delta = obj["choices"][0]["delta"]
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])
        if obj["choices"][0].get("finish_reason") == "stop":
            finishes.append(True)
    return "".join(content), "".join(reasoning), len(finishes)


print("\n========== QWEN PARSER ==========")

# Qwen 1: normal flow — thinking summary, then answer, then finished
q1 = [
    {"choices": [{"delta": {"phase": "thinking_summary", "extra": {
        "summary_title": ["Approach"],
        "summary_thought": ["Parse the user's greeting and reply warmly."]}}}]},
    {"choices": [{"delta": {"phase": "answer", "content": "Hello! "}}]},
    {"choices": [{"delta": {"phase": "answer", "content": "How can I help you today? "}}]},
    {"choices": [{"delta": {"phase": "answer", "status": "finished"}}]},
]
content, reasoning, finishes = run_qwen(q1)
assert reasoning == "Approach\nParse the user's greeting and reply warmly.", f"THINK wrong: {reasoning!r}"
assert content == "Hello! How can I help you today? ", f"ANSWER wrong: {content!r}"
assert finishes == 1, f"finish wrong: {finishes}"
print("✓ Qwen 1 (thinking_summary + answer + finished): OK")

# Qwen 2: web_search noise and tool_calls are ignored
q2 = [
    {"choices": [{"delta": {"phase": "web_search", "content": "search noise"}}]},
    {"choices": [{"delta": {"role": "function", "name": "search", "content": "tool"}}]},
    {"choices": [{"delta": {"phase": "answer", "content": "Clean answer."}}]},
    {"choices": [{"delta": {"phase": "answer", "status": "finished"}}]},
]
content, reasoning, finishes = run_qwen(q2)
assert content == "Clean answer.", f"noise leaked: {content!r}"
assert "search noise" not in content and "tool" != content
print("✓ Qwen 2 (web_search + tool_calls ignored): OK")

# Qwen 3: repeated full summary (incremental suffix emission)
q3 = [
    {"choices": [{"delta": {"phase": "thinking_summary", "extra": {
        "summary_title": ["Step 1"], "summary_thought": ["First idea."]}}}]},
    {"choices": [{"delta": {"phase": "thinking_summary", "extra": {
        "summary_title": ["Step 1"], "summary_thought": ["First idea."]}}}]},
    {"choices": [{"delta": {"phase": "answer", "content": "Final."}}]},
    {"choices": [{"delta": {"phase": "answer", "status": "finished"}}]},
]
content, reasoning, finishes = run_qwen(q3)
assert reasoning == "Step 1\nFirst idea.", f"duplicate summary not deduped: {reasoning!r}"
print("✓ Qwen 3 (repeated summary deduped): OK")

# Qwen 4: provider error payload + stops processing after
q4 = [
    {"error": {"message": "data_inspection_failed: censored"}},
    {"choices": [{"delta": {"phase": "answer", "content": "never"}}]},
]
content, reasoning, finishes = run_qwen(q4)
assert content == "ERROR:data_inspection_failed: censored", f"error wrong: {content!r}"
print("✓ Qwen 4 (error payload surfaced + stops): OK")

# Qwen 5: SEND_THINKING=False drops thinking
old = qwen_app.SEND_THINKING
qwen_app.SEND_THINKING = False
try:
    q5 = [
        {"choices": [{"delta": {"phase": "thinking_summary", "extra": {
            "summary_title": ["T"], "summary_thought": ["hidden"]}}}]},
        {"choices": [{"delta": {"phase": "answer", "content": "Visible."}}]},
        {"choices": [{"delta": {"phase": "answer", "status": "finished"}}]},
    ]
    content, reasoning, finishes = run_qwen(q5)
    assert reasoning == "" and content == "Visible.", f"thinking not dropped: {reasoning!r}"
    print("✓ Qwen 5 (SEND_THINKING=False drops thinking): OK")
finally:
    qwen_app.SEND_THINKING = old

print("\nALL PARSER TESTS PASSED (DeepSeek + Qwen) ✅")
````

## `test_tool_bridge.py`

````python
"""Focused tests for the shared OpenAI/Anthropic tool-call bridge.

These tests use fake provider streams only; no browser, account, or live model
is used. Run with `python test_tool_bridge.py`.
"""
from __future__ import annotations

import json
import unittest

from fastapi.responses import JSONResponse, StreamingResponse

import tool_bridge


START = tool_bridge.DEEPSEEK_TOOL_CALL_START
END = tool_bridge.DEEPSEEK_TOOL_CALL_END
SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def tool_call(name: str, args: dict) -> str:
    return f'<tool_call><name>{name}</name><args>{json.dumps(args)}</args></tool_call>'


class FakeBridge:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    async def stream_response(self, prompt: str):
        self.prompts.append(prompt)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        payload = {
            "id": "fake-provider-id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    @staticmethod
    def _parse_sse(raw: str) -> dict | None:
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    return None
                choices = payload.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    return {
                        "content": delta.get("content", ""),
                        "reasoning": delta.get("reasoning_content", ""),
                    }
        return None


class ToolParserTests(unittest.TestCase):
    def test_native_deepseek_marker_split_across_chunks(self):
        start = START
        end = END
        raw = json.dumps([{"name": "search", "arguments": {"query": "café"}}], ensure_ascii=False)
        sieve = tool_bridge.Sieve()
        events = []
        chunks = [
            "before " + start[:9],
            start[9:] + raw + end[:8],
            end[8:] + " after",
        ]
        for chunk in chunks:
            events.extend(sieve.feed(chunk))
        events.extend(sieve.finish())
        self.assertEqual(events[0], ("content", "before "))
        self.assertEqual(events[1], ("tool_block", "json", raw))
        self.assertEqual(events[2], ("content", " after"))
        self.assertEqual(
            tool_bridge.parse_tool_blocks(events[1][1], events[1][2]),
            [{"name": "search", "arguments": {"query": "café"}}],
        )

    def test_deepseek_dsml_calls_v41_and_double_bar_rendering(self):
        raw = '''<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="Bash">
<｜｜DSML｜｜ parameter name="command" string="true">cd "C:/Users/Agent_Test/.hyperframes-studio/Create a hello world animation" &amp;&amp; hyperframes catalog --query "title animation" 2&gt;&amp;1 | head -50</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="description" string="true">Search the block registry for title animations</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="limit" string="false">5</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="Bash">
<｜｜DSML｜｜ parameter name="command" string="true">cat "C:/Users/Agent_Test/AppData/Roaming/HyperFrames/guides/studio-plugin/skills/hyperframes/references/[format.md](http://format.md)"</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="description" string="true">Read the format reference</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''
        allowed = {"Bash": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }}
        for token in ("｜｜DSML｜｜", "｜DSML｜"):
            with self.subTest(token=token):
                rendered = raw.replace("｜｜DSML｜｜", token)
                _, outcome = tool_bridge._events_to_chunks([("content", rendered)], "test", allowed)
                self.assertEqual(outcome[0], "parsed")
                self.assertEqual(len(outcome[1]), 2)
                self.assertEqual(outcome[1][0]["name"], "Bash")
                self.assertIn(" && ", outcome[1][0]["arguments"]["command"])
                self.assertIn("2>&1", outcome[1][0]["arguments"]["command"])
                self.assertEqual(outcome[1][0]["arguments"]["limit"], 5)
                self.assertEqual(outcome[1][1]["arguments"]["description"], "Read the format reference")

    def test_native_marker_ascii_and_generic_aliases(self):
        pairs = [
            ("<|tool▁calls▁begin|>", "<|tool▁calls▁end|>"),
            ("<｜tool_calls_begin｜>", "<｜tool_calls_end｜>"),
            ("<|tool_calls_begin|>", "<|tool_calls_end|>"),
            ("<tool_calls>", "</tool_calls>"),
        ]
        payload = '[{"name":"search","arguments":{"query":"x"}}]'
        for start, end in pairs:
            with self.subTest(start=start):
                sieve = tool_bridge.Sieve()
                events = sieve.feed(start + payload + end) + sieve.finish()
                block = next(event for event in events if event[0] == "tool_block")
                self.assertEqual(tool_bridge.parse_tool_blocks(block[1], block[2])[0]["name"], "search")

    def test_multiple_tagged_calls_are_not_dropped(self):
        sieve = tool_bridge.Sieve()
        raw = tool_call("search", {"query": "one"}) + "\n" + tool_call("search", {"query": "two"})
        events = sieve.feed(raw) + sieve.finish()
        blocks = [event for event in events if event[0] == "tool_block"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            [tool_bridge.parse_tool_blocks(kind, body)[0]["arguments"]["query"] for _, kind, body in blocks],
            ["one", "two"],
        )

    def test_json_repair_and_openai_envelope(self):
        relaxed = "[{name: 'search', arguments: {query: 'weather', limit: 3,},}]"
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", relaxed),
            [{"name": "search", "arguments": {"query": "weather", "limit": 3}}],
        )
        envelope = json.dumps({"tool_calls": [{
            "id": "call_x",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query":"wind"}'},
        }]})
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", envelope),
            [{"name": "search", "arguments": {"query": "wind"}}],
        )

    def test_bare_json_is_not_extracted_from_explanatory_prose(self):
        prose = 'Example only:\n```json\n{"name":"search","arguments":{"query":"demo"}}\n```'
        self.assertEqual(tool_bridge.parse_tool_blocks("json", prose), [])
        fenced = '```json\n{"name":"search","arguments":{"query":"demo"}}\n```'
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", fenced),
            [{"name": "search", "arguments": {"query": "demo"}}],
        )

    def test_run_code_nested_args_templates_and_string_false_positive(self):
        block = '''<run_code>
const html = `<h1>${title}</h1>`;
await tools.render({html, options: {limit: 3, enabled: true}, labels: ["brace }", 'single'],});
const example = "tools.fake({evil: true})"; // tools.nope({x: 1})
</run_code>'''
        calls = tool_bridge.parse_tool_blocks("run_code", block)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "render")
        self.assertEqual(calls[0]["arguments"], {
            "html": "<h1>${title}</h1>",
            "options": {"limit": 3, "enabled": True},
            "labels": ["brace }", "single"],
        })

    def test_invoke_parameter_blocks(self):
        block = '<invoke name="search"><parameter name="query">weather</parameter><parameter name="limit">3</parameter></invoke>'
        parsed = tool_bridge.parse_tool_blocks("tool_call", block)
        self.assertEqual(parsed, [{"name": "search", "arguments": {"query": "weather", "limit": 3}}])

    def test_external_schema_refs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "external JSON Schema references"):
            tool_bridge._validate_local_schema_refs({"$ref": "https://example.invalid/schema.json"})
        tool_bridge._validate_local_schema_refs({"$ref": "#/$defs/local"})

    def test_schema_and_allowed_tool_validation(self):
        events = [("content", START + json.dumps([
            {"name": "search", "arguments": {"query": "ok", "limit": 2}}
        ]) + END)]
        chunks, outcome = tool_bridge._events_to_chunks(events, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "parsed")
        self.assertEqual(outcome[1][0]["arguments"]["limit"], 2)

        unknown = [("content", START + json.dumps([
            {"name": "shell_exec", "arguments": {"query": "bad"}}
        ]) + END)]
        _, outcome = tool_bridge._events_to_chunks(unknown, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "unparseable")

        invalid = [("content", tool_call("search", {"query": 123}))]
        _, outcome = tool_bridge._events_to_chunks(invalid, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "unparseable")


class ToolApiBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_has_stable_id_role_and_done(self):
        call_text = START + json.dumps([
            {"name": "search", "arguments": {"query": "hello"}}
        ]) + END
        bridge = FakeBridge([call_text])
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "find something"}],
            "tools": [{"type": "function", "function": {
                "name": "search", "parameters": SCHEMA,
            }}],
            "tool_choice": "auto",
            "stream": True,
        }
        response = await tool_bridge.handle_chat(request, bridge)
        self.assertIsInstance(response, StreamingResponse)
        body = "".join([part async for part in response.body_iterator])
        data_lines = [line for line in body.splitlines() if line.startswith("data:")]
        objects = [json.loads(line[5:].strip()) for line in data_lines if line[5:].strip() != "[DONE]"]
        self.assertEqual(body.count("data: [DONE]"), 1)
        self.assertEqual(objects[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertTrue(objects[-1]["choices"][0]["finish_reason"] == "tool_calls")
        self.assertEqual(len({obj["id"] for obj in objects}), 1)
        self.assertEqual(len({obj["created"] for obj in objects}), 1)
        call_deltas = [obj["choices"][0]["delta"].get("tool_calls") for obj in objects]
        self.assertTrue(any(call_deltas))

    async def test_required_choice_without_tools_returns_bad_request(self):
        response = await tool_bridge.handle_chat({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
            "tool_choice": "required",
        }, FakeBridge(["unused"]))
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("param", body["error"])
        self.assertIn("code", body["error"])

    async def test_parallel_tool_calls_false_is_enforced(self):
        response_text = tool_call("search", {"query": "one"}) + tool_call("search", {"query": "two"})
        bridge = FakeBridge([response_text])
        response = await tool_bridge.handle_chat({
            "messages": [{"role": "user", "content": "search two things"}],
            "tools": [{"type": "function", "function": {
                "name": "search", "parameters": SCHEMA,
            }}],
            "parallel_tool_calls": False,
        }, bridge)
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(bridge.prompts), tool_bridge.MAX_REPAIR_RETRIES + 1)
        self.assertIn("at most one", bridge.prompts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `test_anthropic_compat.py`

````python
"""Offline tests for the minimal Anthropic Messages compatibility adapter."""
from __future__ import annotations

import asyncio
import json
import unittest

import anthropic_compat as ac
import tool_bridge


class FakeBridge:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    async def stream_response(self, prompt: str):
        self.prompts.append(prompt)
        for reply in self.replies:
            yield self._make_openai_sse(reply)

    @staticmethod
    def _make_openai_sse(text: str) -> str:
        obj = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        return "data: " + json.dumps(obj) + "\n\n"

    @staticmethod
    def _parse_sse(raw: str):
        try:
            obj = json.loads(raw[6:].strip())
            delta = obj["choices"][0]["delta"]
            return {
                "content": delta.get("content", "") or "",
                "reasoning": delta.get("reasoning_content", "") or "",
            }
        except Exception:
            return None


def parse_anthropic_events(raw_chunks):
    events = []
    for raw in raw_chunks:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event_name = None
        data = None
        for line in raw.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if data:
            events.append((event_name, json.loads(data)))
    return events


async def read_response_body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


class AnthropicCompatTests(unittest.TestCase):
    def test_normalize_messages_and_tool_results(self):
        request = ac.AnthropicMessagesRequest(
            model="claude-test",
            max_tokens=128,
            system=[{"type": "text", "text": "Use tools carefully."}],
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file text"}],
                },
            ],
        )
        normalized = ac.normalize_messages(request)
        self.assertEqual(normalized[0]["role"], "system")
        self.assertEqual(normalized[1]["tool_calls"][0]["function"]["name"], "Read")
        self.assertIn("TOOL RESULT (Read)", normalized[2]["content"])
        self.assertIn("file text", normalized[2]["content"])

    def test_normalize_anthropic_tools_and_choices(self):
        tools = ac.normalize_tools([{
            "name": "Read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }])
        self.assertEqual(tools[0]["function"]["name"], "Read")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["path"])
        self.assertEqual(ac.normalize_tool_choice({"type": "any"}), "required")
        self.assertEqual(ac.normalize_tool_choice({"type": "tool", "name": "Read"})["function"]["name"], "Read")
        self.assertTrue(ac._disable_parallel_tool_use({"type": "auto", "disable_parallel_tool_use": True}))
        self.assertEqual(ac._anthropic_stop_reason("length"), "max_tokens")

    def test_server_side_anthropic_tools_are_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "server-side Anthropic tools are unsupported"):
            ac.normalize_tools([{"type": "web_search_20250305", "name": "web_search"}])

    def test_non_stream_text_response(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["Hello"]), "deepseek-chat"))
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["model"], "web-alias")
        self.assertEqual(response["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertIn("input_tokens", response["usage"])

    def test_stream_text_has_anthropic_event_sequence(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64, stream=True,
            messages=[{"role": "user", "content": "Hi"}],
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["Hello"]), "deepseek-chat"))
        raw = asyncio.run(read_response_body(response))
        events = parse_anthropic_events(raw)
        kinds = [payload["type"] for _, payload in events]
        self.assertEqual(kinds, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ])
        self.assertEqual(events[2][1]["delta"]["text"], "Hello")

    def test_tool_use_round_trip_and_schema_validation(self):
        call = tool_bridge.DEEPSEEK_TOOL_CALL_START + json.dumps([
            {"name": "Read", "arguments": {"path": "a.txt"}}
        ]) + tool_bridge.DEEPSEEK_TOOL_CALL_END
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "Read a.txt"}],
            tools=[{
                "name": "Read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
            }],
            tool_choice={"type": "tool", "name": "Read"},
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge([call]), "deepseek-chat"))
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["content"][0]["type"], "tool_use")
        self.assertEqual(response["content"][0]["name"], "Read")
        self.assertEqual(response["content"][0]["input"], {"path": "a.txt"})

    def test_anthropic_disable_parallel_tool_use_is_enforced(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "Search two things"}],
            tools=[{"name": "Search", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        two_calls = (
            '<tool_call><name>Search</name><args>{"query":"one"}</args></tool_call>'
            '<tool_call><name>Search</name><args>{"query":"two"}</args></tool_call>'
        )
        bridge = FakeBridge([two_calls])
        response = asyncio.run(ac.handle_messages(request, bridge, "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertGreaterEqual(response.status_code, 400)
        self.assertIn("at most one tool", bridge.prompts[0])

    def test_invalid_tool_schema_is_rejected(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64,
            messages=[{"role": "user", "content": "use the tool"}],
            tools=[{"name": "Broken", "input_schema": {"type": 7}}],
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["unused"]), "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertEqual(response.status_code, 400)

    def test_unknown_tool_is_never_returned(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "run something"}],
            tools=[{"name": "Read", "input_schema": {"type": "object"}}],
        )
        bad_call = '<tool_call><name>ShellExec</name><args>{"cmd":"rm -rf /"}</args></tool_call>'
        response = asyncio.run(ac.handle_messages(request, FakeBridge([bad_call]), "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertGreaterEqual(response.status_code, 400)
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(body["type"], "error")

    def test_stream_tool_use_has_anthropic_events(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128, stream=True,
            messages=[{"role": "user", "content": "Read a.txt"}],
            tools=[{
                "name": "Read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }],
            tool_choice={"type": "tool", "name": "Read"},
        )
        call = '<tool_call><name>Read</name><args>{"path":"a.txt"}</args></tool_call>'
        response = asyncio.run(ac.handle_messages(request, FakeBridge([call]), "deepseek-chat"))
        raw = asyncio.run(read_response_body(response))
        events = parse_anthropic_events(raw)
        kinds = [payload["type"] for _, payload in events]
        self.assertEqual(kinds, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ])
        self.assertEqual(events[2][1]["delta"]["type"], "input_json_delta")
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")

    def test_bare_json_tool_call_is_request_scoped(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
        chunks, outcome = tool_bridge._events_to_chunks(
            [("content", '{"name":"Read","arguments":{"path":"a.txt"}}')],
            "test-model",
            {"Read": schema},
        )
        self.assertEqual(outcome[0], "parsed")
        self.assertEqual(outcome[1][0]["name"], "Read")
        unknown_chunks, unknown_outcome = tool_bridge._events_to_chunks(
            [("content", '{"name":"ShellExec","arguments":{"cmd":"x"}}')],
            "test-model",
            {"Read": schema},
        )
        self.assertEqual(unknown_outcome[0], "unparseable")

    def test_sieve_handles_split_and_multiple_tool_blocks(self):
        sieve = tool_bridge.Sieve()
        parsed = sieve.feed("prefix <tool_")
        parsed += sieve.feed('call><name>Read</name><args>{}</args></tool_call>')
        parsed += sieve.feed('<tool_call><name>Write</name><args>{}</args></tool_call> tail')
        parsed += sieve.finish()
        calls = [item for item in parsed if item[0] == "tool_block"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "tool_call")
        self.assertIn("prefix ", [item[1] for item in parsed if item[0] == "content"])

    def test_count_tokens_is_available(self):
        request = ac.AnthropicCountTokensRequest(
            model="web-alias", messages=[{"role": "user", "content": "A short prompt."}]
        )
        result = ac.handle_count_tokens(request)
        self.assertGreater(result["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `test_api_smoke.py`

````python
"""ASGI route smoke tests for both APIs using a fake webchat stream.

Run with `python test_api_smoke.py` after installing requirements. No browser,
provider account, or live web session is used.
"""
from __future__ import annotations

import json
import os
import unittest

import httpx

os.environ.setdefault("PROMPT_MODE", "full")
os.environ.setdefault("DEEPSEEK_RPM", "5")
os.environ.setdefault("QWEN_RPM", "5")
import server  # noqa: E402
import qwen_server  # noqa: E402


TOOL_REPLY = '<tool_call><name>Read</name><args>{"path":"a.txt"}</args></tool_call>'
OPENAI_TOOL_REPLY = '<tool_call><name>read</name><args>{"path":"a.txt"}</args></tool_call>'
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def _openai_sse(text: str) -> str:
    payload = {
        "id": "fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-web-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return "data: " + json.dumps(payload) + "\n\n"


async def _fake_stream_response(prompt: str):
    if "Tool name: Read" in prompt:
        yield _openai_sse(TOOL_REPLY)
    elif "Tool name: read" in prompt:
        yield _openai_sse(OPENAI_TOOL_REPLY)
    else:
        yield _openai_sse("fake reply")


async def _fake_error_stream(prompt: str):
    yield 'data: {"error":{"message":"synthetic provider failure","type":"internal_error"}}\n\n'


class ApiSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.servers = [
            (server.api, server.app, "deepseek-chat"),
            (qwen_server.api, qwen_server.app, "qwen-auto"),
        ]
        self.original_streams = []
        for _, bridge, _ in self.servers:
            self.original_streams.append(bridge.stream_response)
            bridge.stream_response = _fake_stream_response

    async def asyncTearDown(self):
        for (_, bridge, _), original in zip(self.servers, self.original_streams):
            bridge.stream_response = original

    async def test_messages_openai_and_tool_routes_for_both_servers(self):
        for api, bridge, backend_model in self.servers:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api),
                base_url="http://test",
            ) as client:
                response = await client.head("/api/hello")
                self.assertEqual(response.status_code, 200)

                messages = [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Hello"},
                ]
                response = await client.post("/v1/messages", json={
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["content"][0]["text"], "fake reply")
                self.assertEqual(response.json()["model"], "test-alias")

                response = await client.post("/v1/messages/count_tokens", json={
                    "model": "test-alias",
                    "messages": messages,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertGreater(response.json()["input_tokens"], 0)

                response = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "stream": False,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "fake reply")

                queue_status = await client.get("/queue/status")
                self.assertEqual(queue_status.status_code, 200, queue_status.text)
                self.assertEqual(queue_status.json()["pending"], 0)
                self.assertEqual(queue_status.json()["requests_per_minute"], 5)

                invalid_content = await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
                    ]}],
                })
                self.assertEqual(invalid_content.status_code, 400, invalid_content.text)

                invalid_body = await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "safe"}],
                    "parallel_tool_calls": "sensitive-validation-marker",
                })
                self.assertEqual(invalid_body.status_code, 422, invalid_body.text)
                self.assertNotIn("sensitive-validation-marker", invalid_body.text)

                bridge.stream_response = _fake_error_stream
                provider_error = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "stream": False,
                })
                self.assertEqual(provider_error.status_code, 502, provider_error.text)
                self.assertEqual(provider_error.json()["error"]["code"], "provider_error")

                bridge.stream_response = _fake_stream_response
                streaming = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "stream": True,
                })
                self.assertEqual(streaming.status_code, 200, streaming.text)
                chunks = [
                    json.loads(line[6:]) for line in streaming.text.splitlines()
                    if line.startswith("data: {")
                ]
                self.assertGreaterEqual(len(chunks), 3)
                self.assertEqual(len({chunk["id"] for chunk in chunks}), 1)
                self.assertIn("data: [DONE]", streaming.text)

                response = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "tools": [],
                    "tool_choice": "required",
                })
                self.assertEqual(response.status_code, 400, response.text)

                anthropic_tool = {
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                    "tools": [{"name": "Read", "input_schema": TOOL_SCHEMA}],
                    "tool_choice": {"type": "tool", "name": "Read"},
                }
                response = await client.post("/v1/messages", json=anthropic_tool)
                self.assertEqual(response.status_code, 200, response.text)
                block = response.json()["content"][0]
                self.assertEqual((block["type"], block["name"], block["input"]),
                                 ("tool_use", "Read", {"path": "a.txt"}))

                response = await client.post("/v1/messages", json={**anthropic_tool, "stream": True})
                self.assertEqual(response.status_code, 200, response.text)
                events = [line for line in response.text.splitlines() if line.startswith("event:")]
                self.assertIn("event: content_block_start", events)
                self.assertIn("event: message_stop", events)
                self.assertIn('"stop_reason":"tool_use"', response.text)

                response = await client.post("/v1/messages", json={
                    "model": "test-alias", "max_tokens": 64, "messages": messages,
                    "tools": [{"name": "Broken", "input_schema": {"type": 7}}],
                })
                self.assertEqual(response.status_code, 400, response.text)

                openai_tool = {
                    "model": backend_model,
                    "messages": messages,
                    "tools": [{"type": "function", "function": {
                        "name": "read", "parameters": TOOL_SCHEMA,
                    }}],
                    "tool_choice": {"type": "function", "function": {"name": "read"}},
                    "stream": False,
                }
                response = await client.post("/v1/chat/completions", json=openai_tool)
                self.assertEqual(response.status_code, 200, response.text)
                call = response.json()["choices"][0]["message"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "read")

                response = await client.post("/v1/messages", json={
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                    "stream": True,
                })
                self.assertEqual(response.status_code, 200, response.text)
                event_names = [
                    line[7:] for line in response.text.splitlines()
                    if line.startswith("event:")
                ]
                self.assertIn("message_start", event_names)
                self.assertIn("content_block_delta", event_names)
                self.assertEqual(event_names[-1], "message_stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `test_request_queue.py`

````python
"""FIFO queue, RPM pacing, cancellation, and HTTP back-pressure tests."""
from __future__ import annotations

import asyncio
import unittest

import httpx
from fastapi import FastAPI

from request_queue import RequestQueue, RequestQueueFull, RequestQueueMiddleware


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += delay
        await asyncio.sleep(0)


class RequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for_pending(self, queue: RequestQueue, expected: int) -> None:
        for _ in range(100):
            if (await queue.snapshot())["pending"] == expected:
                return
            await asyncio.sleep(0)
        self.fail(f"queue did not reach pending={expected}")

    async def test_provider_starts_are_smoothed_and_rolling_window_bounded(self):
        clock = FakeClock()
        queue = RequestQueue(2, clock=clock.now, sleep=clock.sleep)

        starts = [await queue.wait_for_provider_start() for _ in range(3)]

        self.assertEqual(starts, [0.0, 30.0, 60.0])
        status = await queue.snapshot()
        self.assertEqual(status["provider_requests_started_last_60_seconds"], 2)
        self.assertEqual(status["total_provider_requests_started"], 3)

    async def test_fifo_order_and_bounded_waiting_capacity(self):
        queue = RequestQueue(60, max_pending=2)
        first = await queue.acquire_request()
        order: list[str] = []

        async def queued_request(name: str) -> None:
            lease = await queue.acquire_request()
            try:
                order.append(name)
            finally:
                await lease.release()

        second = asyncio.create_task(queued_request("second"))
        await self._wait_for_pending(queue, 1)
        third = asyncio.create_task(queued_request("third"))
        await self._wait_for_pending(queue, 2)

        with self.assertRaises(RequestQueueFull):
            await queue.acquire_request()

        await first.release()
        await asyncio.gather(second, third)
        self.assertEqual(order, ["second", "third"])
        status = await queue.snapshot()
        self.assertEqual(status["pending"], 0)
        self.assertEqual(status["queue_full_rejections"], 1)
        self.assertEqual(status["completed_requests"], 3)

    async def test_cancelled_waiter_is_removed_and_does_not_block_queue(self):
        queue = RequestQueue(60, max_pending=2)
        active = await queue.acquire_request()

        async def wait_forever() -> None:
            await queue.acquire_request()

        waiter = asyncio.create_task(wait_forever())
        await self._wait_for_pending(queue, 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        status = await queue.snapshot()
        self.assertEqual(status["pending"], 0)
        self.assertEqual(status["cancelled_while_waiting"], 1)
        await active.release()

    async def test_asgi_middleware_queues_and_returns_protocol_specific_429(self):
        queue = RequestQueue(60, max_pending=1)
        app = FastAPI()
        app.add_middleware(RequestQueueMiddleware, queue=queue)
        first_entered = asyncio.Event()
        finish_first = asyncio.Event()
        processed: list[int] = []

        @app.post("/v1/chat/completions")
        async def chat():
            index = len(processed) + 1
            processed.append(index)
            if index == 1:
                first_entered.set()
                await finish_first.wait()
            return {"processed": index}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v1/chat/completions", json={"n": 1}))
            await asyncio.wait_for(first_entered.wait(), timeout=1)
            second = asyncio.create_task(client.post("/v1/chat/completions", json={"n": 2}))
            await self._wait_for_pending(queue, 1)

            openai_rejection = await client.post("/v1/chat/completions", json={"n": 3})
            self.assertEqual(openai_rejection.status_code, 429)
            self.assertEqual(openai_rejection.json()["error"]["code"], "queue_full")
            self.assertEqual(openai_rejection.headers["x-bridge-queue"], "full")

            anthropic_rejection = await client.post("/v1/messages", json={"n": 4})
            self.assertEqual(anthropic_rejection.status_code, 429)
            self.assertEqual(anthropic_rejection.json()["type"], "error")
            self.assertEqual(
                anthropic_rejection.json()["error"]["type"], "rate_limit_error"
            )

            finish_first.set()
            first_response, second_response = await asyncio.gather(first, second)
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(processed, [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `test_startup_config.py`

````python
"""Tests for startup prompt order, validation, and non-interactive overrides."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import startup_config


class StartupConfigTests(unittest.TestCase):
    def test_startup_asks_method_then_rpm_and_retries_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input", side_effect=["x", "2", "zero", "7"]) as prompt:
            mode, rpm, pending = startup_config.choose_startup_config("deepseek")

        self.assertEqual((mode, rpm, pending), ("user_only", 7, 100))
        self.assertEqual(prompt.call_count, 4)
        self.assertIn("method 1 or 2", prompt.call_args_list[1].args[0])
        self.assertIn("requests per minute", prompt.call_args_list[3].args[0])

    def test_environment_values_skip_interactive_prompts(self):
        with patch.dict(os.environ, {
            "PROMPT_MODE": "2",
            "DEEPSEEK_RPM": "12",
            "DEEPSEEK_MAX_QUEUE": "30",
            "BRIDGE_STARTUP_CONFIGURED": "1",
        }, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input") as prompt:
            config = startup_config.choose_startup_config("deepseek")

        self.assertEqual(config, ("user_only", 12, 30))
        prompt.assert_not_called()

    def test_interactive_launch_still_asks_when_environment_defaults_exist(self):
        with patch.dict(os.environ, {
            "PROMPT_MODE": "user_only",
            "DEEPSEEK_RPM": "12",
        }, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input", side_effect=["", ""]) as prompt:
            config = startup_config.choose_startup_config("deepseek")

        self.assertEqual(config, ("user_only", 12, 100))
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("default 2", prompt.call_args_list[0].args[0])
        self.assertIn("[12]", prompt.call_args_list[1].args[0])

    def test_noninteractive_launch_uses_safe_defaults(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=False):
            self.assertEqual(startup_config.choose_startup_config("qwen"), ("full", 5, 100))

    def test_environment_values_are_explicit_for_run_both_children(self):
        base = {"EXISTING": "kept", "BRIDGE_RPM": "99"}
        env = startup_config.environment_for_child("qwen", "full", 8, 25, base)
        self.assertEqual(env["EXISTING"], "kept")
        self.assertEqual(env["PROMPT_MODE"], "full")
        self.assertEqual(env["QWEN_RPM"], "8")
        self.assertEqual(env["QWEN_MAX_QUEUE"], "25")
        self.assertEqual(env["BRIDGE_STARTUP_CONFIGURED"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
````

## `CLAUDE_CODE_SETUP.md`

````markdown
# Anthropic Messages + OpenAI bridge setup

This project keeps the existing OpenAI Chat Completions API and adds a minimum Anthropic Messages API adapter. It targets Claude Code/custom Anthropic Messages clients and OpenAI Chat Completions clients; it cannot guarantee compatibility with every proprietary agent. For the reference-based parser review and exact offline check results, see `REFERENCE_REVIEW.md`.

## Files required for the runtime feature

Copy/replace these complete runtime files in the project root:

- `server.py` — DeepSeek server routes and adapter wiring.
- `qwen_server.py` — Qwen server routes and adapter wiring.
- `anthropic_compat.py` — Anthropic request/response normalization and SSE events.
- `tool_bridge.py` — request-scoped tool formatting, parsing, and JSON Schema validation.
- `requirements.txt` — adds `jsonschema` for tool-input validation.

`app.py` and `qwen_app.py` are unchanged. `run_both.py` only changes its printed endpoint hints and is optional. The test files and this guide are not needed to run the bridge.

Install updated dependencies after replacing the files:

```powershell
python -m pip install -r requirements.txt
```

## Run the bridge and configure the queue

Start either server in a console. By default it asks for the prompt mode first, then asks for a positive provider-submission limit per minute (Enter accepts the conservative default of 5):

```powershell
python server.py       # DeepSeek, port 8000
# or
python qwen_server.py  # Qwen, port 8001
```

The RPM value paces **outbound browser-backed model submissions**, not just client HTTP requests. Each tool-call repair attempt also counts. The prompt is a local safety setting; it cannot discover or override the provider's own web-chat limits.

For non-interactive launches, set environment variables instead. On an interactive terminal, valid environment values prefill the prompts; the console still asks unless a supervisor has already chosen the settings. Provider-specific values take precedence over `BRIDGE_RPM` and `RATE_LIMIT_RPM`:

```powershell
# DeepSeek
$env:PROMPT_MODE = 'full'       # or 'user_only'
$env:DEEPSEEK_RPM = '5'
python server.py

# Qwen
$env:PROMPT_MODE = 'full'
$env:QWEN_RPM = '5'
python qwen_server.py
```

The client queue is on by default. Each server runs one browser-backed generation at a time in FIFO order, then paces provider starts against a rolling 60-second window without a startup burst. By default, up to 100 additional generation requests may wait; change this with `DEEPSEEK_MAX_QUEUE`, `QWEN_MAX_QUEUE`, or the shared `BRIDGE_MAX_QUEUE` environment variable. `GET /queue/status` reports queue depth and recent provider starts.

No finite machine can safely hold an unlimited backlog. If all pending slots are occupied, the bridge returns a structured HTTP 429 `queue_full` response instead of growing memory without bound or crashing. Clients should back off before retrying; while the process remains up, accepted requests stay queued rather than being silently discarded. The in-memory queue is not durable: a process crash/restart can interrupt queued work. The cap is configurable from 1 to 10,000 pending requests.

`run_both.py` asks for the prompt mode and independent DeepSeek/Qwen RPM values in its own console, then passes those settings to each child process. This avoids two child processes racing to read the same terminal input.

Start Claude Code in a second terminal with the base URL set to the server root (do not append `/v1`):

```powershell
# DeepSeek
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8000'
claude

# Qwen uses port 8001 instead.
```

The persistent user-level environment variable can be set using Windows Environment Variables or PowerShell's `[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', 'http://127.0.0.1:8000', 'User')`. Open a new terminal after changing it. Claude Code must inherit that variable; setting it only in the server terminal is not enough.

The DeepSeek/Qwen bridges open a visible browser and wait for manual login. Browser profiles (`browser_data/`, `qwen_browser_data/`) contain session data; keep them private. A desktop GUI is required by the current headed-browser setup.

## Endpoints

- DeepSeek: `POST http://127.0.0.1:8000/v1/messages`, optional `/v1/messages/count_tokens`.
- Qwen: `POST http://127.0.0.1:8001/v1/messages`, optional `/v1/messages/count_tokens`.
- Existing OpenAI route remains `POST /v1/chat/completions` on both servers.
- `GET` and `HEAD /api/hello` are lightweight client connectivity probes; `/health` remains available.
- `GET /queue/status` reports FIFO queue depth, active request time, RPM pacing, and local counters.

## Tool-calling model and limitations

Tool definitions are request-scoped. Before a call is returned, the bridge checks the name against the exact request and validates the parsed argument object against its JSON Schema. It supports OpenAI `tool_choice` modes, named tools, `parallel_tool_calls: false`, Anthropic `auto`/`any`/named/`none` choices, and Anthropic `disable_parallel_tool_use`. The bridge returns calls to the client; it never executes them or bypasses the client's approval policy.

The text parser handles the bridge's `<tool_call>` format, DeepSeek-family native markers and DSML tool wrappers (including the V3.2/V4/V4.1 variants such as `<｜DSML｜ calls>`), exact-response JSON/OpenAI-style tool-call envelopes, Anthropic-style `<invoke>` blocks, and the existing `<run_code>` wrapper. DSML's `string="true"` parameters stay strings; `string="false"` parameters are decoded as JSON. It buffers split markers, preserves multiple calls, parses nested JS-like objects without evaluating code, caps a response at 32 calls, and performs up to two repair retries after the initial attempt. Bare JSON is recognized only when it is the whole response (or a whole fenced response), not when embedded in explanatory prose.

DeepSeek/Qwen **web sessions do not expose native function-calling APIs to this bridge**. Structured tool calls are text the model is asked to emit, so compliance is probabilistic; parsing, retries, and schema validation reduce errors but do not make it native or guarantee success. Anthropic provider-managed/server tools (for example hosted web-search/computer tools) are explicitly unsupported; use client-defined tools with `input_schema`. Tool-enabled replies may be buffered before Anthropic tool-use events are emitted. The bridge targets Anthropic Messages and OpenAI Chat Completions, not every agent-specific or proprietary protocol. It does not implement the OpenAI Responses API, Google-specific formats, full MCP, or every Claude Code beta feature.

Token counts are rough estimates, and OpenAI `stream_options.include_usage` usage accounting is not synthesized. The browser backends do not faithfully enforce generation controls such as `temperature` or `max_tokens`; image/document inputs, signed thinking, prompt caching, and adaptive thinking are unsupported. The model field does not select the backend; port 8000 is DeepSeek and 8001 is Qwen.

The local API has no authentication by default; keep it bound to loopback and do not expose it to a network. Routing Claude Code to non-Claude models is not an officially supported Anthropic configuration. Use a non-sensitive test project and keep tool approval enabled.

## Offline tests

```powershell
python test_parser.py
python test_tool_bridge.py
python test_anthropic_compat.py
python test_api_smoke.py
```

These use fake streams and validate the existing DeepSeek/Qwen stream parsers, tool-call parsing/validation and both API routes, Anthropic/OpenAI SSE framing, and tool-choice behavior. They do not verify live provider selectors, browser logins, or current DeepSeek/Qwen website behavior.
````

## `QUEUE_RATE_LIMIT_RESEARCH.md`

````markdown
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
````

## `REFERENCE_REVIEW.md`

````markdown
# Reference-driven tool-calling review

**Review date:** 2026-10-27 (Asia/Colombo)  
**Reference revision:** `99f77302cebc0b32f10066f96b863f2f27752708` ([NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api/tree/99f77302cebc0b32f10066f96b863f2f27752708)). Relevant source paths:

- [`src/openai_adapter/request/tools.rs`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/src/openai_adapter/request/tools.rs) — validates tool definitions/choices and builds tool-call instructions.
- [`src/openai_adapter/response/tool_parser.rs`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/src/openai_adapter/response/tool_parser.rs) — parses DeepSeek-native tool markers, repairs some malformed structured output, handles invoke-style calls, and includes streaming-specific limits.
- [`src/anthropic_compat`](https://github.com/NIyueeE/ds-free-api/tree/99f77302cebc0b32f10066f96b863f2f27752708/src/anthropic_compat) and [`docs/compat-audit.md`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/docs/compat-audit.md) — document Anthropic event sequencing, stop reasons, error events, and OpenAI SSE termination expectations.
- [DeepSeek V4.1 encoding reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/README.md) — defines the newer DSML `<｜DSML｜ calls>` / `invoke` / `parameter` syntax and the `string="true|false"` parameter rule.

## Findings applied to this project

The parser work also addressed the exact format in the follow-up report: DeepSeek V4.1 documents DSML wrappers such as `<｜DSML｜ calls>` with `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>` tags (note the leading spaces), while earlier variants use `tool_calls` or `function_calls`. The sample's doubled vertical bars are accepted too. The `string="true"` attribute means keep a parameter as literal text; `string="false"` means decode its value as JSON. This is why a V4.1 model may output a different-looking call than the bridge's preferred `<tool_call>` wrapper: it is speaking its own documented tool-call grammar, not accidentally writing shell commands as prose.

More broadly, the project primarily asked the web model to emit `<tool_call>` XML and `<run_code>` code. The reference also recognizes the DeepSeek-native `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>` array form, multiple spellings of those markers, and `<invoke>/<parameter>` blocks. A compatible parser should treat stream chunk boundaries as arbitrary, allow more than one call, bound its buffering, and validate the resulting call against the request before returning it.

The current `tool_bridge.py` now:

- recognizes the bridge XML form, DeepSeek-native marker forms, DeepSeek DSML V3.2/V4/V4.1 wrappers (including the V4.1 leading-space spelling), and generic `<tool_calls>` wrappers; it carries partial start/end markers across chunks and can extract multiple calls;
- accepts standard JSON and OpenAI-like `tool_calls` envelopes, plus a conservative relaxed parser for common model formatting errors (single quotes, unquoted keys, and trailing commas); untagged JSON must be the entire response or a whole code fence, which avoids extracting an illustrative code block from prose;
- parses `<invoke>/<parameter>` blocks and nested JS-like `<run_code>` arguments using a non-evaluating parser; no model-generated code is executed;
- enforces request-scoped names and JSON Schemas (local `$ref` only; remote schema references are rejected), caps returned calls at 32, honors `tool_choice` (including `required`/named tools), supports OpenAI `parallel_tool_calls: false`, and makes up to two repair retries after the first attempt;
- emits standard nested OpenAI error envelopes for tool-path failures and tool-stream chunks with one stable completion ID/timestamp, an initial assistant-role chunk, the correct `tool_calls` finish reason, and exactly one `[DONE]` marker.

The Anthropic adapter now also honors `disable_parallel_tool_use`, maps OpenAI `length`/content-filter finish reasons to Anthropic stop reasons, and explicitly rejects unsupported provider-managed Anthropic tools instead of silently fabricating a local function tool. Tool-enabled Anthropic streams preserve the `content_block_start` / delta / stop event sequence. The existing `app.py` and `qwen_app.py` browser/UI automation files were not edited; no direct provider API completion call was introduced.

## Deliberate exclusions

The reference project documents upstream risk-control tokens, device-fingerprint/emulation options, session/account controls, and ban/mute mitigation. These were not adopted. This project makes no attempt to spoof identities, emulate risk tokens, rotate accounts/devices, or bypass provider enforcement. The changes are limited to protocol parsing, validation, bounded retries, and truthful error behavior. Browser UI automation is still subject to provider terms and can still fail or lead to account restrictions; this work does not make it safer from bans.

## Verification and remaining limits

Offline verification passed after these changes:

- `python test_tool_bridge.py` — 13 focused parser, DSML, validation, parallel-call, schema-reference, SSE, and request-choice tests.
- `python test_anthropic_compat.py` — 13 adapter tests.
- `python test_api_smoke.py` — both server modules' OpenAI/Anthropic routes with a fake stream.
- `python test_parser.py` — 10 existing DeepSeek/Qwen parser scenarios.
- `python -m compileall -q .`, Ruff `--select F,E9`, and `git diff --check`.

These tests use fake provider streams. There is still **no live browser/provider/agent round trip**, so no claim is made that every model will emit parseable tool calls or that every version of Claude Code/other agent clients will work. The bridge's tool calls remain model-generated text, not native provider function calls. Token estimates and unsupported Anthropic features remain as described in `CLAUDE_CODE_SETUP.md`. `pip-audit` was not installed during the earlier review, so that review did not perform a dependency vulnerability audit.

## Queue/rate-limit implementation addendum — 2026-10-27

The startup flow now asks for method 1 or 2, followed by a local provider-submission RPM value (default 5). Each backend has a bounded FIFO queue (one active generation plus 100 pending by default) and a smooth rolling-minute start limiter. Tool-repair submissions count against RPM. `GET /queue/status` exposes non-sensitive counters. Saturation returns a shaped 429 response; see `QUEUE_RATE_LIMIT_RESEARCH.md` for the design, research, and operational caveats.

The prior audit's three reproduced API defects were also addressed: validation errors no longer echo input values, unsupported message content yields 400, and provider failures no longer become successful empty non-streaming completions. Regular OpenAI stream chunks now share a stable ID. The current offline suite comprises `test_parser.py`, `test_tool_bridge.py` (13), `test_anthropic_compat.py` (13), `test_api_smoke.py`, `test_request_queue.py` (4), and `test_startup_config.py` (5); all pass in the workspace. Ruff `F,E9`, compilation, and `git diff --check` pass. These remain fake-stream/offline checks only; live provider behavior is unverified.
````

## `ANALYSIS.md`

````markdown
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
````
