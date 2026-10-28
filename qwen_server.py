"""
qwen_server.py – OpenAI-compatible API server for the Qwen Bridge

Runs on port 8001 by default (DeepSeek server.py keeps port 8000).
To change: set QWEN_PORT env var, e.g.
    QWEN_PORT=8020 python qwen_server.py
Interactive startup asks for prompt mode, then a completion-to-next-request delay in seconds.
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
PROMPT_MODE, INTER_REQUEST_DELAY_SECONDS, MAX_PENDING_REQUESTS = choose_startup_config("qwen")
tool_bridge.PROMPT_MODE = PROMPT_MODE   # sync the global tool-bridge mode

# ==================== Bounded client queue + completion cooldown ====================
REQUEST_QUEUE = RequestQueue(
    inter_request_delay_seconds=INTER_REQUEST_DELAY_SECONDS,
    max_pending=MAX_PENDING_REQUESTS,
)
_original_stream_response = app.stream_response


async def _serialized_stream_response(prompt: str):
    """Hold the provider slot for the whole response and cooldown afterward."""
    async with REQUEST_QUEUE.provider_call():
        async for chunk in _original_stream_response(prompt):
            yield chunk


# All plain chat, tool-bridge, and Anthropic adapter calls share this wrapper.
app.stream_response = _serialized_stream_response

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
    print(f"  Completion cooldown: {INTER_REQUEST_DELAY_SECONDS:g} seconds")
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
