"""
server.py – OpenAI-compatible API server for the DeepSeek Bridge

Runs on port 8000 by default. To change:  set DEEPSEEK_PORT env var, e.g.
    DEEPSEEK_PORT=8010 python server.py

HYBRID (full passthrough + tool calling):
  - PLAIN CHAT (no tools): the FULL original request (system + all messages,
    in order, roles labelled) is rendered and sent to the web AI exactly as
    received — nothing dropped, nothing refined.
  - TOOL REQUESTS (tools present): same full passthrough PLUS a small
    whitelisted tool block is appended, and the stream is sieved so the web
    model's <tool_call> reply is converted into real OpenAI tool_calls that
    DSH executes. Tool results round-trip as [TOOL RESULT (name)] turns.
"""
import asyncio
import json
import os
import time
import uuid
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn
import app          # the DeepSeek bridge module (app.py)
import tool_bridge  # the hybrid tool-calling module (tool_bridge.py)

# ==================== Port ====================
PORT = int(os.environ.get("DEEPSEEK_PORT", "8000"))

# ==================== Prompt mode (asked at startup) ====================
# "full"      -> system + ALL messages (roles labelled) — works with both
#                Continue.dev and DeepSeek Harness
# "user_only" -> ONLY the last user message — also works with both agents
def _choose_prompt_mode() -> str:
    env = os.environ.get("PROMPT_MODE", "").strip().lower()
    if env in ("full", "user_only"):
        return env
    try:
        print("\n=== Prompt mode selection ===")
        print("  [1] FULL passthrough (system + all messages)")
        print("  [2] USER message only")
        print("  (Both modes work with Continue.dev AND DeepSeek Harness.)")
        choice = input("Enter 1 or 2 [default 1]: ").strip()
        return "user_only" if choice == "2" else "full"
    except Exception:
        return "full"

PROMPT_MODE = _choose_prompt_mode()
tool_bridge.PROMPT_MODE = PROMPT_MODE   # sync the global tool-bridge mode

# ==================== FastAPI app ====================
api = FastAPI(title="DeepSeek Bridge API")

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
    tool_choice: Optional[Any] = None     # "auto" | "none" | {"function": {...}}

# ==================== Lifespan ====================
@api.on_event("startup")
async def startup():
    await app.startup()
    print("\n" + "=" * 60)
    print(f"  DeepSeek Bridge API running at http://127.0.0.1:{PORT}")
    print("  Endpoint: POST /v1/chat/completions")
    print(f"  Base URL for agent: http://127.0.0.1:{PORT}/v1")
    print("  Mode: full passthrough + tool calling (tool_bridge.py)")
    print(f"  Prompt mode: {PROMPT_MODE}")
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
    body = await request.body()
    print(f"[VALIDATION ERROR] {exc.errors()}")
    print(f"[REQUEST BODY] {body.decode()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": body.decode()})

# ==================== Chat completions ====================
@api.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # ---- Hybrid tool path: only when tools are present ----
    if getattr(req, "tools", None):
        result = await tool_bridge.handle_chat(req, bridge=app)
        if result is not None:
            return result
        # fall through to plain path if handle_chat returned None (tool_choice none)

    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided."})
    prompt = render_messages(req.messages)
    if not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Empty message."})

    if not req.stream:
        full_text = ""
        full_reasoning = ""
        try:
            async for chunk in app.stream_response(prompt):
                parsed = app._parse_sse(chunk)
                if parsed and "error" not in parsed:
                    full_text += parsed.get("content", "") or ""
                    full_reasoning += parsed.get("reasoning", "") or ""
        except Exception as e:
            print(f"[SERVER] non-streaming error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

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
        yield ": connected\n\n"
        init = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model or "deepseek-chat",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(init)}\n\n"

        try:
            async for chunk in app.stream_response(prompt):
                # Chunk is already a valid OpenAI SSE string – forward as-is
                yield chunk

            final = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model or "deepseek-chat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"[SERVER] stream error: {e}")
            err = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ==================== Health check ====================
@api.get("/health")
async def health():
    return {"status": "ok"}

# ==================== Entry point ====================
if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=PORT)
