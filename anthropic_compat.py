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
