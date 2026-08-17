"""
tool_bridge.py — Hybrid Tool-Calling Bridge Module (shared by server.py and qwen_server.py)

DESIGN:
  - FULL PASSTHROUGH: the system prompt and the entire original conversation
    from the client are rendered and sent to the web AI EXACTLY as received.
  - PLUS, ONLY when the request carries `tools`: a small, whitelisted tool
    block is appended telling the web model to reply with a strict tool-call,
    and the streamed answer is SIEVED. The sieve understands MULTIPLE tool-call
    formats the web models actually emit:
        A) <tool_call><name>X</name><args>{json}</args></tool_call>
        B) <run_code ...> ... tools.write({...}) ... </run_code>   (JS call form)
        C) plain JSON {"name": ..., "arguments": {...}} / [ ... ]
    Each recognized call is converted into real OpenAI `delta.tool_calls` +
    finish_reason:"tool_calls" that DSH executes. Tool results round-trip as
    [TOOL RESULT (name)] turns.

  WHY B: DSH's own system prompt (now passed through) describes a `run_code`
  tool, so web models often emit <run_code> blocks with embedded
  `tools.NAME({...})` calls. The bridge parses those JS-style calls (including
  resolving `const html = `...`;` template variables) so DSH still receives
  proper tool_calls.

The whitelist keeps the prompt small: DSH sends ~25 tools; web models get
overwhelmed and ramble. Keep the list at 6-8 for reliable compliance.

PROMPT MODE (global, set by server at startup):
  "full"      -> system + ALL messages (roles labelled, tool_calls preserved)
  "user_only" -> ONLY the last user message (classic chat behavior)
  Both Continue.dev AND DeepSeek Harness work with either mode.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse

# ==================== Config ====================
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
RUN_CODE_START = "<run_code"
RUN_CODE_END = "</run_code>"
MAX_REPAIR_RETRIES = 2
MAX_TOOL_BLOCK_CHARS = 400_000
MAX_TOTAL_CHARS = 300_000

# Prompt mode — GLOBAL toggle, set at startup by the server (question + env).
PROMPT_MODE = "full"

# Only these tools are injected into the prompt. Add/remove names freely.
TOOL_WHITELIST = {
    "read", "write", "edit", "glob", "grep",
    "todo_write", "web_search", "ask_user_question",
}
MAX_TOOLS_INJECTED = 6


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
def serialize_tools(tools: Any) -> str:
    lines: list[str] = []
    if not isinstance(tools, list):
        return ""
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = str(fn.get("name") or "unnamed")
        if TOOL_WHITELIST and name not in TOOL_WHITELIST:
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        if isinstance(props, dict):
            args_desc = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in list(props.items())[:6]
            )
        else:
            args_desc = ""
        desc = str(fn.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 180:
            desc = desc[:180] + "…"
        lines.append(f"Tool: {name}({args_desc}) — {desc}")
        if len(lines) >= MAX_TOOLS_INJECTED:
            break
    return "\n".join(lines)


def build_tool_block(tools_text: str, forced_tool: Optional[str]) -> str:
    parts = [
        "[SYSTEM RULE — TOOL CALLING]",
        "You are connected to a tool system. If a task requires reading, writing, editing, "
        "searching files, or web search, you MUST call a tool.",
        "The ONLY valid tool-call format is this — exactly, nothing else around it, no markdown, no explanation:",
        '<tool_call><name>TOOL_NAME</name><args>{"key": "value"}</args></tool_call>',
        "Rules:",
        "  - Do NOT use <run_code>, <code>, code fences, or JSON arrays for tool calls.",
        "  - Do NOT think out loud about which tool to use.",
        "  - Do NOT explain what you will do.",
        "  - Do NOT ask the user for confirmation.",
        "  - If you need a tool: emit ONLY the tool_call block, then STOP.",
        "  - If the task can be answered directly, reply normally.",
    ]
    if forced_tool:
        parts.append(f"You MUST use the tool named: {forced_tool}")
    parts.append("")
    parts.append("[TOOLS AVAILABLE]")
    parts.append(tools_text)
    return "\n".join(parts)


def resolve_tool_choice(tool_choice: Any) -> Optional[str]:
    if isinstance(tool_choice, str):
        return None if tool_choice.lower() in ("auto", "none") else tool_choice
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        return str(fn.get("name") or "") or None
    return None


# ==================== Tool block parsing (multi-format) ====================
def _strip_fences(block: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", block, re.S)
    return m.group(1).strip() if m else block.strip()


def _parse_tool_call_xml(block: str) -> Optional[dict]:
    block = _strip_fences(block)
    name: Optional[str] = None
    args: Any = None

    mn = re.search(r"<name>(.*?)</name>", block, re.S)
    ma = re.search(r"<args>(.*?)</args>", block, re.S)
    if mn:
        name = mn.group(1).strip()
    if ma:
        raw = ma.group(1).strip()
        try:
            args = json.loads(raw)
        except Exception:
            mm = re.search(r"\{.*\}", raw, re.S)
            if mm:
                try:
                    args = json.loads(mm.group(0))
                except Exception:
                    args = raw
            else:
                args = raw

    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    if not isinstance(args, dict):
        args = {"value": args} if args is not None else {}
    return {"name": name, "arguments": args}


def _parse_js_object(raw: str, templates: dict) -> Optional[dict]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    obj: dict = {}
    pattern = re.compile(
        r"""([A-Za-z_$][\w$]*)\s*:\s*(?:
              "((?:[^"\\]|\\.)*)"          # double-quoted
            | '((?:[^'\\]|\\.)*)'          # single-quoted
            | `([\s\S]*?)`                 # template literal
            | ([^,}\s][^,}]*)              # bare value / variable
            )""",
        re.X,
    )
    for m in pattern.finditer(raw):
        key = m.group(1)
        if m.group(2) is not None:
            val = m.group(2)
            try:
                val = bytes(val, "utf-8").decode("unicode_escape")
            except Exception:
                pass
        elif m.group(3) is not None:
            val = m.group(3)
        elif m.group(4) is not None:
            val = m.group(4)
        else:
            val = m.group(5).strip()
            if val in templates:
                val = templates[val]
            elif val in ("true",):
                val = True
            elif val in ("false",):
                val = False
            elif val in ("null", "undefined"):
                val = None
            else:
                try:
                    val = json.loads(val)
                except Exception:
                    pass
        obj[key] = val
    return obj if obj else None


def _parse_run_code(block: str) -> list:
    gt = block.find(">")
    if gt != -1:
        block = block[gt + 1:]

    templates: dict = {}
    for m in re.finditer(r"const\s+(\w+)\s*=\s*`([\s\S]*?)`\s*;", block):
        templates[m.group(1)] = m.group(2)

    calls: list = []
    for m in re.finditer(r"tools\.(\w+)\s*\(\s*(\{[\s\S]*?\})\s*\)", block):
        name = m.group(1)
        args = _parse_js_object(m.group(2), templates)
        if args is not None:
            calls.append({"name": name, "arguments": args})
    return calls


def parse_tool_blocks(kind: str, block: str) -> list:
    if kind == "tool_call":
        parsed = _parse_tool_call_xml(block)
        return [parsed] if parsed else []
    if kind == "run_code":
        return _parse_run_code(block)
    if kind == "json":
        block = _strip_fences(block)
        try:
            obj = json.loads(block)
        except Exception:
            return []
        items = obj if isinstance(obj, list) else [obj]
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            args = item.get("arguments", item.get("args"))
            if not name:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"value": args}
            if not isinstance(args, dict):
                args = {"value": args} if args is not None else {}
            out.append({"name": name, "arguments": args})
        return out
    return []


# ==================== Stream sieve (multi-format) ====================
class Sieve:
    """Pull tool-call blocks out of a content stream. Supports
    <tool_call>..</tool_call> and <run_code ..>..</run_code> (and bare JSON)."""

    def __init__(self) -> None:
        self._carry = ""
        self._in_block = False
        self._block = ""
        self._current_end: Optional[str] = None
        self._current_kind: str = "tool_call"
        self._done = False

    def _reset_block(self) -> None:
        self._in_block = False
        self._block = ""
        self._current_end = None
        self._current_kind = "tool_call"

    def feed(self, text: str):
        out: list = []
        if self._done:
            return out
        self._carry += text
        while not self._done and self._carry:
            if not self._in_block:
                earliest = None
                for start, end, kind in (
                    (TOOL_CALL_START, TOOL_CALL_END, "tool_call"),
                    (RUN_CODE_START, RUN_CODE_END, "run_code"),
                ):
                    idx = self._carry.find(start)
                    if idx != -1 and (earliest is None or idx < earliest[0]):
                        earliest = (idx, start, end, kind)
                if earliest is None:
                    out.append(("content", self._carry))
                    self._carry = ""
                    break
                idx, start, end, kind = earliest
                if idx > 0:
                    out.append(("content", self._carry[:idx]))
                self._carry = self._carry[idx + len(start):]
                self._in_block = True
                self._current_end = end
                self._current_kind = kind
                self._block = ""
                continue
            idx = self._carry.find(self._current_end)
            if idx == -1:
                self._block += self._carry
                self._carry = ""
                if len(self._block) > MAX_TOOL_BLOCK_CHARS:
                    out.append(("content", self._block))
                    self._reset_block()
                break
            self._block += self._carry[:idx]
            self._carry = self._carry[idx + len(self._current_end):]
            out.append(("tool_block", self._current_kind, self._block))
            self._reset_block()
            self._done = True
        return out


# ==================== SSE wire helpers ====================
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _content_chunk(text: str, model: str) -> str:
    return _sse({
        "id": f"chatcmpl-{secrets.token_hex(6)}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    })


def _reasoning_chunk(text: str, model: str) -> str:
    return _sse({
        "id": f"chatcmpl-{secrets.token_hex(6)}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
    })


def _tool_call_chunk(name: str, arguments: dict, call_id: str, index: int, model: str) -> str:
    return _sse({
        "id": f"chatcmpl-{secrets.token_hex(6)}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": index, "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }]
            },
            "finish_reason": None,
        }],
    })


def _finish_chunk(reason: str, model: str) -> str:
    return _sse({
        "id": f"chatcmpl-{secrets.token_hex(6)}", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    })


def _error_chunk(message: str, model: str) -> str:
    return _sse({"error": {"message": message, "type": "internal_error"}})


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


def _events_to_chunks(events: list, model: str) -> tuple[list, Optional[tuple]]:
    sieve = Sieve()
    out: list = []
    pending: list = []
    raw_blocks: list = []

    for kind, value in events:
        if kind == "error":
            return [_error_chunk(value, model)], None
        if kind == "content":
            for ev in sieve.feed(value):
                if ev[0] == "content":
                    pending.append(ev[1])
                else:
                    raw_blocks.append((ev[1], ev[2]))
        elif kind == "reasoning":
            out.append(_reasoning_chunk(value, model))

    if not raw_blocks:
        text = "".join(pending)
        if text:
            out.append(_content_chunk(text, model))
        out.append(_finish_chunk("stop", model))
        return out, None

    calls: list = []
    for bkind, braw in raw_blocks:
        calls.extend(parse_tool_blocks(bkind, braw))

    if not calls:
        return out, ("unparseable", "tool block")

    text = "".join(pending)
    if text:
        out.append(_content_chunk(text, model))
    for i, call in enumerate(calls):
        out.append(_tool_call_chunk(call["name"], call["arguments"],
                                    f"call_{secrets.token_hex(8)}", i, model))
    out.append(_finish_chunk("tool_calls", model))
    return out, ("parsed", calls)


# ==================== Main entry ====================
async def handle_chat(req, bridge) -> Any:
    tools = _g(req, "tools", None)
    tool_choice = _g(req, "tool_choice", None)
    has_tools = isinstance(tools, list) and len(tools) > 0

    if isinstance(tool_choice, str) and tool_choice.lower() == "none":
        return None
    if not has_tools:
        return None

    model = _g(req, "model") or "deepseek-chat"
    messages = list(_g(req, "messages") or [])
    forced_tool = resolve_tool_choice(tool_choice)
    stream = bool(_g(req, "stream", False))

    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages must be a non-empty list"})

    total = 0
    for m in messages:
        c = _g(m, "content")
        if isinstance(c, str):
            total += len(c)
    if total > MAX_TOTAL_CHARS:
        return JSONResponse(status_code=413, content={"error": "message payload too large"})

    try:
        base_prompt = render_messages(messages)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    tools_text = serialize_tools(tools)
    tool_block = build_tool_block(tools_text, forced_tool)
    prompt = base_prompt + "\n\n" + tool_block

    chunks: list = []
    retries = 0
    while True:
        events = await _run_bridge_turn(bridge, prompt)
        turn_chunks, outcome = _events_to_chunks(events, model)
        chunks.extend(turn_chunks)

        if outcome is None or outcome[0] == "parsed":
            break

        retries += 1
        if retries > MAX_REPAIR_RETRIES:
            chunks.append(_error_chunk("model failed to produce a valid tool call", model))
            break
        prompt = (
            "Your previous reply used an unsupported tool-call format. "
            "Reply with EXACTLY:\n"
            '<tool_call><name>TOOL_NAME</name><args>{"key": "value"}</args></tool_call>\n'
            "No other text, no <run_code>, no markdown."
        )

    if stream:
        async def gen():
            for chunk in chunks:
                yield chunk
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    content_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    for chunk in chunks:
        try:
            obj = json.loads(chunk[len("data: "):].strip())
        except Exception:
            continue
        if "error" in obj:
            return JSONResponse(status_code=502, content={"error": obj["error"]["message"]})
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id"), "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
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