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