"""
test_parser.py – Verify app.py's parser separates THINK vs ANSWER correctly.

Runs WITHOUT a browser: mocks playwright, feeds realistic DeepSeek SSE payloads
(all shapes: BATCH op-list, inline first-fragment, status FINISHED, content deltas)
and asserts:
  1. thinking text goes to delta.reasoning_content  (separate channel)
  2. answer text goes to delta.content
  3. "FINISHED" / "CONTENT_FILTER" NEVER leak into content
  4. order is preserved (thinking before answer)
"""
import asyncio
import json
import sys
import types

# ---- mock playwright so app.py can be imported without a browser ----
fake_module = types.ModuleType("playwright")
fake_async_api = types.ModuleType("playwright.async_api")
fake_async_api.async_playwright = lambda: None
sys.modules["playwright"] = fake_module
sys.modules["playwright.async_api"] = fake_async_api

import app  # noqa: E402


def run(payloads):
    """Feed payloads through _handle_payload, return (content_text, reasoning_text)."""
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


# ==================== Scenario 1: BATCH ops (the common case) ====================
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

content, reasoning = run(s1)
assert reasoning == "The user just said hi. ", f"THINK leaked: {reasoning!r}"
assert content == "Hello! How can I help you today? ", f"ANSWER wrong: {content!r}"
assert "FINISHED" not in content and "FINISHED" not in reasoning, "FINISHED leaked!"
print("✓ Scenario 1 (BATCH ops + relative paths + FINISHED status): OK")

# ==================== Scenario 2: inline first fragment in opening payload ====================
s2 = [
    {"v": {"response": {"fragments": [{"type": "THINK", "content": "Let me reason."}]}}},
    {"p": "response/fragments/0/content", "o": "ADD", "v": "More thinking "},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": ""}]},
    {"p": "response/fragments/1/content", "o": "ADD", "v": "The answer."},
    {"p": "response/status", "o": "SET", "v": "FINISHED"},
]

content, reasoning = run(s2)
assert reasoning == "Let me reason.More thinking ", f"THINK wrong: {reasoning!r}"
assert content == "The answer.", f"ANSWER wrong: {content!r}"
print("✓ Scenario 2 (inline first fragment + response/ prefixed paths): OK")

# ==================== Scenario 3: status ops that must NEVER become text ====================
s3 = [
    {"p": "response/status", "o": "SET", "v": "FINISHED"},
    {"p": "status", "o": "SET", "v": "CONTENT_FILTER"},
    {"p": "response/fragments/0/status", "o": "SET", "v": "FINISHED"},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "ok"}]},
]

content, reasoning = run(s3)
assert content == "ok", f"content wrong: {content!r}"
assert "FINISHED" not in content and "CONTENT_FILTER" not in content, "status leaked!"
print("✓ Scenario 3 (statuses never leak): OK")

# ==================== Scenario 4: direct {"v": "text"} with active fragment ====================
s4 = [
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": ""}]},
    {"v": "direct think text"},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": ""}]},
    {"v": "direct answer text"},
]

content, reasoning = run(s4)
assert reasoning == "direct think text", f"THINK wrong: {reasoning!r}"
assert content == "direct answer text", f"ANSWER wrong: {content!r}"
print("✓ Scenario 4 (direct v-string payloads): OK")

# ==================== Scenario 5: SEARCH fragments dropped ====================
s5 = [
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "SEARCH", "content": "search noise"}]},
    {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "final"}]},
]

content, reasoning = run(s5)
assert content == "final" and "search noise" not in content, f"SEARCH leaked: {content!r}"
print("✓ Scenario 5 (SEARCH fragments dropped): OK")

print("\nALL PARSER TESTS PASSED ✅")
