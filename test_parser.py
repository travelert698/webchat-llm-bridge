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