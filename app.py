"""
app.py – DeepSeek Bridge (route interception + human idle + clean API output)

FIXED v3. What changed in this version:
1. STATUS LEAK FIXED         -> {"v": "FINISHED"} / {"v": "CONTENT_FILTER"} (status
   payloads) are NEVER emitted as content. A string value is only treated as text
   when the payload has NO path (p is None) — exactly like the reference driver.
2. INLINE FIRST-FRAGMENT      -> DeepSeek sometimes sends the first fragment inline in
   the opening payload: {"v": {"response": {"fragments": [...]}}}. That shape is now
   converted into a normal APPEND op so the THINK type is registered.
3. FRAGMENT TYPES BY INDEX    -> both path styles are resolved:
   "response/fragments/0/content" (index at [2]) and "fragments/0/content" (index at [1]).
   Unknown index falls back to the most recently appended fragment type (active type),
   like the reference driver's _stream_active_fragment_type.
4. THINKING -> delta.reasoning_content  -> this is exactly what Continue's
   fromChatCompletionChunk() maps to { role: "thinking" } and renders as a separate
   thinking block (verified against continuedev/continue PR #6236 + issue #11069).
   Set SEND_THINKING = False to drop thinking entirely.
5. DEBUG_DUMP                -> set DEBUG_DUMP = True to write every raw SSE data line
   to stream_dump.jsonl. Run one request, then send me that file to verify the exact
   DeepSeek payload shapes on your account.
6. INCREMENTAL UTF-8 DECODER -> no more "�" from split multi-byte characters.
7. TAIL LINE FLUSHED, CLIENT-DISCONNECT ABORT, IDLE SIMULATION FIX, 300s TIMEOUT
   (unchanged from v2).
"""
import asyncio
import codecs
import json
import random
import time
import httpx
from playwright.async_api import async_playwright

# ==================== Globals ====================
browser = None
page = None
process_lock = asyncio.Lock()
_route_registered = False

DEEPSEEK_URL = "https://chat.deepseek.com"
USER_DATA_DIR = "./browser_data"
INPUT_SELECTOR = 'textarea[placeholder="Message DeepSeek"]'
COMPLETION_GLOB = "**/api/v0/chat/completion"
REGENERATE_GLOB = "**/api/v0/chat/regenerate"
STREAM_TIMEOUT_S = 300.0
SEND_THINKING = True     # True -> thinking streamed as delta.reasoning_content
DEBUG_DUMP = False       # True -> write raw SSE data lines to stream_dump.jsonl

# Per-turn route state
_armed = asyncio.Event()
_current_queue: asyncio.Queue | None = None
_route_lock = asyncio.Lock()
_intercepted = False
_idle_task: asyncio.Task | None = None
_abort_event: asyncio.Event | None = None
_intercepted_response = None
_send_signature: str | None = None
_fragment_types: list[str] = []
_active_fragment_type: str | None = None

# ==================== Browser Setup ====================
async def startup():
    global browser, page
    p = await async_playwright().start()
    browser = await p.chromium.launch_persistent_context(
        user_data_dir=USER_DATA_DIR, headless=False
    )
    page = browser.pages[0] if browser.pages else await browser.new_page()
    await page.goto(DEEPSEEK_URL)

    print("\nPlease log in manually, then press Enter in the terminal...")
    await asyncio.to_thread(input)          # don't block the event loop
    print("Logged in. Registering route handler...")

    await _register_route()
    print("Bridge ready.\n")

async def shutdown():
    global browser
    if browser:
        await browser.close()
        print("Browser closed.")

# ==================== Fast paste + send ====================
SEND_SELECTOR = (
    "[role='button'].ds-button._52c986b:visible, "
    ".ds-button._52c986b.ds-button--circle:visible, "
    "div.ds-icon-button._52c986b:visible"
)

async def paste_and_send(text: str):
    global page
    input_element = page.locator(INPUT_SELECTOR)
    await input_element.click()
    await input_element.fill("")
    await input_element.fill(text)
    await asyncio.sleep(random.uniform(0.15, 0.35))
    await _remember_send_signature()
    await page.keyboard.press("Enter")

async def _remember_send_signature():
    global _send_signature
    try:
        btn = page.locator(SEND_SELECTOR).first
        if await btn.count() == 0:
            return
        _send_signature = await btn.evaluate(
            "(el) => (el.getAttribute('class')||'') + '|' + (el.getAttribute('aria-label')||'')"
        )
    except Exception:
        pass

async def _click_stop_button():
    """Click the Stop button (the Send control swaps to Stop while generating)."""
    try:
        btn = page.locator(SEND_SELECTOR).first
        if await btn.count() == 0:
            return
        signature = await btn.evaluate(
            "(el) => (el.getAttribute('class')||'') + '|' + (el.getAttribute('aria-label')||'')"
        )
        if _send_signature and signature == _send_signature:
            return                     # still send mode -> nothing to stop
        if not _send_signature:
            return                     # unknown baseline -> never risk a wrong click
        await btn.click(timeout=2000)
    except Exception:
        pass

# ==================== Idle simulation ====================
async def idle_actions(stop_event: asyncio.Event):
    """Perform tiny human-like actions until stop_event is set."""
    global page
    while not stop_event.is_set():
        action = random.choice(("wiggle", "scroll"))
        try:
            if action == "wiggle":
                vp = await page.evaluate("() => ({w: window.innerWidth, h: window.innerHeight})")
                await page.mouse.move(
                    random.randint(0, max(int(vp["w"]), 1)),
                    random.randint(0, max(int(vp["h"]), 1)),
                )
            else:
                await page.evaluate("window.scrollBy(0, 20)")
                await asyncio.sleep(0.3)
                await page.evaluate("window.scrollBy(0, -15)")
        except Exception:
            pass
        await asyncio.sleep(random.uniform(1.5, 4.0))

# ==================== Stream parsing ====================
def _expand_relative_ops(v, base):
    """Expand BATCH op lists: ops with relative paths get the base path prefixed."""
    out = []
    for item in v:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item_p = str(item.get("p") or "")
        if base and item_p and not item_p.startswith("response/") and not item_p.startswith("fragments/"):
            item["p"] = f"{base}/{item_p}"
        out.append(item)
    return out

def _handle_payload(obj: dict, queue) -> None:
    """Process one DeepSeek SSE payload into a list of ops, then interpret them.

    Handles every shape the reference driver (IntenseRP) handles:
      - single op:            {"p": ..., "o": ..., "v": ...}
      - BATCH op list:        {"p": "response", "o": "BATCH", "v": [op, op, ...]}
      - direct content:       {"v": "text"}                       (p is None)
      - inline first fragment:{"v": {"response": {"fragments": [...]}}}
    """
    p = obj.get("p")
    o = obj.get("o")
    v = obj.get("v")

    if "v" not in obj:
        return

    # Case 1: batch update, direct content, or inline fragment payload
    if p is None or (p == "response" and o == "BATCH"):
        if isinstance(v, list) and v and all(isinstance(x, dict) and "p" in x for x in v):
            ops = _expand_relative_ops(v, base=p)
        elif isinstance(v, str):
            # Direct content update (rare DeepSeek quirk) — attribute to active fragment
            _emit_text(v, _active_fragment_type, queue)
            return
        elif isinstance(v, dict):
            # Opening payload: DeepSeek sometimes inlines the first fragment here
            response_obj = v.get("response", v)
            if isinstance(response_obj, dict):
                fragments = response_obj.get("fragments")
                if isinstance(fragments, list) and fragments:
                    ops = [{"p": "response/fragments", "o": "APPEND", "v": fragments}]
            else:
                return
        else:
            return

    # Case 2: single path-based update (or op list under a path)
    else:
        if isinstance(v, list) and v and all(isinstance(x, dict) and "p" in x for x in v):
            ops = _expand_relative_ops(v, base=p) or [{"p": p, "o": o, "v": v}]
        else:
            ops = [{"p": p, "o": o, "v": v}]

    for op in ops:
        if not isinstance(op, dict):
            continue
        _handle_op(op.get("p"), op.get("o"), op.get("v"), queue)

def _handle_op(p, o, v, queue) -> None:
    global _active_fragment_type

    # Status updates (FINISHED / CONTENT_FILTER / ...) — NEVER content
    if p in ("status", "response/status"):
        return

    # Fragment append: record type by index, emit any inline content
    if isinstance(p, str) and (p == "fragments" or p == "response/fragments"
                               or p.endswith("/fragments")) and o == "APPEND" and isinstance(v, list):
        for frag in v:
            if isinstance(frag, dict):
                ftype = str(frag.get("type") or "").upper()
                _fragment_types.append(ftype)
                _active_fragment_type = ftype
                if "content" in frag:
                    _emit_text(str(frag.get("content") or ""), ftype, queue)
        return

    # Fragment content delta: response/fragments/0/content  OR  fragments/0/content
    if (isinstance(p, str)
            and (p.startswith("response/fragments/") or p.startswith("fragments/"))
            and p.endswith("/content")):
        _emit_text(str(v or ""), _type_for_path(p), queue)
        return

    # Fragment status paths, search ops, tool ops, etc. — nothing to emit
    return

def _type_for_path(p: str) -> str | None:
    """Resolve the fragment type for a content path, by fragment index."""
    try:
        parts = p.split("/")
        if len(parts) >= 4 and parts[0] == "response" and parts[1] == "fragments":
            idx = int(parts[2])                      # response/fragments/0/content
        elif len(parts) >= 3 and parts[0] == "fragments":
            idx = int(parts[1])                      # fragments/0/content
        else:
            return _active_fragment_type
    except (ValueError, IndexError):
        return _active_fragment_type

    if 0 <= idx < len(_fragment_types):
        return _fragment_types[idx]
    return _active_fragment_type                     # fallback to active type

def _emit_text(text, ftype, queue) -> None:
    if not text:
        return
    ftype = (ftype or "").upper()
    if ftype == "THINK":
        if SEND_THINKING:
            queue.put_nowait(_make_openai_reasoning_sse(text))
        return
    if ftype in ("SEARCH", "TOOL_SEARCH"):
        return
    queue.put_nowait(_make_openai_sse(text))         # RESPONSE / unknown -> answer

def _dump_line(data: str) -> None:
    if not DEBUG_DUMP:
        return
    try:
        with open("stream_dump.jsonl", "a", encoding="utf-8") as f:
            f.write(data + "\n")
    except Exception:
        pass

# ==================== Route handler (registered once) ====================
async def _register_route():
    global _route_registered, page

    async def handle_route(route):
        global _intercepted, _current_queue, _intercepted_response
        if not _armed.is_set():
            await route.continue_()
            return

        async with _route_lock:
            if _intercepted:
                await route.continue_()
                return
            _intercepted = True

        request = route.request
        headers = await request.all_headers()
        headers.pop("content-length", None)
        headers.pop("host", None)
        cookies = {c["name"]: c["value"] for c in await browser.cookies()}
        body = request.post_data_json
        if body is None:
            _intercepted = False
            await route.continue_()
            return

        full_body = bytearray()
        resp_headers = {}
        response_status = 200
        done_seen = False
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", request.url, headers=headers,
                    cookies=cookies, json=body, timeout=STREAM_TIMEOUT_S
                ) as resp:
                    _intercepted_response = resp
                    response_status = resp.status_code
                    for k, v in resp.headers.items():
                        resp_headers[k] = v

                    async for chunk in resp.aiter_bytes():
                        if _abort_event is not None and _abort_event.is_set():
                            break
                        full_body.extend(chunk)
                        buffer += decoder.decode(chunk)          # incremental UTF-8
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r").strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done_seen = True
                                break
                            _dump_line(data)
                            try:
                                obj = json.loads(data)
                                _handle_payload(obj, _current_queue)
                            except Exception:
                                pass
                        if done_seen:
                            break

                    # Flush the tail (final line often has no trailing newline)
                    if not done_seen:
                        buffer += decoder.decode(b"", final=True)
                        tail = buffer.strip()
                        if tail.startswith("data:"):
                            data = tail[5:].strip()
                            if data and data != "[DONE]":
                                _dump_line(data)
                                try:
                                    obj = json.loads(data)
                                    _handle_payload(obj, _current_queue)
                                except Exception:
                                    pass
        except Exception as e:
            print(f"[Bridge] Interception error: {e}")
            if _current_queue is not None:
                await _current_queue.put(_make_error_sse(f"Interception error: {e}"))
        finally:
            _intercepted_response = None
            if _current_queue is not None:
                await _current_queue.put(None)
            _intercepted = False

        if _abort_event is not None and _abort_event.is_set():
            await _click_stop_button()
            try:
                await route.abort()
            except Exception:
                pass
        else:
            await route.fulfill(body=bytes(full_body), status=response_status, headers=resp_headers)

    await page.route(COMPLETION_GLOB, handle_route)
    await page.route(REGENERATE_GLOB, handle_route)
    _route_registered = True

# ==================== stream_response (called by server.py) ====================
async def stream_response(prompt: str):
    global _current_queue, _armed, _intercepted, _idle_task, _abort_event
    global _fragment_types, _active_fragment_type
    async with process_lock:
        _current_queue = asyncio.Queue()
        _intercepted = False
        _abort_event = asyncio.Event()
        _fragment_types = []
        _active_fragment_type = None
        _armed.set()

        idle_stop = asyncio.Event()
        mini_buffer = ""
        try:
            # Send the prompt
            await paste_and_send(prompt)

            # Start idle simulation
            _idle_task = asyncio.create_task(idle_actions(idle_stop))

            # Collect small chunks into a buffer before yielding
            while True:
                item = await _current_queue.get()
                if item is None:
                    if mini_buffer:
                        yield _make_openai_sse(mini_buffer)
                    break
                parsed = _parse_sse(item)
                if not parsed:
                    continue
                if "error" in parsed:
                    raise RuntimeError(parsed["error"])
                if parsed["reasoning"]:
                    yield _make_openai_reasoning_sse(parsed["reasoning"])
                if parsed["content"]:
                    mini_buffer += parsed["content"]
                    if len(mini_buffer) >= 20 or "\n" in mini_buffer:
                        yield _make_openai_sse(mini_buffer)
                        mini_buffer = ""
        except asyncio.CancelledError:
            # Client disconnected -> abort the provider stream
            await _abort_provider_stream()
            raise
        except Exception as e:
            print(f"[Bridge] stream_response error: {e}")
            await _abort_provider_stream()
            raise
        finally:
            idle_stop.set()
            if _idle_task:
                try:
                    await _idle_task
                except Exception:
                    pass
            _armed.clear()

async def _abort_provider_stream():
    global _intercepted_response
    if _abort_event:
        _abort_event.set()
    resp = _intercepted_response
    if resp is not None:
        try:
            await resp.aclose()
        except Exception:
            pass

# ==================== SSE helpers ====================
def _make_openai_sse(text: str, model: str = "deepseek-chat") -> str:
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    }
    return f"data: {json.dumps(payload)}\n\n"

def _make_openai_reasoning_sse(text: str, model: str = "deepseek-chat") -> str:
    """Thinking chunk — uses delta.reasoning_content. Verified: Continue's
    fromChatCompletionChunk() maps reasoning_content -> { role: "thinking" } and
    renders it as a separate thinking block (continuedev/continue PR #6236)."""
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]
    }
    return f"data: {json.dumps(payload)}\n\n"

def _make_error_sse(message: str) -> str:
    payload = {"error": {"message": message, "type": "internal_error"}}
    return f"data: {json.dumps(payload)}\n\n"

def _parse_sse(sse_string: str) -> dict | None:
    """Parse a bridge SSE chunk into {'content':..., 'reasoning':...} or {'error':...}."""
    if not sse_string.startswith("data: "):
        return None
    try:
        obj = json.loads(sse_string[6:])
    except Exception:
        return None
    if "error" in obj:
        err = obj["error"]
        msg = err.get("message", "stream error") if isinstance(err, dict) else str(err)
        return {"error": str(msg)}
    try:
        delta = obj["choices"][0]["delta"]
    except Exception:
        return None
    return {
        "content": delta.get("content", "") or "",
        "reasoning": delta.get("reasoning_content", "") or "",
    }
