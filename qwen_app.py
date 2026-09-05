"""
qwen_app.py – Qwen AI Bridge (route interception + human idle + clean API output)

Same architecture as the DeepSeek app.py bridge. The flow:
  1. User runs qwen_server.py -> a real browser opens chat.qwen.ai (persistent profile).
  2. User logs in manually once (phone/email login) -> session cookie saved.
  3. Continue (or any OpenAI client) sends a request to /v1/chat/completions.
  4. The prompt is typed into the Qwen textarea and sent like a human (with
     random typing delay + idle mouse/scroll simulation).
  5. Playwright intercepts the browser's OWN POST to the Qwen backend completion
     API:  POST https://chat.qwen.ai/api/v2/chat/completions
  6. The exact request is re-issued with httpx as a real streaming call; bytes
     arrive live from Qwen's SSE stream.
  7. The parser converts each SSE payload into OpenAI-compatible chunks:
       - phase="thinking_summary"  -> delta.reasoning_content (Continue shows it
         as a separate thinking block)
       - phase="answer"            -> delta.content (the final answer)
       - phase="web_search"/tools  -> ignored
  8. Chunks stream to the client; the full captured body is replayed into the
     page (route.fulfill) so the on-screen chat completes normally.

Selectors verified from the live Qwen page:
  INPUT  -> <textarea class="message-input-textarea" placeholder="Ask Qwen">
  SEND   -> <div class="chat-prompt-send-button"><button class="send-button"
            aria-label="Send">
"""
import asyncio
import codecs
import json
import random
import time
import httpx
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ==================== Globals ====================
browser = None
page = None
process_lock = asyncio.Lock()
_route_registered = False

QWEN_URL = "https://chat.qwen.ai"
USER_DATA_DIR = "./qwen_browser_data"
INPUT_SELECTOR = "textarea.message-input-textarea"
INPUT_FALLBACK_SELECTOR = 'textarea[placeholder="Ask Qwen"]'
SEND_SELECTOR = "div.chat-prompt-send-button button"
SEND_FALLBACK_SELECTOR = "button[aria-label='Send']"
COMPLETION_GLOB = "**/api/v2/chat/completions*"
STREAM_TIMEOUT_S = 300.0
SEND_THINKING = True     # True -> thinking streamed as delta.reasoning_content
DEBUG_DUMP = False       # True -> write raw SSE data lines to DUMP_FILE
DUMP_FILE = "qwen_stream_dump.jsonl"

# Per-turn route state
_armed = asyncio.Event()
_current_queue: asyncio.Queue | None = None
_route_lock = asyncio.Lock()
_intercepted = False
_idle_task: asyncio.Task | None = None
_abort_event: asyncio.Event | None = None
_intercepted_response = None
_send_signature: str | None = None
_thinking_emitted: str = ""   # incremental tracking for repeated summary payloads
_stream_error: bool = False   # once an error payload arrives, stop processing

# ==================== Browser Setup ====================
# --- Loading-glitch hardening (same fix as app.py) ---
# Navigate with wait_until="domcontentloaded" (never "load"/"networkidle")
# and give every Playwright wait a generous 15 s ceiling.
NAV_TIMEOUT_MS = 15_000          # safe ceiling for goto / clicks / fills
GOTO_WAIT_UNTIL = "domcontentloaded"

async def _goto_chat_page(page, url: str):
    """page.goto() that survives endless-spinner glitches: waits for DOM
    readiness only, retries once on timeout, and never crashes at startup."""
    for attempt in (1, 2):
        try:
            await page.goto(
                url,
                wait_until=GOTO_WAIT_UNTIL,   # DOM ready only – NOT "load"/"networkidle"
                timeout=NAV_TIMEOUT_MS,       # 15 s instead of 3 s
            )
            return
        except PlaywrightTimeoutError:
            if attempt == 1:
                print("[startup] navigation timed out (spinner glitch?) – retrying...")
            else:
                print(f"[startup] {url} still not settling – continuing anyway.")

async def _wait_for_input_ready(page, timeout_s: int = 60):
    """Wait until the chat textarea exists in the DOM (best effort, never raises)."""
    try:
        await page.wait_for_selector(
            INPUT_SELECTOR, state="attached", timeout=timeout_s * 1000
        )
    except Exception:
        print("[startup] chat input not detected yet – continuing anyway.")

async def startup():
    global browser, page
    p = await async_playwright().start()
    browser = await p.chromium.launch_persistent_context(
        user_data_dir=USER_DATA_DIR, headless=False
    )
    page = browser.pages[0] if browser.pages else await browser.new_page()

    # Raise the default ceiling for EVERY action/navigation on this page.
    page.set_default_timeout(NAV_TIMEOUT_MS)
    page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

    await _goto_chat_page(page, QWEN_URL)       # domcontentloaded + retry
    await _wait_for_input_ready(page)           # tolerant of the spinner

    print("\nPlease log in manually (Qwen login: phone/email), then press Enter in the terminal...")
    await asyncio.to_thread(input)
    print("Logged in. Registering route handler...")

    await _register_route()
    print("Bridge ready.\n")

async def shutdown():
    global browser
    if browser:
        await browser.close()
        print("Browser closed.")

# ==================== Human-like paste + send ====================
async def paste_and_send(text: str):
    global page
    input_element = page.locator(INPUT_SELECTOR)
    if await input_element.count() == 0:
        input_element = page.locator(INPUT_FALLBACK_SELECTOR)
    await input_element.click()
    await input_element.fill("")
    await input_element.fill(text)
    await asyncio.sleep(random.uniform(0.15, 0.35))
    await _remember_send_signature()
    await page.keyboard.press("Enter")

async def _locate_send_button():
    btn = page.locator(SEND_SELECTOR).first
    if await btn.count() == 0:
        btn = page.locator(SEND_FALLBACK_SELECTOR).first
    return btn

async def _remember_send_signature():
    global _send_signature
    try:
        btn = await _locate_send_button()
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
        btn = await _locate_send_button()
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

# ==================== Qwen stream parsing ====================
def _extract_qwen_summary_parts(extra: dict, key: str) -> list[str]:
    value = extra.get(key)
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []

def _extract_thinking_summary(delta: dict) -> str:
    """Join delta.extra.summary_title[] / summary_thought[] like the reference driver."""
    extra = delta.get("extra")
    if not isinstance(extra, dict):
        return ""
    titles = _extract_qwen_summary_parts(extra, "summary_title")
    thoughts = _extract_qwen_summary_parts(extra, "summary_thought")
    section_count = max(len(titles), len(thoughts))
    sections: list[str] = []
    for idx in range(section_count):
        title = titles[idx] if idx < len(titles) else ""
        thought = thoughts[idx] if idx < len(thoughts) else ""
        if title and thought:
            sections.append(f"{title}\n{thought}")
        elif title or thought:
            sections.append(title or thought)
    return "\n\n".join(s for s in sections if s)

def _handle_payload(obj: dict, queue) -> None:
    """Process one Qwen SSE payload: {"choices": [{"delta": {...}}]}."""
    global _thinking_emitted, _stream_error

    if _stream_error:
        return   # an error already occurred; ignore everything after it

    # Provider-side error payloads
    if "error" in obj:
        err = obj["error"]
        msg = err.get("message", "Qwen stream error") if isinstance(err, dict) else str(err)
        queue.put_nowait(_make_error_sse(str(msg)))
        _stream_error = True
        return

    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return
    delta = choice0.get("delta")
    if not isinstance(delta, dict):
        return

    phase = str(delta.get("phase") or "").strip().lower()
    role = str(delta.get("role") or "").strip().lower()
    status = str(delta.get("status") or "").strip().lower()

    # Ignore tool-call chatter + raw search results for stability
    if (phase == "web_search"
            or role == "function"
            or delta.get("function_call") is not None
            or delta.get("tool_calls") is not None):
        return

    if phase == "thinking_summary":
        summary_text = _extract_thinking_summary(delta)
        if not summary_text:
            return
        if SEND_THINKING:
            # Qwen repeats the whole summary each event; emit only the missing suffix
            if summary_text.startswith(_thinking_emitted):
                missing = summary_text[len(_thinking_emitted):]
            else:
                missing = summary_text
            if missing:
                queue.put_nowait(_make_openai_reasoning_sse(missing))
                _thinking_emitted += missing
        return

    if phase == "answer":
        content = delta.get("content")
        if isinstance(content, str) and content:
            queue.put_nowait(_make_openai_sse(content))
        if status == "finished":
            queue.put_nowait(_make_openai_finish_sse())
        return

    # Unknown phases: ignore

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
                            if DEBUG_DUMP:
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
                                if DEBUG_DUMP:
                                    _dump_line(data)
                                try:
                                    obj = json.loads(data)
                                    _handle_payload(obj, _current_queue)
                                except Exception:
                                    pass
        except Exception as e:
            print(f"[Qwen Bridge] Interception error: {e}")
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
    _route_registered = True

# ==================== stream_response (called by qwen_server.py) ====================
async def stream_response(prompt: str):
    global _current_queue, _armed, _intercepted, _idle_task, _abort_event
    global _thinking_emitted, _stream_error
    async with process_lock:
        _current_queue = asyncio.Queue()
        _intercepted = False
        _abort_event = asyncio.Event()
        _thinking_emitted = ""
        _stream_error = False
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
            print(f"[Qwen Bridge] stream_response error: {e}")
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
def _make_openai_sse(text: str, model: str = "qwen-auto") -> str:
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    }
    return f"data: {json.dumps(payload)}\n\n"

def _make_openai_reasoning_sse(text: str, model: str = "qwen-auto") -> str:
    """Thinking chunk — delta.reasoning_content (Continue renders it as thinking)."""
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]
    }
    return f"data: {json.dumps(payload)}\n\n"

def _make_openai_finish_sse(model: str = "qwen-auto") -> str:
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
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

def _dump_line(data: str) -> None:
    try:
        with open(DUMP_FILE, "a", encoding="utf-8") as f:
            f.write(data + "\n")
    except Exception:
        pass