"""
app.py – DeepSeek Bridge (route interception + human idle + clean API output)
"""
import asyncio
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

# Per‑turn route state
_armed = asyncio.Event()
_current_queue: asyncio.Queue | None = None
_route_lock = asyncio.Lock()
_intercepted = False
_idle_task: asyncio.Task | None = None

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
    input()
    print("Logged in. Registering route handler...")

    await _register_route()
    print("Bridge ready.\n")

async def shutdown():
    global browser
    if browser:
        await browser.close()
        print("Browser closed.")

# ==================== Fast paste + send ====================
async def paste_and_send(text: str):
    global page
    input_element = page.locator(INPUT_SELECTOR)
    await input_element.click()
    await input_element.fill("")
    await input_element.fill(text)
    await asyncio.sleep(random.uniform(0.15, 0.35))
    await page.keyboard.press("Enter")

# ==================== Idle simulation ====================
async def idle_actions(stop_event: asyncio.Event):
    """Perform tiny human‑like actions until stop_event is set."""
    global page
    while not stop_event.is_set():
        action = random.choice(("wiggle", "scroll"))
        try:
            if action == "wiggle":
                pos = await page.evaluate(
                    "() => ({x: window.mouseX || 500, y: window.mouseY || 300})"
                )
                new_x = pos["x"] + random.randint(-20, 20)
                new_y = pos["y"] + random.randint(-10, 10)
                await page.mouse.move(new_x, new_y)
            else:
                await page.evaluate("window.scrollBy(0, 20)")
                await asyncio.sleep(0.3)
                await page.evaluate("window.scrollBy(0, -15)")
        except Exception:
            pass
        await asyncio.sleep(random.uniform(1.5, 4.0))

# ==================== Route handler (registered once) ====================
async def _register_route():
    global _route_registered, page

    async def handle_route(route):
        global _intercepted, _current_queue
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

        full_body = bytearray()
        resp_headers = {}
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", request.url, headers=headers,
                    cookies=cookies, json=body, timeout=60.0
                ) as resp:
                    for k, v in resp.headers.items():
                        resp_headers[k] = v
                    buffer = ""
                    active_fragment = None

                    async for chunk in resp.aiter_bytes():
                        full_body.extend(chunk)
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r")
                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(data)
                                    new_type = _detect_fragment_type(obj)
                                    if new_type is not None:
                                        active_fragment = new_type
                                    # Only extract and send content if we're NOT in a THINK fragment
                                    if active_fragment != "THINK":
                                        content = _extract_content(obj)
                                        if content:
                                            await _current_queue.put(_make_openai_sse(content))
                                except Exception:
                                    pass
        except Exception as e:
            print(f"[Bridge] Interception error: {e}")
            await _current_queue.put(_make_error_sse(f"Interception error: {e}"))
        finally:
            await _current_queue.put(None)
            _intercepted = False

        await route.fulfill(body=bytes(full_body), status=200, headers=resp_headers)

    await page.route(COMPLETION_GLOB, handle_route)
    await page.route(REGENERATE_GLOB, handle_route)
    _route_registered = True

# ==================== stream_response (called by server.py) ====================
async def stream_response(prompt: str):
    global _current_queue, _armed, _intercepted, _idle_task, page
    async with process_lock:
        _current_queue = asyncio.Queue()
        _intercepted = False
        _armed.set()

        # Send the prompt
        await paste_and_send(prompt)

        # Start idle simulation
        idle_stop = asyncio.Event()
        _idle_task = asyncio.create_task(idle_actions(idle_stop))

        # Collect small chunks into a buffer before yielding
        mini_buffer = ""
        while True:
            item = await _current_queue.get()
            if item is None:
                if mini_buffer:
                    yield _make_openai_sse(mini_buffer)
                break
            parsed = _parse_sse(item)
            if parsed:
                mini_buffer += parsed
                if len(mini_buffer) >= 20 or "\n" in mini_buffer:
                    yield _make_openai_sse(mini_buffer)
                    mini_buffer = ""

        # Stop idle simulation
        idle_stop.set()
        if _idle_task:
            await _idle_task
        _armed.clear()

# ==================== SSE helpers ====================
def _detect_fragment_type(obj: dict) -> str | None:
    p = obj.get("p")
    o = obj.get("o")
    v = obj.get("v")
    if isinstance(p, str) and p.endswith("/fragments") and o == "APPEND" and isinstance(v, list):
        for frag in v:
            if isinstance(frag, dict) and "type" in frag:
                return str(frag["type"]).upper()
    return None

def _extract_content(obj: dict) -> str | None:
    p = obj.get("p")
    o = obj.get("o")
    v = obj.get("v")
    if p is None and o is None and isinstance(v, str):
        return v
    if isinstance(p, str) and p.endswith("/content") and isinstance(v, str):
        if o in ("ADD", "SET"):
            return v
    if isinstance(p, str) and p.endswith("/fragments") and o == "APPEND" and isinstance(v, list):
        parts = []
        for frag in v:
            if isinstance(frag, dict) and "content" in frag:
                parts.append(str(frag["content"] or ""))
        if parts:
            return "".join(parts)
    return None

def _make_openai_sse(text: str, model: str = "deepseek-chat") -> str:
    payload = {
        "id": "chatcmpl-" + str(int(time.time() * 1000)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    }
    return f"data: {json.dumps(payload)}\n\n"

def _make_error_sse(message: str) -> str:
    payload = {"error": {"message": message, "type": "internal_error"}}
    return f"data: {json.dumps(payload)}\n\n"

def _parse_sse(sse_string: str) -> str | None:
    if not sse_string.startswith("data: "):
        return None
    try:
        obj = json.loads(sse_string[6:])
        return obj["choices"][0]["delta"].get("content", "")
    except Exception:
        return None