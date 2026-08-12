"""
test_interception.py – Route interception with human‑idle behaviour
(terminal streams live, web UI updated after completion but looks active)
"""
import asyncio
import json
import random
import httpx
from playwright.async_api import async_playwright

DEEPSEEK_URL = "https://chat.deepseek.com"
USER_DATA_DIR = "./browser_data"
INPUT_SELECTOR = 'textarea[placeholder="Message DeepSeek"]'

async def paste_and_send(page, text: str):
    input_element = page.locator(INPUT_SELECTOR)
    await input_element.click()
    await input_element.fill("")
    await input_element.fill(text)
    await asyncio.sleep(random.uniform(0.15, 0.35))
    await page.keyboard.press("Enter")

# ----- Idle simulation helpers -----
async def idle_actions(page, stop_event):
    """Perform random, tiny human‑like actions until stop_event is set."""
    while not stop_event.is_set():
        action = random.choice(("wiggle", "scroll"))
        try:
            if action == "wiggle":
                # Move mouse a few pixels from current position
                pos = await page.evaluate("() => ({x: window.mouseX || 500, y: window.mouseY || 300})")
                new_x = pos["x"] + random.randint(-20, 20)
                new_y = pos["y"] + random.randint(-10, 10)
                await page.mouse.move(new_x, new_y)
            else:
                # Scroll a tiny amount and back
                await page.evaluate("window.scrollBy(0, 20)")
                await asyncio.sleep(0.3)
                await page.evaluate("window.scrollBy(0, -15)")
        except Exception:
            pass
        await asyncio.sleep(random.uniform(1.5, 4.0))

# ----- Content helpers (unchanged) -----
def extract_content(obj: dict) -> str | None:
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

def detect_fragment_type(obj: dict) -> str | None:
    p = obj.get("p")
    o = obj.get("o")
    v = obj.get("v")
    if isinstance(p, str) and p.endswith("/fragments") and o == "APPEND" and isinstance(v, list):
        for frag in v:
            if isinstance(frag, dict) and "type" in frag:
                return str(frag["type"]).upper()
    return None

async def main():
    print("Starting browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR, headless=False
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()
        await page.goto(DEEPSEEK_URL)
        print("Log in manually, then press Enter...")
        input()

        print("\n" + "=" * 60)
        print("  Terminal Chat – Interception + human idle")
        print("  Type 'exit' to quit.")
        print("=" * 60 + "\n")

        # ----- Shared route state -----
        armed = asyncio.Event()
        current_queue: asyncio.Queue | None = None
        route_lock = asyncio.Lock()
        intercepted = False

        async def handle_route(route):
            nonlocal intercepted
            if not armed.is_set():
                await route.continue_()
                return

            async with route_lock:
                if intercepted:
                    await route.continue_()
                    return
                intercepted = True

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
                        thinking_open = False

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
                                        new_type = detect_fragment_type(obj)
                                        if new_type is not None:
                                            if thinking_open and new_type != "THINK":
                                                await current_queue.put(("tag", "</think>"))
                                                thinking_open = False
                                            active_fragment = new_type
                                        content = extract_content(obj)
                                        if content:
                                            if active_fragment == "THINK":
                                                if not thinking_open:
                                                    await current_queue.put(("tag", "<think>"))
                                                    thinking_open = True
                                                await current_queue.put(("text", content))
                                            else:
                                                await current_queue.put(("text", content))
                                    except Exception:
                                        pass
                        if thinking_open:
                            await current_queue.put(("tag", "</think>"))
            except Exception as e:
                print(f"\n[Error] {e}")
            finally:
                await current_queue.put(("done", None))
                intercepted = False

            # Fulfill the pending request – now the web UI will update instantly
            await route.fulfill(body=bytes(full_body), status=200, headers=resp_headers)

        await page.route("**/api/v0/chat/completion", handle_route)

        # ----- Chat loop -----
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() == "exit":
                break
            if not prompt:
                continue

            # Reset state
            current_queue = asyncio.Queue()
            intercepted = False
            armed.set()

            # Send the prompt
            await paste_and_send(page, prompt)

            # Start idle simulation (mouse wiggles, scrolling) while we wait
            idle_stop = asyncio.Event()
            idle_task = asyncio.create_task(idle_actions(page, idle_stop))

            # Print streamed response
            print("AI: ", end="", flush=True)
            while True:
                kind, value = await current_queue.get()
                if kind == "done":
                    break
                print(value, end="", flush=True)

            # Stop idle simulation
            idle_stop.set()
            await idle_task
            print("\n")
            armed.clear()

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())