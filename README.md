# WebChat LLM Bridge

A local OpenAI Chat Completions / minimal Anthropic Messages adapter for the existing **visible-browser** DeepSeek Web and Qwen Web automation. It preserves the browser/UI workflow; it is not a direct-provider-API client or a local model.

## Quick start

1. Install Python 3.10+ and the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   python -m playwright install chromium
   ```

2. Start one backend:

   ```powershell
   python server.py       # DeepSeek, default port 8000
   # or
   python qwen_server.py  # Qwen, default port 8001
   ```

3. In the console, choose prompt method **1** (full conversation) or **2** (last user message only). Then enter the cooldown in seconds. Press Enter for the default of **3 seconds**. A visible browser opens; log in manually and confirm in the server console.

4. Configure your client:

   - DeepSeek OpenAI base URL: `http://127.0.0.1:8000/v1`
   - Qwen OpenAI base URL: `http://127.0.0.1:8001/v1`
   - Anthropic Messages base URL: the corresponding server root, without `/v1`

For both backends, run `python run_both.py`; it asks for one prompt mode and separate DeepSeek/Qwen cooldowns in seconds. Full setup, environment overrides, compatibility limits, and safety notes are in [`CLAUDE_CODE_SETUP.md`](CLAUDE_CODE_SETUP.md).

## Queue and completion cooldown

The generation endpoints are FIFO-queued, one active browser-backed generation at a time. The server holds each provider stream until it finishes, then enforces the chosen cooldown before the next provider submission. It also starts the client-request cooldown only after the entire response (including SSE streaming) has been sent. Up to 100 client requests may wait by default; when full, the API returns a structured 429 instead of accumulating an unbounded backlog. Check `GET /queue/status` on either server.

The delay is a **local pacing choice**, not an official provider quota. DeepSeek/Qwen web-chat limits can differ from their developer API quotas and may be unpublished or account-specific. The cooldown does not bypass provider throttling or controls. See [`QUEUE_RATE_LIMIT_RESEARCH.md`](QUEUE_RATE_LIMIT_RESEARCH.md).

The browser adapters replay the provider's captured SSE response to the visible page. HTTPX may decompress that body, so the adapters remove stale compression/framing headers before replay; this prevents a decoded body from being mislabeled as compressed or given the wrong byte length. The server console logs only safe replay metadata (status, type, encoding, byte count, completion marker, and fulfilled/aborted outcome), never prompts or cookies.

## Privacy and use

The local APIs have no authentication and bind to loopback by default; do not expose them to a network. Prompts are sent through the provider website using your logged-in browser session. Keep browser profiles private, use non-sensitive test data, and retain the downstream tool approval policy. Provider compatibility and account-safety cannot be guaranteed.

## Offline checks

```powershell
python test_parser.py
python test_tool_bridge.py
python test_anthropic_compat.py
python test_api_smoke.py
python test_browser_response.py
python test_request_queue.py
python test_startup_config.py
```

The tests use fake streams and do not verify live logins, provider website behavior, or actual account-specific web limits.
