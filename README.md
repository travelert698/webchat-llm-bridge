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

3. In the console, choose prompt method **1** (full conversation) or **2** (last user message only). Then enter the local provider submissions-per-minute limit. Press Enter to use the conservative default of **5**. A visible browser opens; log in manually and confirm in the server console.

4. Configure your client:

   - DeepSeek OpenAI base URL: `http://127.0.0.1:8000/v1`
   - Qwen OpenAI base URL: `http://127.0.0.1:8001/v1`
   - Anthropic Messages base URL: the corresponding server root, without `/v1`

For both backends, run `python run_both.py`; it asks for one prompt mode and separate DeepSeek/Qwen RPM values. Full setup, environment overrides, compatibility limits, and safety notes are in [`CLAUDE_CODE_SETUP.md`](CLAUDE_CODE_SETUP.md).

## Queue and local rate limit

The generation endpoints are FIFO-queued, one active browser-backed generation at a time. Every outbound model submission—including tool-repair attempts—is paced against the selected RPM. Up to 100 requests may wait by default; when full, the API returns a structured 429 instead of accumulating an unbounded backlog. Check `GET /queue/status` on either server.

The value you enter is a **local pacing cap**, not an official provider quota. DeepSeek/Qwen web-chat limits can differ from their developer API quotas and may be unpublished or account-specific. The queue does not bypass throttling or provider controls. See [`QUEUE_RATE_LIMIT_RESEARCH.md`](QUEUE_RATE_LIMIT_RESEARCH.md).

## Privacy and use

The local APIs have no authentication and bind to loopback by default; do not expose them to a network. Prompts are sent through the provider website using your logged-in browser session. Keep browser profiles private, use non-sensitive test data, and retain the downstream tool approval policy. Provider compatibility and account-safety cannot be guaranteed.

## Offline checks

```powershell
python test_parser.py
python test_tool_bridge.py
python test_anthropic_compat.py
python test_api_smoke.py
python test_request_queue.py
python test_startup_config.py
```

The tests use fake streams and do not verify live logins, provider website behavior, or actual account-specific web limits.
