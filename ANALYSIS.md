# Initial repository review: `webchat-llm-bridge` (pre-patch baseline)

> This is the historical review from before the Anthropic Messages compatibility work. The current implementation adds `anthropic_compat.py`, updates both server files and `tool_bridge.py`, and adds `jsonschema` to `requirements.txt`. Tool calling received a further reference-driven review; see `REFERENCE_REVIEW.md` for current changes, checks, and limitations. See `CLAUDE_CODE_SETUP.md` for the current install/run guide. Findings below describe the pre-patch baseline unless explicitly noted.

**Review date:** 2026-10-26 (Asia/Colombo)  
**Repository:** `travelert698/webchat-llm-bridge`  
**Reviewed revision:** `372c049ccb7e500241a03df644ed186882065adf` (`main`, 2026-09-05)

## Executive summary

This is a small Python prototype that exposes local OpenAI-shaped endpoints for DeepSeek Web and Qwen Web. It opens a visible Chromium session, expects a person to log in, types a prompt into the provider website, intercepts that site's authenticated completion request, and replays it through HTTPX. A separate module attempts to turn model-emitted text into OpenAI tool calls.

**Recommendation: do not treat this as production-ready or expose it to a network.** The source review found no obvious hidden installer, `eval`/`exec`, or arbitrary shell execution. `run_both.py` starts the two fixed local Python scripts. However, the program intentionally sends conversation data through the logged-in provider website, stores browser-session data locally, has no API authentication, and has important tool-call validation and parser defects. Use only for isolated experimentation with non-sensitive prompts until those issues are fixed. Official provider APIs are a safer and more stable choice for production.

## What was installed and what was not

- Cloned the repository into `/home/user/webchat-llm-bridge`.
- Installed `requirements.txt` into an **isolated project virtual environment** (`.venv`), not into the assistant host OS or your personal computer. The requirements are unpinned; this sandbox resolved FastAPI 0.162.0, Uvicorn 0.54.0, Playwright 1.64.1, Pydantic 2.14.1, and HTTPX 0.28.1.
- Did **not** install Playwright's Chromium browser, open a provider website, sign in, or send a live prompt. Those steps require an actual browser runtime and your own account/session. So the web automation and provider integration remain unverified end-to-end.
- No source files were changed; `ANALYSIS.md` is the only added deliverable. The local `.venv` and cache artifacts are ignored/excluded from workspace snapshots, so they should not be considered a portable installation.

## What the project contains

| File | Purpose |
|---|---|
| `app.py` | DeepSeek Web automation, SSE parsing, local response streaming |
| `qwen_app.py` | Qwen Web automation and SSE parsing |
| `server.py` | DeepSeek OpenAI-style API, normally port 8000 |
| `qwen_server.py` | Qwen OpenAI-style API, normally port 8001 |
| `tool_bridge.py` | Converts model text into OpenAI-style tool-call responses |
| `run_both.py` | Starts both local servers as child processes |
| `test_parser.py` | Small mocked tests for the DeepSeek and Qwen provider parsers |
| `requirements.txt` | Unpinned Python runtime dependencies |

The expected runtime is interactive: a visible browser opens and waits for manual login. Browser profiles (`browser_data/` and `qwen_browser_data/`) retain provider login state. `.gitignore` excludes those directories and optional raw stream dumps, which is good, but the profile directories still contain sensitive session data on disk.

## Security and privacy findings

### 1. Tool calls are not checked against the request's allowed tools — **high priority**

`tool_bridge.py` describes a tool whitelist and filters tool descriptions sent to the model. But after parsing a returned call, it does not verify that the function name was included in the current request's `tools`, nor validate arguments against that function's JSON schema. A direct probe showed that a `shell_exec` tool name is parsed even though it is not in the whitelist. The bridge itself does not execute that call; it returns it to the calling agent/client, which may execute it. **Do not rely on the bridge's whitelist as a security boundary.** Validate every name and argument at the bridge/client boundary before execution.

The injected prompt also tells the model not to ask the user for confirmation. A downstream tool runner must retain its own permission checks and confirmation policy; it must not trust that prompt.

### 2. Local API has no authentication — **medium priority**

The servers bind to `127.0.0.1` by default, which limits ordinary network exposure, but endpoints have no API-token check, rate limit, or meaningful request-size limit. Treat them as accessible to other local processes. Do not change the bind address or put them behind a network/reverse proxy without adding authentication and access controls.

### 3. Request bodies and provider session data are sensitive — **medium priority**

- On request-validation failures, both server modules print the entire request body and include it in the 422 response. This can put prompts, private documents, or other user-supplied data in logs and error output.
- Prompts and chat history are deliberately sent to DeepSeek or Qwen using the logged-in browser session. This is not a local-only inference path.
- Enabling `DEBUG_DUMP` writes raw provider SSE data to disk. It is disabled by default, but dumps can contain private response content and should be treated as sensitive.

Remove raw-body logging/echoing or redact it, and make debug dumps opt-in with explicit retention and permissions.

### 4. Web automation is brittle and may have provider-policy implications

The implementation relies on provider page selectors and private web-app completion endpoints, and includes random delays plus mouse/scroll activity described as human-like. Provider changes can break it, and this automation may conflict with provider terms or trigger account protections. Verify the applicable terms before using an account; an official API is more stable.

### 5. No obvious arbitrary-code execution found in the reviewed source

The `<run_code>` handling in `tool_bridge.py` parses strings with regular expressions; it does not execute JavaScript or Python. The only subprocess use found is `run_both.py` launching `server.py` and `qwen_server.py` from the project directory. This is a source-level observation, not a guarantee that dependencies or future revisions are safe.

## Correctness and reliability findings

### Tool-call parser limitations (reproduced)

- **Split markers are missed.** If a stream chunk ends with `<tool_` and the next starts with `call>`, the first chunk is emitted as ordinary text and the tool call is not recognized.
- **Only the first tagged tool block is accepted.** `Sieve` marks itself done after one `<tool_call>`/`<run_code>` block, so later separate call blocks are ignored.
- **Bare JSON is not actually detected by the stream sieve.** There is a JSON parser helper, but the sieve never classifies untagged JSON as a JSON tool block; a JSON call is returned as assistant text.
- **Nested JavaScript objects can be truncated.** The regular-expression parser for `tools.NAME({...})` stops at the first closing brace; a nested argument probe returned a malformed/truncated value.
- `tool_choice="required"` is interpreted as though `required` were a function name, rather than as the OpenAI request to choose any available tool.

These are functional defects as well as safety concerns. Add a streaming-safe parser and enforce the exact tool set and schema supplied on each request.

### Browser/network response handling

- The response handler buffers the provider's entire stream in memory before fulfilling the browser route, while also queuing chunks for the API client. There is no explicit total-response size cap or queue back-pressure.
- `STREAM_TIMEOUT_S` applies once HTTPX is reading a provider response. If the browser action fails or the expected completion request is never intercepted, `stream_response()` can wait indefinitely on the queue; there is no timeout around that wait.
- The code reads with `httpx.Response.aiter_bytes()` (decoded bytes) but forwards the provider response headers unchanged. For compressed responses, stale `content-encoding` or `content-length` headers can make the fulfilled browser response inconsistent with its body.
- DeepSeek provider error/status payloads are not handled as comprehensively as Qwen's; some errors can become an empty or incomplete answer.
- Tool-mode handling collects a whole model turn before returning chunks, so `stream: true` is not genuinely live for tool-enabled requests.

### API compatibility limitations

- System, user, assistant, and tool messages are flattened into text labels such as `[SYSTEM]` and `[USER]`; they are not preserved as separate provider roles. “Full passthrough” is therefore approximate, and the system instruction does not retain a separate high-priority role at the web provider.
- Image content is unsupported. The shared text extractor raises for image content; the plain-chat endpoint does not consistently convert that into a clean 400 response.
- `temperature`, `max_tokens`, and `stream_options.include_usage` are accepted but not applied. Some clients may also expect one stable completion ID across all streamed chunks; this implementation creates new IDs in several chunk helpers.
- Startup asks for prompt mode while importing the server module. Running as `python server.py` is the intended path; importing it through a generic ASGI command can prompt/block unexpectedly.

## Checks performed

| Check | Result |
|---|---|
| Python compilation (`compileall`) | Passed |
| Direct `python test_parser.py` | **Failed** before tests: the fake `playwright.async_api` module in the test file omits `TimeoutError`, which `app.py` imports |
| Existing DeepSeek/Qwen parser assertions with an in-memory test-only shim for that missing fake attribute | Passed: all 10 included scenarios |
| Focused tool-sieve probes | Reproduced the split-marker, multi-block, unvalidated-name, bare-JSON, and nested-object issues described above |
| `pip-audit -r requirements.txt` against the current unpinned resolution | No known vulnerable dependency versions reported at review time; this does not audit application logic or guarantee future resolutions |
| Live browser/provider test | Not run; no Chromium browser or account session was installed/used |

The repository's test script currently cannot run unmodified, and it has no tests for `tool_bridge.py`, the HTTP endpoints, cancellation/timeouts, or browser-route fulfillment.

## Packaging and maintenance gaps

There is no README, license file, `pyproject.toml`/`setup.py`, or lockfile. This is not a standard `pip install .` package; installation is manual (`pip install -r requirements.txt`, then install a compatible Playwright browser). Dependency versions are unconstrained and Python/browser requirements are undocumented. No CI configuration was present in the reviewed file set.

If you want to finish the setup on a desktop computer, use Python 3.10+ and run from the repository directory:

```bash
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell instead:
# .\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
PROMPT_MODE=full python server.py
```

On Windows PowerShell, set the mode with `$env:PROMPT_MODE='full'` before `python server.py`. For Qwen, launch `python qwen_server.py` instead (default port 8001). The program opens a visible browser and waits for manual login; it is not configured for unattended/headless operation by default.

## Recommended order of fixes

1. Validate parsed tool names against the exact `req.tools` list and validate arguments against their schemas before returning any tool call; reject unknown calls. Handle `tool_choice="required"` correctly.
2. Fix the Playwright test stub and add tests for split SSE markers, multiple calls, malformed/incomplete blocks, bare JSON, nested arguments, unknown tools, and both streaming/non-streaming API responses.
3. Add total timeouts around browser submission and queue reads; impose prompt/response limits and queue back-pressure.
4. Normalize response headers before `route.fulfill()` (or preserve raw encoded bytes consistently), and check provider HTTP status/error payloads.
5. Remove request-body logging/echoing; add an optional API token and keep the server loopback-only by default.
6. Document Python/browser setup, provider data flow, terms/privacy caveats, supported content, and limitations; pin dependencies or provide a lockfile and add an explicit license.
7. Prefer official provider APIs for production use.

## Bottom line

The isolated Python dependencies were installed in the workspace, but the full browser bridge was **not** installed or verified on a user's machine. The code is an experimental browser-session relay, not a private/local model server. It may be useful for controlled testing, but the tool parser, request handling, documentation, and test coverage need work before trusting it with sensitive data or agent tools.
