# Anthropic Messages + OpenAI bridge setup

This project keeps the existing OpenAI Chat Completions API and adds a minimum Anthropic Messages API adapter. It targets Claude Code/custom Anthropic Messages clients and OpenAI Chat Completions clients; it cannot guarantee compatibility with every proprietary agent. For the reference-based parser review and exact offline check results, see `REFERENCE_REVIEW.md`.

## Files required for the runtime feature

Copy/replace these complete runtime files in the project root:

- `server.py` — DeepSeek server routes and adapter wiring.
- `qwen_server.py` — Qwen server routes and adapter wiring.
- `anthropic_compat.py` — Anthropic request/response normalization and SSE events.
- `tool_bridge.py` — request-scoped tool formatting, parsing, and JSON Schema validation.
- `requirements.txt` — adds `jsonschema` for tool-input validation.

`app.py` and `qwen_app.py` are unchanged. `run_both.py` only changes its printed endpoint hints and is optional. The test files and this guide are not needed to run the bridge.

Install updated dependencies after replacing the files:

```powershell
python -m pip install -r requirements.txt
```

## Run the bridge

Start the backend server in one terminal. Use `PROMPT_MODE=full` so it includes the full conversation:

```powershell
# DeepSeek
$env:PROMPT_MODE = 'full'
python server.py

# Or Qwen, in a separate run
$env:PROMPT_MODE = 'full'
python qwen_server.py
```

Start Claude Code in a second terminal with the base URL set to the server root (do not append `/v1`):

```powershell
# DeepSeek
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8000'
claude

# Qwen uses port 8001 instead.
```

The persistent user-level environment variable can be set using Windows Environment Variables or PowerShell's `[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', 'http://127.0.0.1:8000', 'User')`. Open a new terminal after changing it. Claude Code must inherit that variable; setting it only in the server terminal is not enough.

The DeepSeek/Qwen bridges open a visible browser and wait for manual login. Browser profiles (`browser_data/`, `qwen_browser_data/`) contain session data; keep them private. A desktop GUI is required by the current headed-browser setup.

## Endpoints

- DeepSeek: `POST http://127.0.0.1:8000/v1/messages`, optional `/v1/messages/count_tokens`.
- Qwen: `POST http://127.0.0.1:8001/v1/messages`, optional `/v1/messages/count_tokens`.
- Existing OpenAI route remains `POST /v1/chat/completions` on both servers.
- `GET` and `HEAD /api/hello` are lightweight client connectivity probes; `/health` remains available.

## Tool-calling model and limitations

Tool definitions are request-scoped. Before a call is returned, the bridge checks the name against the exact request and validates the parsed argument object against its JSON Schema. It supports OpenAI `tool_choice` modes, named tools, `parallel_tool_calls: false`, Anthropic `auto`/`any`/named/`none` choices, and Anthropic `disable_parallel_tool_use`. The bridge returns calls to the client; it never executes them or bypasses the client's approval policy.

The text parser handles the bridge's `<tool_call>` format, DeepSeek-family native markers and DSML tool wrappers (including the V3.2/V4/V4.1 variants such as `<｜DSML｜ calls>`), exact-response JSON/OpenAI-style tool-call envelopes, Anthropic-style `<invoke>` blocks, and the existing `<run_code>` wrapper. DSML's `string="true"` parameters stay strings; `string="false"` parameters are decoded as JSON. It buffers split markers, preserves multiple calls, parses nested JS-like objects without evaluating code, caps a response at 32 calls, and performs up to two repair retries after the initial attempt. Bare JSON is recognized only when it is the whole response (or a whole fenced response), not when embedded in explanatory prose.

DeepSeek/Qwen **web sessions do not expose native function-calling APIs to this bridge**. Structured tool calls are text the model is asked to emit, so compliance is probabilistic; parsing, retries, and schema validation reduce errors but do not make it native or guarantee success. Anthropic provider-managed/server tools (for example hosted web-search/computer tools) are explicitly unsupported; use client-defined tools with `input_schema`. Tool-enabled replies may be buffered before Anthropic tool-use events are emitted. The bridge targets Anthropic Messages and OpenAI Chat Completions, not every agent-specific or proprietary protocol. It does not implement the OpenAI Responses API, Google-specific formats, full MCP, or every Claude Code beta feature.

Token counts are rough estimates, and OpenAI `stream_options.include_usage` usage accounting is not synthesized. The browser backends do not faithfully enforce generation controls such as `temperature` or `max_tokens`; image/document inputs, signed thinking, prompt caching, and adaptive thinking are unsupported. The model field does not select the backend; port 8000 is DeepSeek and 8001 is Qwen.

The local API has no authentication by default; keep it bound to loopback and do not expose it to a network. Routing Claude Code to non-Claude models is not an officially supported Anthropic configuration. Use a non-sensitive test project and keep tool approval enabled.

## Offline tests

```powershell
python test_parser.py
python test_tool_bridge.py
python test_anthropic_compat.py
python test_api_smoke.py
```

These use fake streams and validate the existing DeepSeek/Qwen stream parsers, tool-call parsing/validation and both API routes, Anthropic/OpenAI SSE framing, and tool-choice behavior. They do not verify live provider selectors, browser logins, or current DeepSeek/Qwen website behavior.
