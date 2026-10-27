# Reference-driven tool-calling review

**Review date:** 2026-10-27 (Asia/Colombo)  
**Reference revision:** `99f77302cebc0b32f10066f96b863f2f27752708` ([NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api/tree/99f77302cebc0b32f10066f96b863f2f27752708)). Relevant source paths:

- [`src/openai_adapter/request/tools.rs`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/src/openai_adapter/request/tools.rs) — validates tool definitions/choices and builds tool-call instructions.
- [`src/openai_adapter/response/tool_parser.rs`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/src/openai_adapter/response/tool_parser.rs) — parses DeepSeek-native tool markers, repairs some malformed structured output, handles invoke-style calls, and includes streaming-specific limits.
- [`src/anthropic_compat`](https://github.com/NIyueeE/ds-free-api/tree/99f77302cebc0b32f10066f96b863f2f27752708/src/anthropic_compat) and [`docs/compat-audit.md`](https://github.com/NIyueeE/ds-free-api/blob/99f77302cebc0b32f10066f96b863f2f27752708/docs/compat-audit.md) — document Anthropic event sequencing, stop reasons, error events, and OpenAI SSE termination expectations.
- [DeepSeek V4.1 encoding reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/README.md) — defines the newer DSML `<｜DSML｜ calls>` / `invoke` / `parameter` syntax and the `string="true|false"` parameter rule.

## Findings applied to this project

The parser work also addressed the exact format in the follow-up report: DeepSeek V4.1 documents DSML wrappers such as `<｜DSML｜ calls>` with `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>` tags (note the leading spaces), while earlier variants use `tool_calls` or `function_calls`. The sample's doubled vertical bars are accepted too. The `string="true"` attribute means keep a parameter as literal text; `string="false"` means decode its value as JSON. This is why a V4.1 model may output a different-looking call than the bridge's preferred `<tool_call>` wrapper: it is speaking its own documented tool-call grammar, not accidentally writing shell commands as prose.

More broadly, the project primarily asked the web model to emit `<tool_call>` XML and `<run_code>` code. The reference also recognizes the DeepSeek-native `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>` array form, multiple spellings of those markers, and `<invoke>/<parameter>` blocks. A compatible parser should treat stream chunk boundaries as arbitrary, allow more than one call, bound its buffering, and validate the resulting call against the request before returning it.

The current `tool_bridge.py` now:

- recognizes the bridge XML form, DeepSeek-native marker forms, DeepSeek DSML V3.2/V4/V4.1 wrappers (including the V4.1 leading-space spelling), and generic `<tool_calls>` wrappers; it carries partial start/end markers across chunks and can extract multiple calls;
- accepts standard JSON and OpenAI-like `tool_calls` envelopes, plus a conservative relaxed parser for common model formatting errors (single quotes, unquoted keys, and trailing commas); untagged JSON must be the entire response or a whole code fence, which avoids extracting an illustrative code block from prose;
- parses `<invoke>/<parameter>` blocks and nested JS-like `<run_code>` arguments using a non-evaluating parser; no model-generated code is executed;
- enforces request-scoped names and JSON Schemas (local `$ref` only; remote schema references are rejected), caps returned calls at 32, honors `tool_choice` (including `required`/named tools), supports OpenAI `parallel_tool_calls: false`, and makes up to two repair retries after the first attempt;
- emits standard nested OpenAI error envelopes for tool-path failures and tool-stream chunks with one stable completion ID/timestamp, an initial assistant-role chunk, the correct `tool_calls` finish reason, and exactly one `[DONE]` marker.

The Anthropic adapter now also honors `disable_parallel_tool_use`, maps OpenAI `length`/content-filter finish reasons to Anthropic stop reasons, and explicitly rejects unsupported provider-managed Anthropic tools instead of silently fabricating a local function tool. Tool-enabled Anthropic streams preserve the `content_block_start` / delta / stop event sequence. The existing `app.py` and `qwen_app.py` browser/UI automation files were not edited; no direct provider API completion call was introduced.

## Deliberate exclusions

The reference project documents upstream risk-control tokens, device-fingerprint/emulation options, session/account controls, and ban/mute mitigation. These were not adopted. This project makes no attempt to spoof identities, emulate risk tokens, rotate accounts/devices, or bypass provider enforcement. The changes are limited to protocol parsing, validation, bounded retries, and truthful error behavior. Browser UI automation is still subject to provider terms and can still fail or lead to account restrictions; this work does not make it safer from bans.

## Verification and remaining limits

Offline verification passed after these changes:

- `python test_tool_bridge.py` — 13 focused parser, DSML, validation, parallel-call, schema-reference, SSE, and request-choice tests.
- `python test_anthropic_compat.py` — 13 adapter tests.
- `python test_api_smoke.py` — both server modules' OpenAI/Anthropic routes with a fake stream.
- `python test_parser.py` — 10 existing DeepSeek/Qwen parser scenarios.
- `python -m compileall -q .`, Ruff `--select F,E9`, and `git diff --check`.

These tests use fake provider streams. There is still **no live browser/provider/agent round trip**, so no claim is made that every model will emit parseable tool calls or that every version of Claude Code/other agent clients will work. The bridge's tool calls remain model-generated text, not native provider function calls. Token estimates and unsupported Anthropic features remain as described in `CLAUDE_CODE_SETUP.md`. `pip-audit` is not installed in the workspace, so a current dependency vulnerability audit was not performed.
