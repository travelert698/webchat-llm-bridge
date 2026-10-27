"""Focused tests for the shared OpenAI/Anthropic tool-call bridge.

These tests use fake provider streams only; no browser, account, or live model
is used. Run with `python test_tool_bridge.py`.
"""
from __future__ import annotations

import json
import unittest

from fastapi.responses import JSONResponse, StreamingResponse

import tool_bridge


START = tool_bridge.DEEPSEEK_TOOL_CALL_START
END = tool_bridge.DEEPSEEK_TOOL_CALL_END
SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def tool_call(name: str, args: dict) -> str:
    return f'<tool_call><name>{name}</name><args>{json.dumps(args)}</args></tool_call>'


class FakeBridge:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    async def stream_response(self, prompt: str):
        self.prompts.append(prompt)
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        payload = {
            "id": "fake-provider-id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    @staticmethod
    def _parse_sse(raw: str) -> dict | None:
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    return None
                choices = payload.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    return {
                        "content": delta.get("content", ""),
                        "reasoning": delta.get("reasoning_content", ""),
                    }
        return None


class ToolParserTests(unittest.TestCase):
    def test_native_deepseek_marker_split_across_chunks(self):
        start = START
        end = END
        raw = json.dumps([{"name": "search", "arguments": {"query": "café"}}], ensure_ascii=False)
        sieve = tool_bridge.Sieve()
        events = []
        chunks = [
            "before " + start[:9],
            start[9:] + raw + end[:8],
            end[8:] + " after",
        ]
        for chunk in chunks:
            events.extend(sieve.feed(chunk))
        events.extend(sieve.finish())
        self.assertEqual(events[0], ("content", "before "))
        self.assertEqual(events[1], ("tool_block", "json", raw))
        self.assertEqual(events[2], ("content", " after"))
        self.assertEqual(
            tool_bridge.parse_tool_blocks(events[1][1], events[1][2]),
            [{"name": "search", "arguments": {"query": "café"}}],
        )

    def test_deepseek_dsml_calls_v41_and_double_bar_rendering(self):
        raw = '''<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="Bash">
<｜｜DSML｜｜ parameter name="command" string="true">cd "C:/Users/Agent_Test/.hyperframes-studio/Create a hello world animation" &amp;&amp; hyperframes catalog --query "title animation" 2&gt;&amp;1 | head -50</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="description" string="true">Search the block registry for title animations</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="limit" string="false">5</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="Bash">
<｜｜DSML｜｜ parameter name="command" string="true">cat "C:/Users/Agent_Test/AppData/Roaming/HyperFrames/guides/studio-plugin/skills/hyperframes/references/[format.md](http://format.md)"</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="description" string="true">Read the format reference</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''
        allowed = {"Bash": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }}
        for token in ("｜｜DSML｜｜", "｜DSML｜"):
            with self.subTest(token=token):
                rendered = raw.replace("｜｜DSML｜｜", token)
                _, outcome = tool_bridge._events_to_chunks([("content", rendered)], "test", allowed)
                self.assertEqual(outcome[0], "parsed")
                self.assertEqual(len(outcome[1]), 2)
                self.assertEqual(outcome[1][0]["name"], "Bash")
                self.assertIn(" && ", outcome[1][0]["arguments"]["command"])
                self.assertIn("2>&1", outcome[1][0]["arguments"]["command"])
                self.assertEqual(outcome[1][0]["arguments"]["limit"], 5)
                self.assertEqual(outcome[1][1]["arguments"]["description"], "Read the format reference")

    def test_native_marker_ascii_and_generic_aliases(self):
        pairs = [
            ("<|tool▁calls▁begin|>", "<|tool▁calls▁end|>"),
            ("<｜tool_calls_begin｜>", "<｜tool_calls_end｜>"),
            ("<|tool_calls_begin|>", "<|tool_calls_end|>"),
            ("<tool_calls>", "</tool_calls>"),
        ]
        payload = '[{"name":"search","arguments":{"query":"x"}}]'
        for start, end in pairs:
            with self.subTest(start=start):
                sieve = tool_bridge.Sieve()
                events = sieve.feed(start + payload + end) + sieve.finish()
                block = next(event for event in events if event[0] == "tool_block")
                self.assertEqual(tool_bridge.parse_tool_blocks(block[1], block[2])[0]["name"], "search")

    def test_multiple_tagged_calls_are_not_dropped(self):
        sieve = tool_bridge.Sieve()
        raw = tool_call("search", {"query": "one"}) + "\n" + tool_call("search", {"query": "two"})
        events = sieve.feed(raw) + sieve.finish()
        blocks = [event for event in events if event[0] == "tool_block"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            [tool_bridge.parse_tool_blocks(kind, body)[0]["arguments"]["query"] for _, kind, body in blocks],
            ["one", "two"],
        )

    def test_json_repair_and_openai_envelope(self):
        relaxed = "[{name: 'search', arguments: {query: 'weather', limit: 3,},}]"
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", relaxed),
            [{"name": "search", "arguments": {"query": "weather", "limit": 3}}],
        )
        envelope = json.dumps({"tool_calls": [{
            "id": "call_x",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query":"wind"}'},
        }]})
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", envelope),
            [{"name": "search", "arguments": {"query": "wind"}}],
        )

    def test_bare_json_is_not_extracted_from_explanatory_prose(self):
        prose = 'Example only:\n```json\n{"name":"search","arguments":{"query":"demo"}}\n```'
        self.assertEqual(tool_bridge.parse_tool_blocks("json", prose), [])
        fenced = '```json\n{"name":"search","arguments":{"query":"demo"}}\n```'
        self.assertEqual(
            tool_bridge.parse_tool_blocks("json", fenced),
            [{"name": "search", "arguments": {"query": "demo"}}],
        )

    def test_run_code_nested_args_templates_and_string_false_positive(self):
        block = '''<run_code>
const html = `<h1>${title}</h1>`;
await tools.render({html, options: {limit: 3, enabled: true}, labels: ["brace }", 'single'],});
const example = "tools.fake({evil: true})"; // tools.nope({x: 1})
</run_code>'''
        calls = tool_bridge.parse_tool_blocks("run_code", block)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "render")
        self.assertEqual(calls[0]["arguments"], {
            "html": "<h1>${title}</h1>",
            "options": {"limit": 3, "enabled": True},
            "labels": ["brace }", "single"],
        })

    def test_invoke_parameter_blocks(self):
        block = '<invoke name="search"><parameter name="query">weather</parameter><parameter name="limit">3</parameter></invoke>'
        parsed = tool_bridge.parse_tool_blocks("tool_call", block)
        self.assertEqual(parsed, [{"name": "search", "arguments": {"query": "weather", "limit": 3}}])

    def test_external_schema_refs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "external JSON Schema references"):
            tool_bridge._validate_local_schema_refs({"$ref": "https://example.invalid/schema.json"})
        tool_bridge._validate_local_schema_refs({"$ref": "#/$defs/local"})

    def test_schema_and_allowed_tool_validation(self):
        events = [("content", START + json.dumps([
            {"name": "search", "arguments": {"query": "ok", "limit": 2}}
        ]) + END)]
        chunks, outcome = tool_bridge._events_to_chunks(events, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "parsed")
        self.assertEqual(outcome[1][0]["arguments"]["limit"], 2)

        unknown = [("content", START + json.dumps([
            {"name": "shell_exec", "arguments": {"query": "bad"}}
        ]) + END)]
        _, outcome = tool_bridge._events_to_chunks(unknown, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "unparseable")

        invalid = [("content", tool_call("search", {"query": 123}))]
        _, outcome = tool_bridge._events_to_chunks(invalid, "test", {"search": SCHEMA})
        self.assertEqual(outcome[0], "unparseable")


class ToolApiBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_has_stable_id_role_and_done(self):
        call_text = START + json.dumps([
            {"name": "search", "arguments": {"query": "hello"}}
        ]) + END
        bridge = FakeBridge([call_text])
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "find something"}],
            "tools": [{"type": "function", "function": {
                "name": "search", "parameters": SCHEMA,
            }}],
            "tool_choice": "auto",
            "stream": True,
        }
        response = await tool_bridge.handle_chat(request, bridge)
        self.assertIsInstance(response, StreamingResponse)
        body = "".join([part async for part in response.body_iterator])
        data_lines = [line for line in body.splitlines() if line.startswith("data:")]
        objects = [json.loads(line[5:].strip()) for line in data_lines if line[5:].strip() != "[DONE]"]
        self.assertEqual(body.count("data: [DONE]"), 1)
        self.assertEqual(objects[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertTrue(objects[-1]["choices"][0]["finish_reason"] == "tool_calls")
        self.assertEqual(len({obj["id"] for obj in objects}), 1)
        self.assertEqual(len({obj["created"] for obj in objects}), 1)
        call_deltas = [obj["choices"][0]["delta"].get("tool_calls") for obj in objects]
        self.assertTrue(any(call_deltas))

    async def test_required_choice_without_tools_returns_bad_request(self):
        response = await tool_bridge.handle_chat({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
            "tool_choice": "required",
        }, FakeBridge(["unused"]))
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("param", body["error"])
        self.assertIn("code", body["error"])

    async def test_parallel_tool_calls_false_is_enforced(self):
        response_text = tool_call("search", {"query": "one"}) + tool_call("search", {"query": "two"})
        bridge = FakeBridge([response_text])
        response = await tool_bridge.handle_chat({
            "messages": [{"role": "user", "content": "search two things"}],
            "tools": [{"type": "function", "function": {
                "name": "search", "parameters": SCHEMA,
            }}],
            "parallel_tool_calls": False,
        }, bridge)
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(bridge.prompts), tool_bridge.MAX_REPAIR_RETRIES + 1)
        self.assertIn("at most one", bridge.prompts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
