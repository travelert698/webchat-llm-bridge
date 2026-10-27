"""Offline tests for the minimal Anthropic Messages compatibility adapter."""
from __future__ import annotations

import asyncio
import json
import unittest

import anthropic_compat as ac
import tool_bridge


class FakeBridge:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    async def stream_response(self, prompt: str):
        self.prompts.append(prompt)
        for reply in self.replies:
            yield self._make_openai_sse(reply)

    @staticmethod
    def _make_openai_sse(text: str) -> str:
        obj = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        return "data: " + json.dumps(obj) + "\n\n"

    @staticmethod
    def _parse_sse(raw: str):
        try:
            obj = json.loads(raw[6:].strip())
            delta = obj["choices"][0]["delta"]
            return {
                "content": delta.get("content", "") or "",
                "reasoning": delta.get("reasoning_content", "") or "",
            }
        except Exception:
            return None


def parse_anthropic_events(raw_chunks):
    events = []
    for raw in raw_chunks:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event_name = None
        data = None
        for line in raw.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if data:
            events.append((event_name, json.loads(data)))
    return events


async def read_response_body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


class AnthropicCompatTests(unittest.TestCase):
    def test_normalize_messages_and_tool_results(self):
        request = ac.AnthropicMessagesRequest(
            model="claude-test",
            max_tokens=128,
            system=[{"type": "text", "text": "Use tools carefully."}],
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file text"}],
                },
            ],
        )
        normalized = ac.normalize_messages(request)
        self.assertEqual(normalized[0]["role"], "system")
        self.assertEqual(normalized[1]["tool_calls"][0]["function"]["name"], "Read")
        self.assertIn("TOOL RESULT (Read)", normalized[2]["content"])
        self.assertIn("file text", normalized[2]["content"])

    def test_normalize_anthropic_tools_and_choices(self):
        tools = ac.normalize_tools([{
            "name": "Read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }])
        self.assertEqual(tools[0]["function"]["name"], "Read")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["path"])
        self.assertEqual(ac.normalize_tool_choice({"type": "any"}), "required")
        self.assertEqual(ac.normalize_tool_choice({"type": "tool", "name": "Read"})["function"]["name"], "Read")
        self.assertTrue(ac._disable_parallel_tool_use({"type": "auto", "disable_parallel_tool_use": True}))
        self.assertEqual(ac._anthropic_stop_reason("length"), "max_tokens")

    def test_server_side_anthropic_tools_are_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "server-side Anthropic tools are unsupported"):
            ac.normalize_tools([{"type": "web_search_20250305", "name": "web_search"}])

    def test_non_stream_text_response(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64, messages=[{"role": "user", "content": "Hi"}]
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["Hello"]), "deepseek-chat"))
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["model"], "web-alias")
        self.assertEqual(response["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertIn("input_tokens", response["usage"])

    def test_stream_text_has_anthropic_event_sequence(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64, stream=True,
            messages=[{"role": "user", "content": "Hi"}],
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["Hello"]), "deepseek-chat"))
        raw = asyncio.run(read_response_body(response))
        events = parse_anthropic_events(raw)
        kinds = [payload["type"] for _, payload in events]
        self.assertEqual(kinds, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ])
        self.assertEqual(events[2][1]["delta"]["text"], "Hello")

    def test_tool_use_round_trip_and_schema_validation(self):
        call = tool_bridge.DEEPSEEK_TOOL_CALL_START + json.dumps([
            {"name": "Read", "arguments": {"path": "a.txt"}}
        ]) + tool_bridge.DEEPSEEK_TOOL_CALL_END
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "Read a.txt"}],
            tools=[{
                "name": "Read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
            }],
            tool_choice={"type": "tool", "name": "Read"},
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge([call]), "deepseek-chat"))
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["content"][0]["type"], "tool_use")
        self.assertEqual(response["content"][0]["name"], "Read")
        self.assertEqual(response["content"][0]["input"], {"path": "a.txt"})

    def test_anthropic_disable_parallel_tool_use_is_enforced(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "Search two things"}],
            tools=[{"name": "Search", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        two_calls = (
            '<tool_call><name>Search</name><args>{"query":"one"}</args></tool_call>'
            '<tool_call><name>Search</name><args>{"query":"two"}</args></tool_call>'
        )
        bridge = FakeBridge([two_calls])
        response = asyncio.run(ac.handle_messages(request, bridge, "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertGreaterEqual(response.status_code, 400)
        self.assertIn("at most one tool", bridge.prompts[0])

    def test_invalid_tool_schema_is_rejected(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=64,
            messages=[{"role": "user", "content": "use the tool"}],
            tools=[{"name": "Broken", "input_schema": {"type": 7}}],
        )
        response = asyncio.run(ac.handle_messages(request, FakeBridge(["unused"]), "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertEqual(response.status_code, 400)

    def test_unknown_tool_is_never_returned(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128,
            messages=[{"role": "user", "content": "run something"}],
            tools=[{"name": "Read", "input_schema": {"type": "object"}}],
        )
        bad_call = '<tool_call><name>ShellExec</name><args>{"cmd":"rm -rf /"}</args></tool_call>'
        response = asyncio.run(ac.handle_messages(request, FakeBridge([bad_call]), "deepseek-chat"))
        self.assertIsInstance(response, ac.JSONResponse)
        self.assertGreaterEqual(response.status_code, 400)
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(body["type"], "error")

    def test_stream_tool_use_has_anthropic_events(self):
        request = ac.AnthropicMessagesRequest(
            model="web-alias", max_tokens=128, stream=True,
            messages=[{"role": "user", "content": "Read a.txt"}],
            tools=[{
                "name": "Read", "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }],
            tool_choice={"type": "tool", "name": "Read"},
        )
        call = '<tool_call><name>Read</name><args>{"path":"a.txt"}</args></tool_call>'
        response = asyncio.run(ac.handle_messages(request, FakeBridge([call]), "deepseek-chat"))
        raw = asyncio.run(read_response_body(response))
        events = parse_anthropic_events(raw)
        kinds = [payload["type"] for _, payload in events]
        self.assertEqual(kinds, [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ])
        self.assertEqual(events[2][1]["delta"]["type"], "input_json_delta")
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")

    def test_bare_json_tool_call_is_request_scoped(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
        chunks, outcome = tool_bridge._events_to_chunks(
            [("content", '{"name":"Read","arguments":{"path":"a.txt"}}')],
            "test-model",
            {"Read": schema},
        )
        self.assertEqual(outcome[0], "parsed")
        self.assertEqual(outcome[1][0]["name"], "Read")
        unknown_chunks, unknown_outcome = tool_bridge._events_to_chunks(
            [("content", '{"name":"ShellExec","arguments":{"cmd":"x"}}')],
            "test-model",
            {"Read": schema},
        )
        self.assertEqual(unknown_outcome[0], "unparseable")

    def test_sieve_handles_split_and_multiple_tool_blocks(self):
        sieve = tool_bridge.Sieve()
        parsed = sieve.feed("prefix <tool_")
        parsed += sieve.feed('call><name>Read</name><args>{}</args></tool_call>')
        parsed += sieve.feed('<tool_call><name>Write</name><args>{}</args></tool_call> tail')
        parsed += sieve.finish()
        calls = [item for item in parsed if item[0] == "tool_block"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "tool_call")
        self.assertIn("prefix ", [item[1] for item in parsed if item[0] == "content"])

    def test_count_tokens_is_available(self):
        request = ac.AnthropicCountTokensRequest(
            model="web-alias", messages=[{"role": "user", "content": "A short prompt."}]
        )
        result = ac.handle_count_tokens(request)
        self.assertGreater(result["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
