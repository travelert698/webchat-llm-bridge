"""ASGI route smoke tests for both APIs using a fake webchat stream.

Run with `python test_api_smoke.py` after installing requirements. No browser,
provider account, or live web session is used.
"""
from __future__ import annotations

import json
import os
import unittest

import httpx

os.environ.setdefault("PROMPT_MODE", "full")
import server  # noqa: E402
import qwen_server  # noqa: E402


TOOL_REPLY = '<tool_call><name>Read</name><args>{"path":"a.txt"}</args></tool_call>'
OPENAI_TOOL_REPLY = '<tool_call><name>read</name><args>{"path":"a.txt"}</args></tool_call>'
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def _openai_sse(text: str) -> str:
    payload = {
        "id": "fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "fake-web-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return "data: " + json.dumps(payload) + "\n\n"


async def _fake_stream_response(prompt: str):
    if "Tool name: Read" in prompt:
        yield _openai_sse(TOOL_REPLY)
    elif "Tool name: read" in prompt:
        yield _openai_sse(OPENAI_TOOL_REPLY)
    else:
        yield _openai_sse("fake reply")


class ApiSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.servers = [
            (server.api, server.app, "deepseek-chat"),
            (qwen_server.api, qwen_server.app, "qwen-auto"),
        ]
        self.original_streams = []
        for _, bridge, _ in self.servers:
            self.original_streams.append(bridge.stream_response)
            bridge.stream_response = _fake_stream_response

    async def asyncTearDown(self):
        for (_, bridge, _), original in zip(self.servers, self.original_streams):
            bridge.stream_response = original

    async def test_messages_openai_and_tool_routes_for_both_servers(self):
        for api, _, backend_model in self.servers:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api),
                base_url="http://test",
            ) as client:
                response = await client.head("/api/hello")
                self.assertEqual(response.status_code, 200)

                messages = [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Hello"},
                ]
                response = await client.post("/v1/messages", json={
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["content"][0]["text"], "fake reply")
                self.assertEqual(response.json()["model"], "test-alias")

                response = await client.post("/v1/messages/count_tokens", json={
                    "model": "test-alias",
                    "messages": messages,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertGreater(response.json()["input_tokens"], 0)

                response = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "stream": False,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "fake reply")

                response = await client.post("/v1/chat/completions", json={
                    "model": backend_model,
                    "messages": messages,
                    "tools": [],
                    "tool_choice": "required",
                })
                self.assertEqual(response.status_code, 400, response.text)

                anthropic_tool = {
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                    "tools": [{"name": "Read", "input_schema": TOOL_SCHEMA}],
                    "tool_choice": {"type": "tool", "name": "Read"},
                }
                response = await client.post("/v1/messages", json=anthropic_tool)
                self.assertEqual(response.status_code, 200, response.text)
                block = response.json()["content"][0]
                self.assertEqual((block["type"], block["name"], block["input"]),
                                 ("tool_use", "Read", {"path": "a.txt"}))

                response = await client.post("/v1/messages", json={**anthropic_tool, "stream": True})
                self.assertEqual(response.status_code, 200, response.text)
                events = [line for line in response.text.splitlines() if line.startswith("event:")]
                self.assertIn("event: content_block_start", events)
                self.assertIn("event: message_stop", events)
                self.assertIn('"stop_reason":"tool_use"', response.text)

                response = await client.post("/v1/messages", json={
                    "model": "test-alias", "max_tokens": 64, "messages": messages,
                    "tools": [{"name": "Broken", "input_schema": {"type": 7}}],
                })
                self.assertEqual(response.status_code, 400, response.text)

                openai_tool = {
                    "model": backend_model,
                    "messages": messages,
                    "tools": [{"type": "function", "function": {
                        "name": "read", "parameters": TOOL_SCHEMA,
                    }}],
                    "tool_choice": {"type": "function", "function": {"name": "read"}},
                    "stream": False,
                }
                response = await client.post("/v1/chat/completions", json=openai_tool)
                self.assertEqual(response.status_code, 200, response.text)
                call = response.json()["choices"][0]["message"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "read")

                response = await client.post("/v1/messages", json={
                    "model": "test-alias",
                    "max_tokens": 64,
                    "messages": messages,
                    "stream": True,
                })
                self.assertEqual(response.status_code, 200, response.text)
                event_names = [
                    line[7:] for line in response.text.splitlines()
                    if line.startswith("event:")
                ]
                self.assertIn("message_start", event_names)
                self.assertIn("content_block_delta", event_names)
                self.assertEqual(event_names[-1], "message_stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
