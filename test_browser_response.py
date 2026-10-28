"""Offline tests for compressed SSE replay into the browser route.

HTTPX decodes ``aiter_bytes()`` bodies. These tests verify that response
headers are normalized before the decoded body is passed to Playwright, which
prevents browser fetch failures caused by stale encoding/framing metadata.
"""
from __future__ import annotations

import gzip
import io
import sys
import types
import unittest
from contextlib import redirect_stdout

import httpx

# Import the bridge modules without launching or requiring a real browser.
fake_module = types.ModuleType("playwright")
fake_async_api = types.ModuleType("playwright.async_api")
fake_async_api.async_playwright = lambda: None
fake_async_api.TimeoutError = TimeoutError
sys.modules["playwright"] = fake_module
sys.modules["playwright.async_api"] = fake_async_api

import app
import qwen_app


class FakeRoute:
    def __init__(self) -> None:
        self.fulfilled: dict | None = None

    async def fulfill(self, **kwargs) -> None:
        self.fulfilled = kwargs


class BrowserResponseReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_compressed_sse_is_replayed_with_consistent_headers(self):
        payload = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        compressed = gzip.compress(payload)
        headers = {
            "content-type": "text/event-stream; charset=utf-8",
            "content-encoding": "gzip",
            "content-length": str(len(compressed)),
            "connection": "keep-alive, x-hop-token",
            "x-hop-token": "must-not-be-forwarded",
            "x-safe-provider-header": "preserved",
        }

        async def upstream(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers=headers, content=compressed)

        transport = httpx.MockTransport(upstream)
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("GET", "https://example.test/completion") as response:
                decoded_body = b"".join([chunk async for chunk in response.aiter_bytes()])
                self.assertEqual(decoded_body, payload)

                for bridge in (app, qwen_app):
                    forwarded_headers = bridge._headers_for_httpx_request({
                        "Host": "provider.test",
                        "Content-Length": "123",
                        "Accept-Encoding": "br, zstd",
                        "Content-Type": "application/json",
                        "Authorization": "private-session-token",
                    })
                    forwarded_names = {name.lower() for name in forwarded_headers}
                    self.assertNotIn("host", forwarded_names)
                    self.assertNotIn("content-length", forwarded_names)
                    self.assertNotIn("accept-encoding", forwarded_names)
                    self.assertIn("content-type", forwarded_names)
                    self.assertIn("authorization", forwarded_names)

                    route = FakeRoute()
                    await bridge._fulfill_decoded_response(
                        route, decoded_body, response.status_code, response.headers
                    )
                    self.assertIsNotNone(route.fulfilled)
                    self.assertEqual(route.fulfilled["body"], payload)
                    self.assertEqual(route.fulfilled["status"], 200)
                    replay_headers = {
                        key.lower(): value
                        for key, value in route.fulfilled["headers"].items()
                    }
                    self.assertEqual(
                        replay_headers["content-type"],
                        "text/event-stream; charset=utf-8",
                    )
                    self.assertEqual(replay_headers["x-safe-provider-header"], "preserved")
                    self.assertNotIn("content-encoding", replay_headers)
                    self.assertNotIn("content-length", replay_headers)
                    self.assertNotIn("transfer-encoding", replay_headers)
                    self.assertNotIn("connection", replay_headers)
                    self.assertNotIn("x-hop-token", replay_headers)
                    self.assertIn(b"data: [DONE]", route.fulfilled["body"])

    def test_diagnostics_distinguish_aborts_without_logging_body(self):
        for bridge in (app, qwen_app):
            output = io.StringIO()
            with redirect_stdout(output):
                bridge._log_response_replay(
                    status=200,
                    headers={
                        "content-type": "text/event-stream",
                        "content-encoding": "gzip",
                    },
                    decoded_bytes=128,
                    done_seen=False,
                    outcome="aborted_client_cancel",
                )
            report = output.getvalue()
            self.assertIn("encoding='gzip'", report)
            self.assertIn("decoded_bytes=128", report)
            self.assertIn("done_marker=False", report)
            self.assertIn("route=aborted_client_cancel", report)
            self.assertNotIn("prompt", report.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
