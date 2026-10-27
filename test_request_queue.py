"""FIFO queue, RPM pacing, cancellation, and HTTP back-pressure tests."""
from __future__ import annotations

import asyncio
import unittest

import httpx
from fastapi import FastAPI

from request_queue import RequestQueue, RequestQueueFull, RequestQueueMiddleware


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += delay
        await asyncio.sleep(0)


class RequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for_pending(self, queue: RequestQueue, expected: int) -> None:
        for _ in range(100):
            if (await queue.snapshot())["pending"] == expected:
                return
            await asyncio.sleep(0)
        self.fail(f"queue did not reach pending={expected}")

    async def test_provider_starts_are_smoothed_and_rolling_window_bounded(self):
        clock = FakeClock()
        queue = RequestQueue(2, clock=clock.now, sleep=clock.sleep)

        starts = [await queue.wait_for_provider_start() for _ in range(3)]

        self.assertEqual(starts, [0.0, 30.0, 60.0])
        status = await queue.snapshot()
        self.assertEqual(status["provider_requests_started_last_60_seconds"], 2)
        self.assertEqual(status["total_provider_requests_started"], 3)

    async def test_fifo_order_and_bounded_waiting_capacity(self):
        queue = RequestQueue(60, max_pending=2)
        first = await queue.acquire_request()
        order: list[str] = []

        async def queued_request(name: str) -> None:
            lease = await queue.acquire_request()
            try:
                order.append(name)
            finally:
                await lease.release()

        second = asyncio.create_task(queued_request("second"))
        await self._wait_for_pending(queue, 1)
        third = asyncio.create_task(queued_request("third"))
        await self._wait_for_pending(queue, 2)

        with self.assertRaises(RequestQueueFull):
            await queue.acquire_request()

        await first.release()
        await asyncio.gather(second, third)
        self.assertEqual(order, ["second", "third"])
        status = await queue.snapshot()
        self.assertEqual(status["pending"], 0)
        self.assertEqual(status["queue_full_rejections"], 1)
        self.assertEqual(status["completed_requests"], 3)

    async def test_cancelled_waiter_is_removed_and_does_not_block_queue(self):
        queue = RequestQueue(60, max_pending=2)
        active = await queue.acquire_request()

        async def wait_forever() -> None:
            await queue.acquire_request()

        waiter = asyncio.create_task(wait_forever())
        await self._wait_for_pending(queue, 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        status = await queue.snapshot()
        self.assertEqual(status["pending"], 0)
        self.assertEqual(status["cancelled_while_waiting"], 1)
        await active.release()

    async def test_asgi_middleware_queues_and_returns_protocol_specific_429(self):
        queue = RequestQueue(60, max_pending=1)
        app = FastAPI()
        app.add_middleware(RequestQueueMiddleware, queue=queue)
        first_entered = asyncio.Event()
        finish_first = asyncio.Event()
        processed: list[int] = []

        @app.post("/v1/chat/completions")
        async def chat():
            index = len(processed) + 1
            processed.append(index)
            if index == 1:
                first_entered.set()
                await finish_first.wait()
            return {"processed": index}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v1/chat/completions", json={"n": 1}))
            await asyncio.wait_for(first_entered.wait(), timeout=1)
            second = asyncio.create_task(client.post("/v1/chat/completions", json={"n": 2}))
            await self._wait_for_pending(queue, 1)

            openai_rejection = await client.post("/v1/chat/completions", json={"n": 3})
            self.assertEqual(openai_rejection.status_code, 429)
            self.assertEqual(openai_rejection.json()["error"]["code"], "queue_full")
            self.assertEqual(openai_rejection.headers["x-bridge-queue"], "full")

            anthropic_rejection = await client.post("/v1/messages", json={"n": 4})
            self.assertEqual(anthropic_rejection.status_code, 429)
            self.assertEqual(anthropic_rejection.json()["type"], "error")
            self.assertEqual(
                anthropic_rejection.json()["error"]["type"], "rate_limit_error"
            )

            finish_first.set()
            first_response, second_response = await asyncio.gather(first, second)
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(processed, [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
