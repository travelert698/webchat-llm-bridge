"""Bounded FIFO admission and per-provider request pacing for the local bridge.

The bridge's browser modules have single-session state. This scheduler therefore
admits one model-generation HTTP request per server at a time, queues later
OpenAI/Anthropic generation requests in FIFO order, and limits every underlying
provider submission (including tool-repair attempts) to the configured RPM.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestQueueFull(Exception):
    """Raised when the bounded number of waiting client requests is reached."""

    def __init__(self, pending: int, capacity: int) -> None:
        self.pending = pending
        self.capacity = capacity
        super().__init__(f"request queue full ({pending}/{capacity} waiting)")


@dataclass
class RequestLease:
    """A granted turn in the FIFO request queue."""

    _owner: "RequestQueue"
    _token: object
    _released: bool = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            # Keep admission state consistent if the client disconnect cancels
            # the ASGI task while its response is being finalized.
            await asyncio.shield(self._owner._release_request(self._token))


class RequestQueue:
    """Fair in-process request gate plus smooth rolling-window RPM limiter."""

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        requests_per_minute: int,
        max_pending: int = 100,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if isinstance(requests_per_minute, bool) or requests_per_minute < 1:
            raise ValueError("requests_per_minute must be a positive integer")
        if isinstance(max_pending, bool) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.requests_per_minute = int(requests_per_minute)
        self.max_pending = int(max_pending)
        self._clock = clock
        self._sleep = sleep

        self._condition = asyncio.Condition()
        self._waiting: deque[object] = deque()
        self._active_token: object | None = None
        self._active_since: float | None = None
        self._accepted = 0
        self._completed = 0
        self._rejected = 0
        self._cancelled_while_waiting = 0

        self._rate_lock = asyncio.Lock()
        self._provider_starts: deque[float] = deque()
        self._next_provider_start: float | None = None
        self._provider_requests_started = 0
        self._provider_interval = self.WINDOW_SECONDS / self.requests_per_minute

    async def acquire_request(self) -> RequestLease:
        """Wait for a FIFO turn; fail fast only when the pending queue is full."""
        token = object()
        async with self._condition:
            if len(self._waiting) >= self.max_pending:
                self._rejected += 1
                raise RequestQueueFull(len(self._waiting), self.max_pending)
            self._waiting.append(token)
            self._accepted += 1
            try:
                await self._condition.wait_for(
                    lambda: (
                        self._active_token is None
                        and bool(self._waiting)
                        and self._waiting[0] is token
                    )
                )
                self._waiting.popleft()
                self._active_token = token
                self._active_since = self._clock()
                return RequestLease(self, token)
            except BaseException:
                try:
                    self._waiting.remove(token)
                    self._cancelled_while_waiting += 1
                except ValueError:
                    # The ticket may already have been promoted when a task is
                    # cancelled; middleware's finally block releases its lease.
                    pass
                self._condition.notify_all()
                raise

    async def _release_request(self, token: object) -> None:
        async with self._condition:
            if self._active_token is token:
                self._active_token = None
                self._active_since = None
                self._completed += 1
                self._condition.notify_all()

    async def wait_for_provider_start(self) -> float:
        """Charge one provider submission and wait until it may safely begin.

        Combines smooth pacing (no burst at process start) with a rolling
        60-second count. Returns the monotonic start time for observability.
        A cancelled wait does not consume quota.
        """
        while True:
            async with self._rate_lock:
                now = self._clock()
                while self._provider_starts and now - self._provider_starts[0] >= self.WINDOW_SECONDS:
                    self._provider_starts.popleft()

                earliest = self._next_provider_start
                if len(self._provider_starts) >= self.requests_per_minute:
                    window_expiry = self._provider_starts[0] + self.WINDOW_SECONDS
                    earliest = window_expiry if earliest is None else max(earliest, window_expiry)

                delay = 0.0 if earliest is None else earliest - now
                if delay <= 1e-9:
                    started_at = now
                    self._provider_starts.append(started_at)
                    self._next_provider_start = started_at + self._provider_interval
                    self._provider_requests_started += 1
                    return started_at

            await self._sleep(delay)

    async def snapshot(self) -> dict[str, int | bool | float | None]:
        """Return a small, prompt-free queue/rate-limit status snapshot."""
        async with self._condition:
            pending = len(self._waiting)
            active = self._active_token is not None
            active_seconds = (
                max(0.0, self._clock() - self._active_since)
                if self._active_since is not None
                else None
            )
            accepted = self._accepted
            completed = self._completed
            rejected = self._rejected
            cancelled = self._cancelled_while_waiting

        async with self._rate_lock:
            now = self._clock()
            while self._provider_starts and now - self._provider_starts[0] >= self.WINDOW_SECONDS:
                self._provider_starts.popleft()
            next_start = self._next_provider_start
            if len(self._provider_starts) >= self.requests_per_minute:
                window_expiry = self._provider_starts[0] + self.WINDOW_SECONDS
                next_start = window_expiry if next_start is None else max(next_start, window_expiry)
            wait = 0.0 if next_start is None else max(0.0, next_start - now)
            starts_last_60 = len(self._provider_starts)
            total_provider_starts = self._provider_requests_started

        return {
            "requests_per_minute": self.requests_per_minute,
            "provider_requests_started_last_60_seconds": starts_last_60,
            "total_provider_requests_started": total_provider_starts,
            "minimum_spacing_seconds": round(self._provider_interval, 3),
            "next_provider_start_in_seconds": round(wait, 3),
            "max_pending": self.max_pending,
            "pending": pending,
            "active": active,
            "active_seconds": round(active_seconds, 3) if active_seconds is not None else None,
            "accepted_requests": accepted,
            "completed_requests": completed,
            "queue_full_rejections": rejected,
            "cancelled_while_waiting": cancelled,
        }


class RequestQueueMiddleware:
    """FIFO-gate model-generation requests for the full ASGI response lifetime."""

    QUEUED_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})

    def __init__(self, app: ASGIApp, queue: RequestQueue) -> None:
        self.app = app
        self.queue = queue

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method", "").upper() != "POST"
            or scope.get("path") not in self.QUEUED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        try:
            lease = await self.queue.acquire_request()
        except RequestQueueFull as exc:
            message = (
                f"The local request queue is full ({exc.pending} of "
                f"{exc.capacity} pending). Retry later with backoff."
            )
            headers = {"Cache-Control": "no-store", "X-Bridge-Queue": "full"}
            if scope.get("path") == "/v1/messages":
                response = JSONResponse(
                    status_code=429,
                    headers=headers,
                    content={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": message,
                        },
                    },
                )
            else:
                response = JSONResponse(
                    status_code=429,
                    headers=headers,
                    content={
                        "error": {
                            "message": message,
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "queue_full",
                        }
                    },
                )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            # Starlette fully consumes streaming iterators inside the ASGI call.
            # Releasing here therefore keeps the browser slot through stream end,
            # and also releases it on exceptions or client disconnects.
            await lease.release()
