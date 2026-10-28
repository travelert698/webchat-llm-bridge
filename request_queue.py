"""Bounded FIFO admission and completion-based cooldowns for the local bridge.

The browser backends have one persistent page and mutable per-turn state. This
scheduler therefore admits one generation request at a time, keeps concurrent
client requests in FIFO order, and applies a configurable cooldown after each
provider stream completes and again after the full client response is sent.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

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
    """A granted turn in the FIFO client-request queue."""

    _owner: "RequestQueue"
    _token: object
    _released: bool = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            # Keep admission state consistent if a client disconnect cancels
            # the ASGI task while its response is being finalized.
            await asyncio.shield(self._owner._release_request(self._token))


@dataclass
class ProviderLease:
    """A granted turn to use the single browser-backed provider stream."""

    _owner: "RequestQueue"
    _token: object
    _released: bool = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await asyncio.shield(self._owner._release_provider_call(self._token))


class RequestQueue:
    """Fair FIFO request gate with a post-completion cooldown at both layers."""

    def __init__(
        self,
        inter_request_delay_seconds: float = 3.0,
        max_pending: int = 100,
    ) -> None:
        if isinstance(inter_request_delay_seconds, bool):
            raise ValueError("inter_request_delay_seconds must be a finite non-negative number")
        try:
            delay = float(inter_request_delay_seconds)
        except (TypeError, ValueError):
            raise ValueError("inter_request_delay_seconds must be a finite non-negative number") from None
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("inter_request_delay_seconds must be a finite non-negative number")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")

        self.inter_request_delay_seconds = delay
        self.max_pending = max_pending
        self._clock = time.monotonic

        self._request_condition = asyncio.Condition()
        self._waiting_requests: deque[object] = deque()
        self._active_request_token: object | None = None
        self._active_request_since: float | None = None
        self._next_request_at = 0.0
        self._accepted_requests = 0
        self._completed_requests = 0
        self._rejected_requests = 0
        self._cancelled_request_waiters = 0

        self._provider_condition = asyncio.Condition()
        self._waiting_provider_calls: deque[object] = deque()
        self._active_provider_token: object | None = None
        self._active_provider_since: float | None = None
        self._next_provider_at = 0.0
        self._provider_calls_started = 0
        self._provider_calls_completed = 0
        self._cancelled_provider_waiters = 0

    @staticmethod
    async def _wait_for_condition(
        condition: asyncio.Condition,
        timeout: float | None = None,
    ) -> None:
        if timeout is None:
            await condition.wait()
            return
        try:
            await asyncio.wait_for(condition.wait(), timeout=max(timeout, 0.0001))
        except asyncio.TimeoutError:
            # The caller rechecks its FIFO/cooldown predicate while holding the
            # condition lock; timeouts are normal when a cooldown expires.
            pass

    async def acquire_request(self) -> RequestLease:
        """Wait for a FIFO client-request turn, rejecting only at queue capacity."""
        token = object()
        async with self._request_condition:
            if len(self._waiting_requests) >= self.max_pending:
                self._rejected_requests += 1
                raise RequestQueueFull(len(self._waiting_requests), self.max_pending)
            self._waiting_requests.append(token)
            self._accepted_requests += 1
            try:
                while True:
                    is_head = bool(self._waiting_requests) and self._waiting_requests[0] is token
                    if self._active_request_token is None and is_head:
                        wait = self._next_request_at - self._clock()
                        if wait <= 0:
                            self._waiting_requests.popleft()
                            self._active_request_token = token
                            self._active_request_since = self._clock()
                            return RequestLease(self, token)
                        await self._wait_for_condition(self._request_condition, wait)
                    else:
                        await self._request_condition.wait()
            except BaseException:
                try:
                    self._waiting_requests.remove(token)
                    self._cancelled_request_waiters += 1
                except ValueError:
                    # Once promoted, middleware's finally block releases the lease.
                    pass
                self._request_condition.notify_all()
                raise

    async def _release_request(self, token: object) -> None:
        async with self._request_condition:
            if self._active_request_token is token:
                self._active_request_token = None
                self._active_request_since = None
                self._completed_requests += 1
                # The cooldown starts after the complete ASGI response body has
                # been sent, not when the provider request merely began.
                self._next_request_at = self._clock() + self.inter_request_delay_seconds
                self._request_condition.notify_all()

    async def acquire_provider_call(self) -> ProviderLease:
        """Acquire the browser stream, honoring FIFO and post-stream cooldown."""
        token = object()
        async with self._provider_condition:
            self._waiting_provider_calls.append(token)
            try:
                while True:
                    is_head = (
                        bool(self._waiting_provider_calls)
                        and self._waiting_provider_calls[0] is token
                    )
                    if self._active_provider_token is None and is_head:
                        wait = self._next_provider_at - self._clock()
                        if wait <= 0:
                            self._waiting_provider_calls.popleft()
                            self._active_provider_token = token
                            self._active_provider_since = self._clock()
                            self._provider_calls_started += 1
                            return ProviderLease(self, token)
                        await self._wait_for_condition(self._provider_condition, wait)
                    else:
                        await self._provider_condition.wait()
            except BaseException:
                try:
                    self._waiting_provider_calls.remove(token)
                    self._cancelled_provider_waiters += 1
                except ValueError:
                    pass
                self._provider_condition.notify_all()
                raise

    async def _release_provider_call(self, token: object) -> None:
        async with self._provider_condition:
            if self._active_provider_token is token:
                self._active_provider_token = None
                self._active_provider_since = None
                self._provider_calls_completed += 1
                # This timer begins only after the previous browser response
                # stream has completely ended or has been cancelled/aborted.
                self._next_provider_at = self._clock() + self.inter_request_delay_seconds
                self._provider_condition.notify_all()

    @asynccontextmanager
    async def provider_call(self) -> AsyncIterator[None]:
        """Hold the provider slot for the full stream and always release it."""
        lease = await self.acquire_provider_call()
        try:
            yield
        finally:
            await lease.release()

    async def snapshot(self) -> dict[str, int | bool | float | None]:
        """Return non-sensitive queue, active-stream, and cooldown information."""
        now = self._clock()
        async with self._request_condition:
            pending = len(self._waiting_requests)
            request_active = self._active_request_token is not None
            request_active_seconds = (
                max(0.0, now - self._active_request_since)
                if self._active_request_since is not None
                else None
            )
            accepted = self._accepted_requests
            completed = self._completed_requests
            rejected = self._rejected_requests
            cancelled_requests = self._cancelled_request_waiters
            request_cooldown = max(0.0, self._next_request_at - now)

        async with self._provider_condition:
            provider_waiting = len(self._waiting_provider_calls)
            provider_active = self._active_provider_token is not None
            provider_active_seconds = (
                max(0.0, now - self._active_provider_since)
                if self._active_provider_since is not None
                else None
            )
            provider_started = self._provider_calls_started
            provider_completed = self._provider_calls_completed
            cancelled_provider = self._cancelled_provider_waiters
            provider_cooldown = max(0.0, self._next_provider_at - now)

        return {
            "inter_request_delay_seconds": self.inter_request_delay_seconds,
            "max_pending": self.max_pending,
            "pending": pending,
            "active": request_active,
            "active_seconds": round(request_active_seconds, 3)
            if request_active_seconds is not None else None,
            "request_cooldown_remaining_seconds": round(request_cooldown, 3),
            "accepted_requests": accepted,
            "completed_requests": completed,
            "queue_full_rejections": rejected,
            "cancelled_while_waiting": cancelled_requests,
            "provider_active": provider_active,
            "provider_active_seconds": round(provider_active_seconds, 3)
            if provider_active_seconds is not None else None,
            "provider_waiting": provider_waiting,
            "provider_cooldown_remaining_seconds": round(provider_cooldown, 3),
            "provider_calls_started": provider_started,
            "provider_calls_completed": provider_completed,
            "cancelled_provider_waiters": cancelled_provider,
        }


class RequestQueueMiddleware:
    """FIFO-gate generation requests for the full ASGI response lifetime."""

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
            # Starlette consumes streaming iterators during this ASGI call, so
            # release starts the client-facing cooldown after the final body.
            await lease.release()
