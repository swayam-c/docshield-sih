"""Simple in-memory rate limiter for mutating upload endpoints."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict

from fastapi import HTTPException, Request, status

from app.config import get_settings

_lock = Lock()
_hits: Dict[str, Deque[float]] = defaultdict(deque)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    limit = int(settings.rate_limit_per_minute)
    if limit <= 0:
        return
    key = client_key(request)
    now = time.time()
    window = 60.0
    with _lock:
        q = _hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded. Try again shortly.",
                    "code": "RATE_LIMIT",
                },
            )
        q.append(now)
