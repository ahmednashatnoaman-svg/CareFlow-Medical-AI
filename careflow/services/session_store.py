"""Pluggable session store for multi-turn triage interviews.

Triage sessions previously lived in a bare `dict` on the service singleton, which has two
failure modes:

1. Serverless. Each Vercel invocation may land on a fresh instance, so `/triage/step`
   often could not find the session created by `/triage/start` and silently began a brand
   new empty interview -- the patient's answers vanished every turn with no error.
2. Long-running processes. Nothing ever evicted a session, so the dict grew without bound
   for the lifetime of the process.

`RedisSessionStore` fixes (1) by sharing state across instances; TTL fixes (2) in both
backends. The in-memory backend remains the default for local single-process development.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from careflow.core.config import settings

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Backend for serialized triage session state, keyed by session id."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    def set(self, session_id: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    """Process-local store with TTL and a hard capacity ceiling.

    Correct only for a single-process deployment (local dev, a single container). On
    serverless or multi-replica deployments use RedisSessionStore, or sessions will be
    invisible to the instance that handles the next turn.
    """

    def __init__(self, ttl_seconds: int, max_sessions: int):
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}

    def _purge_expired(self, now: float) -> None:
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            del self._data[k]

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        entry = self._data.get(session_id)
        if entry is None:
            return None
        expires_at, state = entry
        if expires_at <= time.monotonic():
            del self._data[session_id]
            return None
        return state

    def set(self, session_id: str, state: dict[str, Any]) -> None:
        now = time.monotonic()
        self._purge_expired(now)
        # Capacity backstop for the pathological case of more live sessions than the
        # ceiling: evict whatever expires soonest so memory stays bounded.
        if len(self._data) >= self._max and session_id not in self._data:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]
            logger.warning("Session capacity %d reached; evicted %s", self._max, oldest)
        self._data[session_id] = (now + self._ttl, state)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore(SessionStore):
    """Shared store backed by Redis, with TTL handled by Redis key expiry.

    This is the backend that makes multi-turn triage correct on serverless, where
    consecutive turns are not guaranteed to reach the same process.
    """

    _PREFIX = "careflow:triage:"

    def __init__(self, url: str, ttl_seconds: int):
        import redis  # noqa: PLC0415 - optional dependency, only needed for this backend

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        raw = self._client.get(f"{self._PREFIX}{session_id}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt value must not wedge the session forever; drop it and let the
            # caller start a fresh interview.
            logger.warning("Discarding corrupt session payload for %s", session_id)
            self.delete(session_id)
            return None

    def set(self, session_id: str, state: dict[str, Any]) -> None:
        self._client.setex(f"{self._PREFIX}{session_id}", self._ttl, json.dumps(state))

    def delete(self, session_id: str) -> None:
        self._client.delete(f"{self._PREFIX}{session_id}")


def build_session_store() -> SessionStore:
    """Select a backend from configuration, preferring Redis when reachable.

    A Redis connection failure degrades to in-memory rather than refusing to boot: a
    single-instance deployment stays fully functional, and the warning names the real
    consequence so the degradation is not mistaken for a healthy configuration.
    """
    ttl = settings.SESSION_TTL_SECONDS
    if settings.SESSION_BACKEND == "redis" or (
        settings.SESSION_BACKEND == "auto" and settings.REDIS_URL and settings.APP_ENV == "production"
    ):
        try:
            store = RedisSessionStore(settings.REDIS_URL, ttl)
            store._client.ping()
            logger.info("Triage sessions: Redis backend (ttl=%ds)", ttl)
            return store
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail startup
            logger.warning(
                "Redis session backend unavailable (%s); using in-memory sessions. "
                "Multi-turn triage will break across instances if more than one is running.",
                exc,
            )

    logger.info("Triage sessions: in-memory backend (ttl=%ds, max=%d)", ttl, settings.SESSION_MAX)
    return InMemorySessionStore(ttl, settings.SESSION_MAX)
