"""Dependency Injection module for FastAPI."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from careflow.core.config import settings
from careflow.models.base import Base

logger = logging.getLogger(__name__)

_SQLITE_MEMORY_URL = "sqlite+aiosqlite:///:memory:"


# Database Engine setup (Supports Supabase Cloud & Local Postgres)
def _create_engine_for_url(db_url: Optional[str]):
    if not db_url:
        logger.warning(
            "DATABASE_URL is not configured; using an in-memory SQLite engine. "
            "Set DATABASE_URL (or SUPABASE_*/POSTGRES_* vars) for persistent storage."
        )
        return create_async_engine(_SQLITE_MEMORY_URL, echo=False, future=True)
    connect_args = {}
    if "postgresql" in db_url:
        connect_args["statement_cache_size"] = 0
    if any(k in db_url for k in ["supabase.co", "supabase.com", "pooler.supabase.com"]):
        connect_args["ssl"] = "require"
    return create_async_engine(db_url, connect_args=connect_args, echo=False, future=True)


engine = _create_engine_for_url(settings.DATABASE_URL)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db():
    """Initializes database tables if they do not exist with automatic fallback."""
    global engine, async_session_factory
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.debug(
            "Configured database unreachable; falling back to local SQLite in-memory engine: %s",
            exc,
        )
        # Fallback to SQLite in-memory engine when Supabase Cloud/Postgres is unreachable/unauthenticated
        engine = create_async_engine(_SQLITE_MEMORY_URL, echo=False, future=True)
        async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for obtaining an async session in standalone scripts."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency yielding async database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
