"""Pytest Configuration and Fixtures."""

import asyncio
import pytest
from careflow.core.dependencies.deps import get_db
from careflow.main import app
from careflow.models.base import Base

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="function")
def db_session():
    """Fixture providing in-memory async SQLite database session for unit/integration tests."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    loop = asyncio.get_event_loop()

    async def init_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop.run_until_complete(init_tables())

    session = session_factory()
    yield session

    async def drop_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    loop.run_until_complete(session.close())
    loop.run_until_complete(drop_tables())


@pytest.fixture(scope="function")
def client(db_session):
    """Fixture providing TestClient for FastAPI endpoint testing."""
    from fastapi.testclient import TestClient

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
