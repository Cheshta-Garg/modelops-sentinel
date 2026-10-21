import tempfile
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.auth import get_current_user
from app.db.base import Base
from app.db.session import get_db_session
from app.main import app
from app.models.enums import UserRole
from app.models.user import User
from app.storage.artifact_store import LocalArtifactStore

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

DEFAULT_TEST_ADMIN = User(
    id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
    email="admin@modelops.test",
    full_name="Default Admin User",
    hashed_password="mock_hashed_password",
    role=UserRole.ADMIN,
    is_active=True,
)


@pytest_asyncio.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Provide an isolated in-memory async SQLite engine with all tables created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async session scoped per test."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def async_client(test_engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    """Authenticated async test client with admin credentials override for regression testing."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def override_get_current_user() -> User:
        return DEFAULT_TEST_ADMIN

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def unauthenticated_client(test_engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    """Unauthenticated client requiring genuine JWT bearer tokens for auth verification."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides.pop(get_current_user, None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
def temp_artifact_store() -> AsyncGenerator[LocalArtifactStore, None]:
    """Provide an isolated temporary artifact storage directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LocalArtifactStore(base_dir=Path(temp_dir))
        yield store
