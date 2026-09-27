from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "test",
        "DATABASE_URL": "postgresql+psycopg://examguard:examguard@postgres:5432/examguard",
        "JWT_SECRET": "test-secret-with-at-least-thirty-two-characters",
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
        "UPLOAD_ROOT": "/tmp/examguard-test/uploads",
        "EVIDENCE_ROOT": "/tmp/examguard-test/evidence",
        "MODEL_ROOT": "/tmp/examguard-test/models",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def engine() -> Generator[Engine, None, None]:
    test_engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(test_engine, "connect")
    def enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(test_engine)
        test_engine.dispose()


@pytest.fixture
def db(engine: Engine) -> Generator[Session, None, None]:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def client(db: Session, settings: Settings) -> Generator[TestClient, None, None]:
    application = create_app()

    def override_db() -> Generator[Session, None, None]:
        yield db

    application.dependency_overrides[get_db] = override_db
    application.dependency_overrides[get_settings] = lambda: settings
    with TestClient(application) as test_client:
        yield test_client
