from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_db
from app.main import create_app


def test_health_reports_application_and_database(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"application": "ok", "database": "ok"}


def test_health_reports_database_unavailable() -> None:
    class UnavailableSession:
        def execute(self, _: object) -> None:
            raise OperationalError("SELECT 1", {}, RuntimeError("database unavailable"))

    application = create_app()

    def unavailable_db() -> Generator[UnavailableSession, None, None]:
        yield UnavailableSession()

    application.dependency_overrides[get_db] = unavailable_db
    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {"application": "ok", "database": "unavailable"}
