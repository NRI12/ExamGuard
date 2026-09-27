from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "assets").mkdir()
    index = "<!doctype html><title>ExamGuard</title>"
    (tmp_path / "index.html").write_text(index, encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    for key, value in {
        "APP_ENV": "test",
        "DATABASE_URL": "postgresql+psycopg://examguard:examguard@localhost:5432/examguard",
        "JWT_SECRET": "test-secret-with-at-least-thirty-two-characters",
        "UPLOAD_ROOT": str(tmp_path / "uploads"),
        "EVIDENCE_ROOT": str(tmp_path / "evidence"),
        "MODEL_ROOT": str(tmp_path / "models"),
        "FRONTEND_DIST": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_backend_serves_the_built_frontend_with_spa_fallback(dist: Path) -> None:
    client = TestClient(create_app())
    assert "ExamGuard" in client.get("/").text
    assert "ExamGuard" in client.get("/monitoring?session=1").text  # client-side route
    asset = client.get("/assets/app.js")
    assert asset.text == "console.log(1)" and "immutable" in asset.headers["cache-control"]
    assert client.get("/api/v1/does-not-exist").status_code == 404
    assert client.get("/../../etc/passwd").status_code in (200, 404)
    assert "root:" not in client.get("/..%2F..%2Fetc%2Fpasswd").text


def test_cookie_secure_follows_the_environment_unless_overridden(
    dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    assert get_settings().secure_cookies is True
    monkeypatch.setenv("COOKIE_SECURE", "false")
    get_settings.cache_clear()
    assert get_settings().secure_cookies is False
