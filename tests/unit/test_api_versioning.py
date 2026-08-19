"""Tests for the exclusive /api/v1 routing surface and version reporting."""

from fastapi.testclient import TestClient

from careflow.core.config import settings
from careflow.core.version import API_VERSION
from careflow.main import app

client = TestClient(app)


def test_root_status():
    """`/` is a JSON service descriptor, not an HTML page.

    It used to serve careflow/static/index.html -- a second frontend competing with the
    Next.js app. That UI is gone and the backend is JSON-only.
    """
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["service"] == settings.APP_NAME
    assert data["api_base"] == "/api/v1"


def test_health_check_api_v1():
    """Liveness probe reports the real package version.

    This previously asserted the literal "1.0.0", which was the value of a broken
    `hasattr(settings, "APP_VERSION")` fallback -- the test was pinning the bug in place
    while `/` simultaneously advertised 2.5.0.
    """
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.APP_NAME
    assert data["version"] == API_VERSION


def test_version_is_consistent_across_endpoints():
    """Regression guard for the three-way version drift."""
    root = client.get("/").json()
    health = client.get("/api/v1/health").json()
    assert root["version"] == health["version"] == API_VERSION


def test_unversioned_api_paths_are_not_served():
    """Only /api/v1 is exposed; unversioned aliases must 404."""
    assert client.get("/api/health").status_code == 404
