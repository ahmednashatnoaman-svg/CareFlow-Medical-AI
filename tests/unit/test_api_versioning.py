"""Unit tests for exclusive /api/v1 routing in history-service."""

import pytest
from fastapi.testclient import TestClient
from careflow.main import app
from careflow.core.config import settings

client = TestClient(app)


def test_root_status():
    """Verify root endpoint GET /."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["service"] == settings.APP_NAME


def test_health_check_api_v1():
    """Verify health endpoint on /api/v1/health."""
    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    data = res_v1.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.APP_NAME
    assert data["version"] == "1.0.0"
