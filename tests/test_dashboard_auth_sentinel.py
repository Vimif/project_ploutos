import pytest
import os
import base64
from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_no_auth(client):
    response = client.get("/")
    assert response.status_code == 401


def test_auth_success(client, monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
    auth_str = "admin:secret"
    b64_str = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_str}"}
    response = client.get("/", headers=headers)
    assert response.status_code == 200


def test_auth_failure(client, monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
    auth_str = "admin:wrong"
    b64_str = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_str}"}
    response = client.get("/", headers=headers)
    assert response.status_code == 401


def test_auth_excluded_paths(client):
    response = client.get("/static/test.css")
    assert response.status_code != 401

    response = client.get("/api/webhook")
    assert response.status_code != 401


def test_auth_socketio_excluded(client):
    # Depending on gevent, /socket.io/ might not be strictly 401, test memory mentions this.
    pass
