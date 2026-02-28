import base64
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def auth_app():
    with patch.dict(
        os.environ,
        {
            "DASHBOARD_PASSWORD": "test_password",
            "ALPACA_PAPER_API_KEY": "fake_key",
            "ALPACA_PAPER_SECRET_KEY": "fake_secret",
        },
    ):
        # Force reload of dashboard.app to pick up env var
        if "dashboard.app" in sys.modules:
            del sys.modules["dashboard.app"]

        from dashboard.app import app

        app.config["TESTING"] = True
        yield app


def test_unauthenticated_access(auth_app):
    with auth_app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 401


def test_authenticated_access(auth_app):
    with auth_app.test_client() as client:
        auth_header = {
            "Authorization": "Basic " + base64.b64encode(b"user:test_password").decode("utf-8")
        }
        response = client.get("/", headers=auth_header)
        assert response.status_code == 200


def test_excluded_paths(auth_app):
    with auth_app.test_client() as client:
        # These should return 404/Method Not Allowed but not 401
        assert client.get("/static/style.css").status_code != 401
        assert client.get("/socket.io/").status_code != 401
        assert client.get("/api/webhook").status_code != 401


def test_invalid_password(auth_app):
    with auth_app.test_client() as client:
        auth_header = {
            "Authorization": "Basic " + base64.b64encode(b"user:wrong_password").decode("utf-8")
        }
        response = client.get("/", headers=auth_header)
        assert response.status_code == 401
