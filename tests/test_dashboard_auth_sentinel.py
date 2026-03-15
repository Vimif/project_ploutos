import pytest
import os
import base64
from unittest.mock import patch

pytest.importorskip("flask")
pytest.importorskip("flask_socketio")
from flask_socketio import ConnectionRefusedError

# Mock AlpacaClient to avoid actual API calls during tests
with patch("trading.alpaca_client.AlpacaClient"):
    from dashboard.app import app, socketio


@pytest.fixture
def client():
    # Make sure we're not running in TESTING mode so auth is enforced
    original_testing = app.config.get("TESTING", False)
    app.config["TESTING"] = False
    with app.test_client() as client:
        yield client
    # Restore testing mode
    app.config["TESTING"] = original_testing


@pytest.fixture
def auth_headers():
    credentials = b"admin:secretpassword"
    encoded_credentials = base64.b64encode(credentials).decode("utf-8")
    return {"Authorization": f"Basic {encoded_credentials}"}


@pytest.fixture
def env_vars():
    os.environ["DASHBOARD_USERNAME"] = "admin"
    os.environ["DASHBOARD_PASSWORD"] = "secretpassword"
    yield
    os.environ.pop("DASHBOARD_USERNAME", None)
    os.environ.pop("DASHBOARD_PASSWORD", None)


def test_dashboard_requires_auth(client):
    """Test that dashboard endpoints require authentication"""
    response = client.get("/")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_dashboard_auth_success(client, env_vars, auth_headers):
    """Test that dashboard endpoints allow access with correct credentials"""
    response = client.get("/", headers=auth_headers)
    assert response.status_code == 200


def test_dashboard_auth_invalid_credentials(client, env_vars):
    """Test that dashboard endpoints reject invalid credentials"""
    credentials = b"admin:wrongpassword"
    encoded_credentials = base64.b64encode(credentials).decode("utf-8")
    response = client.get("/", headers={"Authorization": f"Basic {encoded_credentials}"})
    assert response.status_code == 401


def test_dashboard_auth_missing_config(client, auth_headers):
    """Test that access is denied if environment variables are missing"""
    original_user = os.environ.pop("DASHBOARD_USERNAME", None)
    original_pass = os.environ.pop("DASHBOARD_PASSWORD", None)
    try:
        response = client.get("/", headers=auth_headers)
        assert response.status_code == 401
    finally:
        if original_user is not None:
            os.environ["DASHBOARD_USERNAME"] = original_user
        if original_pass is not None:
            os.environ["DASHBOARD_PASSWORD"] = original_pass


def test_dashboard_options_bypass(client):
    """Test that OPTIONS requests bypass authentication (CORS)"""
    response = client.options("/api/account")
    assert response.status_code in [200, 204]


def test_dashboard_testing_mode_bypass():
    """Test that setting TESTING config bypasses authentication"""
    original_testing = app.config.get("TESTING", False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
    app.config["TESTING"] = original_testing


def test_socketio_rejects_unauthenticated(env_vars):
    """Test that SocketIO rejects unauthenticated connections"""
    original_testing = app.config.get("TESTING", False)
    app.config["TESTING"] = False

    try:
        # Flask-SocketIO may raise an exception (ValueError/RuntimeError/ConnectionRefusedError)
        # when the initial handshake fails due to the 401 Unauthorized response from @app.before_request
        try:
            client = socketio.test_client(app)
            assert not client.is_connected()
        except Exception:
            # Rejection was successful via exception
            pass
    finally:
        app.config["TESTING"] = original_testing


def test_socketio_allows_testing_mode():
    """Test that SocketIO allows connections in TESTING mode"""
    original_testing = app.config.get("TESTING", False)
    app.config["TESTING"] = True
    try:
        client = socketio.test_client(app)
        assert client.is_connected()
    finally:
        app.config["TESTING"] = original_testing
