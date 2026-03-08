import pytest
import os
import base64
from dashboard.app import app, socketio


@pytest.fixture
def client():
    # Set standard testing configuration
    app.config["TESTING"] = False

    # Store old env vars
    old_username = os.environ.get("DASHBOARD_USERNAME")
    old_password = os.environ.get("DASHBOARD_PASSWORD")

    os.environ["DASHBOARD_USERNAME"] = "admin"
    os.environ["DASHBOARD_PASSWORD"] = "secretpassword"

    with app.test_client() as client:
        yield client

    # Restore old env vars
    if old_username is not None:
        os.environ["DASHBOARD_USERNAME"] = old_username
    else:
        os.environ.pop("DASHBOARD_USERNAME", None)

    if old_password is not None:
        os.environ["DASHBOARD_PASSWORD"] = old_password
    else:
        os.environ.pop("DASHBOARD_PASSWORD", None)


def get_auth_headers(username, password):
    auth_string = f"{username}:{password}"
    base64_auth = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {base64_auth}"}


def test_dashboard_auth_missing_credentials_fails(client):
    """Test that requests fail with 401 when no auth is provided"""
    response = client.get("/api/account")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == 'Basic realm="Login Required"'


def test_dashboard_auth_invalid_credentials_fails(client):
    """Test that requests fail with 401 when invalid auth is provided"""
    headers = get_auth_headers("admin", "wrongpassword")
    response = client.get("/api/account", headers=headers)
    assert response.status_code == 401


def test_dashboard_auth_valid_credentials_succeeds(client):
    """Test that requests pass auth with correct credentials"""
    headers = get_auth_headers("admin", "secretpassword")
    # Use an endpoint that might return 500 due to no alpaca client in test,
    # but at least it shouldn't return 401.
    response = client.get("/api/account", headers=headers)
    assert response.status_code != 401


def test_dashboard_auth_static_exempt(client):
    """Test that static paths are exempt from auth"""
    response = client.get("/static/test.css")
    assert response.status_code != 401


def test_dashboard_auth_webhook_exempt(client):
    """Test that webhook paths are exempt from auth"""
    response = client.post("/api/webhook/test")
    assert response.status_code != 401


def test_dashboard_auth_options_exempt(client):
    """Test that OPTIONS requests for CORS are exempt"""
    response = client.options("/api/account")
    assert response.status_code != 401


def test_dashboard_auth_testing_mode(client):
    """Test that TESTING mode bypasses auth"""
    app.config["TESTING"] = True
    response = client.get("/api/account")
    assert response.status_code != 401
    app.config["TESTING"] = False  # Reset


def test_dashboard_auth_env_vars_missing(client):
    """Test that if the env vars are missing, we fail securely"""
    os.environ.pop("DASHBOARD_USERNAME", None)
    os.environ.pop("DASHBOARD_PASSWORD", None)

    headers = get_auth_headers("admin", "secretpassword")
    response = client.get("/api/account", headers=headers)
    assert response.status_code == 401
