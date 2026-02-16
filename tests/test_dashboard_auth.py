import base64
import os

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    # Save original config
    orig_username = app.config.get("DASHBOARD_USERNAME")
    orig_password = app.config.get("DASHBOARD_PASSWORD")

    # Set default credentials for testing
    app.config["DASHBOARD_USERNAME"] = "admin"
    app.config["DASHBOARD_PASSWORD"] = "ploutos"

    app.config["TESTING"] = True
    # Disable CSRF for testing if it was enabled (it's not, but good practice)
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        yield client

    # Restore config
    if orig_username:
        app.config["DASHBOARD_USERNAME"] = orig_username
    else:
        app.config.pop("DASHBOARD_USERNAME", None)

    if orig_password:
        app.config["DASHBOARD_PASSWORD"] = orig_password
    else:
        app.config.pop("DASHBOARD_PASSWORD", None)


def test_unauthorized_access_root(client):
    """Test that accessing root without credentials returns 401"""
    response = client.get("/")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_unauthorized_access_api(client):
    """Test that accessing API without credentials returns 401"""
    response = client.get("/api/account")
    assert response.status_code == 401


def test_authorized_access(client):
    """Test that accessing root with correct credentials works (returns 200 or 500 but not 401)"""
    username = "admin"
    password = "ploutos"
    creds = f"{username}:{password}"
    b64_creds = base64.b64encode(creds.encode()).decode()
    headers = {"Authorization": f"Basic {b64_creds}"}

    response = client.get("/", headers=headers)
    # 200 is expected for root as it doesn't need alpaca
    assert response.status_code == 200


def test_incorrect_credentials(client):
    """Test that accessing root with incorrect credentials returns 401"""
    username = "admin"
    password = "wrong_password"
    creds = f"{username}:{password}"
    b64_creds = base64.b64encode(creds.encode()).decode()
    headers = {"Authorization": f"Basic {b64_creds}"}

    response = client.get("/", headers=headers)
    assert response.status_code == 401
