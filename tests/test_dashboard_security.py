import base64
import os
import sys
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("dashboard.app.alpaca_client")
@patch("dashboard.app.init_alpaca")
def test_api_account_security(mock_init, mock_client, client):
    """
    Test security for /api/account endpoint.
    1. Verify unauthorized access returns 401.
    2. Verify authorized access returns 200.
    """
    # Mock init_alpaca to return True
    mock_init.return_value = True

    # Mock alpaca_client.get_account
    mock_client.get_account.return_value = {
        "portfolio_value": 100000,
        "cash": 50000,
        "buying_power": 200000,
        "equity": 100000,
        "last_equity": 95000,
    }

    # 1. Unauthorized access
    response = client.get("/api/account")
    assert response.status_code == 401
    assert "Login Required" in response.headers.get("WWW-Authenticate", "")
    assert b"Could not verify your access level" in response.data

    # 2. Authorized access (default credentials: admin:ploutos)
    auth_string = "admin:ploutos"
    creds = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {creds}"}

    response = client.get("/api/account", headers=headers)

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["data"]["portfolio_value"] == 100000


@patch("dashboard.app.alpaca_client")
@patch("dashboard.app.init_alpaca")
def test_close_position_security(mock_init, mock_client, client):
    """
    Test security for critical action /api/close_position/<symbol>.
    """
    mock_init.return_value = True
    mock_client.close_position.return_value = True

    # 1. Unauthorized access
    response = client.post("/api/close_position/AAPL")
    assert response.status_code == 401

    # 2. Authorized access
    auth_string = "admin:ploutos"
    creds = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {creds}"}

    response = client.post("/api/close_position/AAPL", headers=headers)
    assert response.status_code == 200
    assert response.json["success"] is True
