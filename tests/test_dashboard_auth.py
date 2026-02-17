import base64
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock AlpacaClient globally to prevent side effects on import
sys.modules["trading.alpaca_client"] = MagicMock()

# Set env vars BEFORE importing app to ensure they are picked up
os.environ["DASHBOARD_USERNAME"] = "testuser"
os.environ["DASHBOARD_PASSWORD"] = "testpass"

# Now import the app
# Note: If dashboard.app was already imported, this might use the cached version with old env vars.
# But since we run this as a script/pytest in a fresh process, it should be fine.
from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_unauthorized_access(client):
    """Test that endpoints require auth (FIXED)"""
    # Expect 401 Unauthorized
    response = client.get("/")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert 'Basic realm="Login Required"' in response.headers["WWW-Authenticate"]

    response = client.get("/api/account")
    assert response.status_code == 401


def test_authorized_access(client):
    """Test that endpoints are accessible with correct credentials"""
    # Create Basic Auth header
    creds = "testuser:testpass"
    auth_header = {"Authorization": f"Basic {base64.b64encode(creds.encode()).decode()}"}

    # Should be 200 OK
    response = client.get("/", headers=auth_header)
    assert response.status_code == 200


def test_bad_credentials(client):
    """Test that bad credentials return 401"""
    creds = "testuser:wrongpass"
    auth_header = {"Authorization": f"Basic {base64.b64encode(creds.encode()).decode()}"}

    response = client.get("/", headers=auth_header)
    assert response.status_code == 401
