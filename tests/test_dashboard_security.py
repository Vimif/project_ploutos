import os
import sys
import base64
import pytest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing the app
# This prevents errors if dependencies are missing and avoids side effects
sys.modules["trading.alpaca_client"] = MagicMock()
sys.modules["flask_socketio"] = MagicMock()
sys.modules["gevent"] = MagicMock()

# Set environment variables for the test
os.environ["DASHBOARD_USERNAME"] = "testuser"
os.environ["DASHBOARD_PASSWORD"] = "testpass"
os.environ["FLASK_SECRET_KEY"] = "testsecret"

# Import the app after mocking
# We need to make sure the app uses the mocked modules
with patch.dict(sys.modules, {"trading.alpaca_client": MagicMock()}):
    from dashboard.app import app, init_alpaca


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Ensure config matches env vars (in case app loaded before env vars set)
    app.config["DASHBOARD_USERNAME"] = "testuser"
    app.config["DASHBOARD_PASSWORD"] = "testpass"

    # Mock init_alpaca to succeed
    with patch("dashboard.app.init_alpaca", return_value=True):
        with app.test_client() as client:
            yield client


def test_unauthorized_access(client):
    """Test that accessing protected routes without auth returns 401."""
    # Currently this will fail (return 200) because auth is not implemented
    response = client.get("/")
    assert response.status_code == 401


def test_api_unauthorized_access(client):
    """Test that accessing API routes without auth returns 401."""
    response = client.get("/api/account")
    assert response.status_code == 401


def test_authorized_access(client):
    """Test that accessing with correct credentials works."""
    # Create Basic Auth header
    creds = base64.b64encode(b"testuser:testpass").decode("utf-8")
    headers = {"Authorization": f"Basic {creds}"}

    response = client.get("/", headers=headers)
    # Depending on implementation, index might return 200 or render template
    # Since we didn't mock render_template, it might try to render and fail if templates missing
    # But usually status code check is enough if it passes auth check
    assert response.status_code == 200


def test_incorrect_password(client):
    """Test that accessing with incorrect password returns 401."""
    creds = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
    headers = {"Authorization": f"Basic {creds}"}

    response = client.get("/", headers=headers)
    assert response.status_code == 401


def test_incorrect_username(client):
    """Test that accessing with incorrect username returns 401."""
    creds = base64.b64encode(b"wronguser:testpass").decode("utf-8")
    headers = {"Authorization": f"Basic {creds}"}

    response = client.get("/", headers=headers)
    assert response.status_code == 401
