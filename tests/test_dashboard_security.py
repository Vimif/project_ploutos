import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set env vars before importing app
os.environ["DASHBOARD_USERNAME"] = "testuser"
os.environ["DASHBOARD_PASSWORD"] = "testpass"
os.environ["FLASK_SECRET_KEY"] = "testsecret"
os.environ["ALPACA_PAPER_API_KEY"] = "dummy_key"
os.environ["ALPACA_PAPER_SECRET_KEY"] = "dummy_secret"

# Mock external dependencies to avoid side effects
with patch.dict(
    sys.modules,
    {
        "trading.alpaca_client": MagicMock(),
        "flask_socketio": MagicMock(),
        "gevent": MagicMock(),
    },
):
    # Import app after mocking
    import dashboard.app
    from dashboard.app import app


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.username = "testuser"
        self.password = "testpass"

    def test_unauthorized_access(self):
        """Test accessing protected route without credentials."""
        # /api/account is a good candidate
        response = self.client.get("/api/account")
        # Expect 401 Unauthorized
        self.assertEqual(response.status_code, 401)

    def test_authorized_access(self):
        """Test accessing protected route with correct credentials."""
        import base64

        creds = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        headers = {"Authorization": f"Basic {creds}"}

        # We use patch.object on the imported module to avoid import issues
        with patch.object(dashboard.app, "init_alpaca", return_value=True):
            # We also need to mock the alpaca_client global variable in dashboard.app
            mock_client = MagicMock()
            mock_client.get_account.return_value = {
                "portfolio_value": 10000,
                "cash": 5000,
                "buying_power": 20000,
                "equity": 10000,
                "last_equity": 9000,
            }

            with patch.object(dashboard.app, "alpaca_client", mock_client):
                response = self.client.get("/api/account", headers=headers)
                self.assertEqual(response.status_code, 200)

    def test_wrong_credentials(self):
        """Test accessing protected route with wrong credentials."""
        import base64

        creds = base64.b64encode(b"wrong:pass").decode()
        headers = {"Authorization": f"Basic {creds}"}

        response = self.client.get("/api/account", headers=headers)
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
