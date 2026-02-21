import base64
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies that might be missing or require configuration
sys.modules["flask_socketio"] = MagicMock()
sys.modules["gevent"] = MagicMock()
sys.modules["trading.alpaca_client"] = MagicMock()
sys.modules["core.utils"] = MagicMock()

try:
    from dashboard.app import app
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.username = "admin"
        self.password = "testpass"

        # Patch the password in the app module so we know what it is
        self.password_patcher = patch("dashboard.app.DASHBOARD_PASSWORD", self.password)
        self.password_patcher.start()

        # Also patch username just in case it was changed in env
        self.username_patcher = patch("dashboard.app.DASHBOARD_USERNAME", self.username)
        self.username_patcher.start()

    def tearDown(self):
        self.password_patcher.stop()
        self.username_patcher.stop()

    def get_auth_headers(self, username, password):
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded_credentials}"}

    def test_close_position_no_auth(self):
        """Test accessing protected endpoint without credentials returns 401"""
        response = self.app.post("/api/close_position/AAPL")
        self.assertEqual(response.status_code, 401)
        self.assertIn(b"Connexion requise", response.data)

    def test_close_position_invalid_auth(self):
        """Test accessing protected endpoint with wrong credentials returns 401"""
        headers = self.get_auth_headers("admin", "wrongpass")
        response = self.app.post("/api/close_position/AAPL", headers=headers)
        self.assertEqual(response.status_code, 401)

    def test_close_position_valid_auth(self):
        """Test accessing protected endpoint with correct credentials works"""
        headers = self.get_auth_headers(self.username, self.password)

        # Mocking internal logic
        with patch("dashboard.app.alpaca_client") as mock_client:
            mock_client.close_position.return_value = True
            with patch("dashboard.app.init_alpaca", return_value=True):
                response = self.app.post("/api/close_position/AAPL", headers=headers)
                self.assertEqual(response.status_code, 200)

    def test_index_auth(self):
        """Test index page is protected"""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 401)

        headers = self.get_auth_headers(self.username, self.password)
        response = self.app.get("/", headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_api_db_trades_auth(self):
        """Test JSON DB endpoint is protected"""
        response = self.app.get("/api/db/trades")
        self.assertEqual(response.status_code, 401)

        headers = self.get_auth_headers(self.username, self.password)
        response = self.app.get("/api/db/trades", headers=headers)
        # Assuming TRADES_LOG_DIR doesn't exist or is empty, it returns 200 with empty list
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
