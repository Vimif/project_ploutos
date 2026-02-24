import base64
import os
import unittest

# Set env vars BEFORE importing app
os.environ["DASHBOARD_USERNAME"] = "testuser"
os.environ["DASHBOARD_PASSWORD"] = "testpass"
os.environ["FLASK_SECRET_KEY"] = "testing"
# Avoid Alpaca credentials check failing
os.environ["ALPACA_PAPER_API_KEY"] = "fake_key"
os.environ["ALPACA_PAPER_SECRET_KEY"] = "fake_secret"

# Mock dependencies if they are missing or problematic
# We'll try to import app directly first.
try:
    from dashboard.app import app
except ImportError:
    # If dependencies are missing in test env, we might need to mock them
    # But since we installed them, we expect success.
    raise


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_access_without_auth(self):
        """Test that accessing root without auth returns 401"""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 401)
        self.assertIn("WWW-Authenticate", response.headers)
        self.assertIn('Basic realm="Ploutos Dashboard Login"', response.headers["WWW-Authenticate"])

    def test_access_with_wrong_credentials(self):
        """Test that accessing with wrong credentials returns 401"""
        # Create Basic Auth header with wrong password
        creds = base64.b64encode(b"testuser:wrongpass").decode("utf-8")
        headers = {"Authorization": f"Basic {creds}"}

        response = self.app.get("/", headers=headers)
        self.assertEqual(response.status_code, 401)

    def test_access_with_correct_credentials(self):
        """Test that accessing with correct credentials DOES NOT return 401"""
        # Create Basic Auth header
        creds = base64.b64encode(b"testuser:testpass").decode("utf-8")
        headers = {"Authorization": f"Basic {creds}"}

        response = self.app.get("/", headers=headers)

        # We expect 200 (if template renders) or 500 (if alpaca init fails or template error)
        # But definitively NOT 401
        self.assertNotEqual(response.status_code, 401)
        # Verify we got through auth
        if response.status_code == 200:
            # Check if we got the dashboard
            self.assertIn(b"Ploutos Dashboard", response.data)


if __name__ == "__main__":
    unittest.main()
