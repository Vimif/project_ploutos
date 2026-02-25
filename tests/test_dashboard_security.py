import sys
import os
import unittest
import base64
from unittest.mock import MagicMock, patch

# Mock dependencies to avoid side effects and missing credentials
# We must do this BEFORE importing dashboard.app

# Mock core.utils
sys.modules['core'] = MagicMock()
sys.modules['core.utils'] = MagicMock()
sys.modules['core.utils'].setup_logging = MagicMock(return_value=MagicMock())

# Mock trading.alpaca_client
mock_alpaca_module = MagicMock()
sys.modules['trading'] = MagicMock()
sys.modules['trading.alpaca_client'] = mock_alpaca_module

# Add project root to path so dashboard.app can find other modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.app import app
import dashboard.app as dashboard_module

class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config['TESTING'] = True

        # Mock the Alpaca client to return valid data
        self.mock_alpaca = MagicMock()
        self.mock_alpaca.get_account.return_value = {
            'portfolio_value': 100000.0,
            'cash': 50000.0,
            'buying_power': 200000.0,
            'equity': 100000.0,
            'last_equity': 99000.0
        }
        # Inject the mock client directly into the module
        dashboard_module.alpaca_client = self.mock_alpaca

    def test_unauthorized_access(self):
        """Test that sensitive endpoints require authentication."""
        # Attempt to access a sensitive endpoint without auth
        response = self.client.get('/api/account')

        # Should return 401 Unauthorized
        self.assertEqual(response.status_code, 401, "Endpoint /api/account should require authentication")
        self.assertIn('WWW-Authenticate', response.headers)

    def test_authorized_access(self):
        """Test that access is granted with correct credentials."""
        # Get the current credentials (even if generated)
        username = dashboard_module.DASHBOARD_USERNAME
        password = dashboard_module.DASHBOARD_PASSWORD

        # Create Basic Auth header
        auth_str = f"{username}:{password}"
        auth_bytes = auth_str.encode('ascii')
        base64_bytes = base64.b64encode(auth_bytes)
        base64_str = base64_bytes.decode('ascii')
        headers = {'Authorization': f'Basic {base64_str}'}

        response = self.client.get('/api/account', headers=headers)

        # Should return 200 OK
        self.assertEqual(response.status_code, 200, "Should allow access with correct credentials")

if __name__ == '__main__':
    unittest.main()
