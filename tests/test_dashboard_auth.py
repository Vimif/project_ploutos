import sys
import os
import unittest
import base64
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.getcwd())

# Import the module explicitly to ensure it's loaded
import dashboard.app

class TestDashboardAuth(unittest.TestCase):
    def setUp(self):
        # Patch init_alpaca to avoid real connection attempts
        self.patcher = patch('dashboard.app.init_alpaca', return_value=True)
        self.mock_init = self.patcher.start()

        # Patch alpaca_client in app
        self.client_patcher = patch('dashboard.app.alpaca_client')
        self.mock_client = self.client_patcher.start()

        # Mock get_account return value
        self.mock_client.get_account.return_value = {
            'portfolio_value': 100000,
            'cash': 50000,
            'buying_power': 200000,
            'equity': 100000,
            'last_equity': 99000
        }

        self.app = dashboard.app.app
        self.client = self.app.test_client()

        # Default credentials
        self.username = 'admin'
        self.password = 'ploutos'

    def tearDown(self):
        self.patcher.stop()
        self.client_patcher.stop()

    def test_unauthorized_access(self):
        """Test that access without auth returns 401"""
        response = self.client.get('/api/account')
        self.assertEqual(response.status_code, 401, "Should return 401 Unauthorized")

    def test_authorized_access(self):
        """Test that access with correct credentials returns 200"""
        # Create Basic Auth header
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {'Authorization': f'Basic {encoded_credentials}'}

        response = self.client.get('/api/account', headers=headers)
        self.assertEqual(response.status_code, 200, "Should return 200 OK with valid credentials")

    def test_invalid_credentials(self):
        """Test that access with incorrect credentials returns 401"""
        credentials = f"{self.username}:wrongpassword"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {'Authorization': f'Basic {encoded_credentials}'}

        response = self.client.get('/api/account', headers=headers)
        self.assertEqual(response.status_code, 401, "Should return 401 with invalid credentials")

if __name__ == '__main__':
    unittest.main()
