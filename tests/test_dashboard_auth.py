import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import base64

# Mock trading.alpaca_client before importing dashboard.app
sys.modules['trading.alpaca_client'] = MagicMock()
sys.modules['trading.alpaca_client'].AlpacaClient = MagicMock()

# Set environment variables for auth
os.environ['DASHBOARD_USERNAME'] = 'admin'
os.environ['DASHBOARD_PASSWORD'] = 'secret'
os.environ['FLASK_SECRET_KEY'] = 'test_secret'

# Import app after mocking
from dashboard.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Patch init_alpaca to avoid real initialization and set global client
    with patch('dashboard.app.init_alpaca', return_value=True):
        # We also need to set the global alpaca_client variable in app.py
        # because the routes use it directly.
        # But wait, init_alpaca sets the global variable.
        # So if we patch it, the global variable might remain None unless we set it.
        # However, the routes check `if not alpaca_client: if not init_alpaca(): ...`
        # If init_alpaca returns True, it assumes client is set.
        # So we should probably set dashboard.app.alpaca_client manually.
        import dashboard.app
        dashboard.app.alpaca_client = MagicMock()

        # Configure mock client to return dummy data so we don't get 500 errors
        dashboard.app.alpaca_client.get_account.return_value = {
            'portfolio_value': 10000,
            'cash': 5000,
            'buying_power': 20000,
            'equity': 10000,
            'last_equity': 9000
        }

        with app.test_client() as client:
            yield client

def test_unauthorized_access_api(client):
    """Test that accessing API without auth returns 401"""
    response = client.get('/api/account')
    # EXPECTED FAILURE: Currently returns 200 because no auth is implemented
    assert response.status_code == 401

def test_authorized_access_api(client):
    """Test that accessing API with correct credentials returns 200"""
    # Create Basic Auth header
    creds = base64.b64encode(b"admin:secret").decode('utf-8')
    headers = {'Authorization': f'Basic {creds}'}

    response = client.get('/api/account', headers=headers)
    assert response.status_code == 200

def test_wrong_credentials_api(client):
    """Test that accessing API with wrong credentials returns 401"""
    creds = base64.b64encode(b"admin:wrong").decode('utf-8')
    headers = {'Authorization': f'Basic {creds}'}

    response = client.get('/api/account', headers=headers)
    assert response.status_code == 401

def test_unauthorized_access_index(client):
    """Test that accessing index page without auth returns 401"""
    response = client.get('/')
    assert response.status_code == 401
