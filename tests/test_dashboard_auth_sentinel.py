import pytest
import os
import base64
from unittest.mock import patch

# Mock AlpacaClient to avoid actual API calls during tests
with patch('trading.alpaca_client.AlpacaClient'):
    from dashboard.app import app, socketio

@pytest.fixture
def client():
    # Make sure we're not running in TESTING mode so auth is enforced
    app.config['TESTING'] = False
    with app.test_client() as client:
        yield client
    # Restore testing mode if needed
    app.config['TESTING'] = True

@pytest.fixture
def auth_headers():
    credentials = b"admin:secretpassword"
    encoded_credentials = base64.b64encode(credentials).decode("utf-8")
    return {"Authorization": f"Basic {encoded_credentials}"}

@pytest.fixture
def env_vars():
    os.environ['DASHBOARD_USERNAME'] = 'admin'
    os.environ['DASHBOARD_PASSWORD'] = 'secretpassword'
    yield
    os.environ.pop('DASHBOARD_USERNAME', None)
    os.environ.pop('DASHBOARD_PASSWORD', None)

def test_dashboard_requires_auth(client):
    """Test that dashboard endpoints require authentication"""
    response = client.get('/')
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == 'Basic realm="Dashboard Login Required"'

def test_dashboard_auth_success(client, env_vars, auth_headers):
    """Test that dashboard endpoints allow access with correct credentials"""
    response = client.get('/', headers=auth_headers)
    assert response.status_code == 200

def test_dashboard_auth_invalid_credentials(client, env_vars):
    """Test that dashboard endpoints reject invalid credentials"""
    credentials = b"admin:wrongpassword"
    encoded_credentials = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    response = client.get('/', headers=headers)
    assert response.status_code == 401

def test_dashboard_auth_missing_config(client, auth_headers):
    """Test that access is denied if environment variables are missing"""
    # Ensure variables are missing
    os.environ.pop('DASHBOARD_USERNAME', None)
    os.environ.pop('DASHBOARD_PASSWORD', None)

    response = client.get('/', headers=auth_headers)
    assert response.status_code == 401

def test_dashboard_options_bypass(client):
    """Test that OPTIONS requests bypass authentication (CORS)"""
    response = client.options('/api/account')
    assert response.status_code in [200, 204]  # Depending on Flask-CORS setup

def test_dashboard_testing_mode_bypass():
    """Test that setting TESTING config bypasses authentication"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
    app.config['TESTING'] = False

def test_socketio_rejects_unauthenticated(env_vars):
    """Test that SocketIO rejects unauthenticated connections"""
    app.config['TESTING'] = False
    client = socketio.test_client(app)
    # The connection should not be established due to returning False
    assert not client.is_connected()

def test_socketio_allows_testing_mode():
    """Test that SocketIO allows connections in TESTING mode"""
    app.config['TESTING'] = True
    client = socketio.test_client(app)
    # The connection should be established in testing mode
    assert client.is_connected()
    app.config['TESTING'] = False
