import pytest
import os
import base64
from dashboard.app import app, socketio

@pytest.fixture
def client():
    # Make sure testing bypass is disabled to test actual auth logic
    app.config['TESTING'] = False
    with app.test_client() as client:
        yield client

def test_missing_env_credentials(client, monkeypatch):
    """Test failing securely when environment variables are missing."""
    monkeypatch.delenv('DASHBOARD_USERNAME', raising=False)
    monkeypatch.delenv('DASHBOARD_PASSWORD', raising=False)

    response = client.get('/')
    assert response.status_code == 401
    assert b'Authentication configuration missing' in response.data

def test_missing_auth_header(client, monkeypatch):
    """Test access denied when no authentication is provided."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret')

    response = client.get('/')
    assert response.status_code == 401
    assert b'Could not verify your access level' in response.data

def test_invalid_credentials(client, monkeypatch):
    """Test access denied with wrong credentials."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret')

    auth_bytes = b'admin:wrongpass'
    auth_header = 'Basic ' + base64.b64encode(auth_bytes).decode('ascii')

    response = client.get('/', headers={'Authorization': auth_header})
    assert response.status_code == 401
    assert b'Invalid credentials' in response.data

def test_valid_credentials(client, monkeypatch):
    """Test successful access with correct credentials."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret')

    auth_bytes = b'admin:secret'
    auth_header = 'Basic ' + base64.b64encode(auth_bytes).decode('ascii')

    response = client.get('/', headers={'Authorization': auth_header})
    assert response.status_code == 200

def test_excluded_paths(client, monkeypatch):
    """Test that public paths are accessible without authentication."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret')

    # We test /static even if file might not exist, 404 means it passed auth
    response = client.get('/static/test.css')
    assert response.status_code != 401

    response = client.options('/')
    assert response.status_code != 401

def test_socketio_path_excluded(client, monkeypatch):
    """Test that /socket.io/ is accessible without authentication."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret')

    response = client.get('/socket.io/?EIO=4&transport=polling')
    # The SocketIO endpoint should not return 401
    assert response.status_code != 401
