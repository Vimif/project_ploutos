import os
import pytest
from base64 import b64encode
from unittest.mock import patch

from dashboard.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers():
    credentials = b"admin:secret"
    return {
        'Authorization': f'Basic {b64encode(credentials).decode("utf-8")}'
    }

def test_dashboard_auth_missing_credentials(client):
    """Test that requests without credentials return 401 Unauthorized."""
    response = client.get('/')
    assert response.status_code == 401
    assert b'WWW-Authenticate' in response.headers.get('WWW-Authenticate', b'').encode() or response.headers.get('WWW-Authenticate') == 'Basic realm="Ploutos Dashboard"'

@patch.dict(os.environ, {'DASHBOARD_USERNAME': 'admin', 'DASHBOARD_PASSWORD': 'secret'})
def test_dashboard_auth_valid_credentials(client, auth_headers):
    """Test that requests with valid credentials succeed."""
    response = client.get('/', headers=auth_headers)
    assert response.status_code == 200

@patch.dict(os.environ, {'DASHBOARD_USERNAME': 'admin', 'DASHBOARD_PASSWORD': 'secret'})
def test_dashboard_auth_invalid_credentials(client):
    """Test that requests with invalid credentials return 401 Unauthorized."""
    credentials = b"admin:wrongpassword"
    headers = {
        'Authorization': f'Basic {b64encode(credentials).decode("utf-8")}'
    }
    response = client.get('/', headers=headers)
    assert response.status_code == 401

def test_dashboard_auth_excluded_paths(client):
    """Test that static files and webhooks do not require authentication."""
    # Note: Flask's default static folder might return 404 if no files are there,
    # but it shouldn't return 401.
    response_static = client.get('/static/css/style.css')
    assert response_static.status_code != 401

    response_webhook = client.post('/api/webhook')
    # Since we are not providing valid webhook payload, it might be 404 or 405, but not 401.
    assert response_webhook.status_code != 401

def test_dashboard_auth_options_request(client):
    """Test that CORS preflight requests bypass authentication."""
    response = client.options('/')
    assert response.status_code != 401

def test_dashboard_auth_socket_io(client):
    """Test that socket.io connections are handled correctly (no strict 401)."""
    response = client.get('/socket.io/?EIO=4&transport=polling')
    # According to memory guidelines, do not assert strict 401 for socket.io polling endpoint.
    # It might return 200 or 400.
    assert response.status_code in [200, 400]
