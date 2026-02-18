
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

# Import the app
from dashboard.app import app, init_alpaca

@pytest.fixture
def client(monkeypatch):
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test_secret'

    # Mock credentials using monkeypatch
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'secret123')

    with app.test_client() as client:
        yield client

def test_unauthorized_access(client):
    """Test accessing protected route without credentials"""
    response = client.get('/')
    assert response.status_code == 401
    assert 'WWW-Authenticate' in response.headers
    assert response.headers['WWW-Authenticate'] == 'Basic realm="Login Required"'

def test_authorized_access(client):
    """Test accessing protected route with correct credentials"""
    # Create Basic Auth header
    headers = {
        'Authorization': 'Basic YWRtaW46c2VjcmV0MTIz' # admin:secret123 in base64
    }

    # Mock render_template to avoid template issues if any
    with patch('dashboard.app.render_template', return_value='<html></html>'):
        response = client.get('/', headers=headers)
        assert response.status_code == 200

def test_wrong_credentials(client):
    """Test accessing protected route with wrong credentials"""
    headers = {
        'Authorization': 'Basic YWRtaW46d3Jvbmc=' # admin:wrong
    }
    response = client.get('/', headers=headers)
    assert response.status_code == 401

def test_api_protection(client):
    """Test API route protection"""
    response = client.post('/api/close_position/AAPL')
    assert response.status_code == 401
