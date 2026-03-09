import base64

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    # Make sure TESTING mode is strictly False, to test the auth layer
    app.config['TESTING'] = False
    try:
        with app.test_client() as client:
            yield client
    finally:
        app.config['TESTING'] = True  # reset if other tests assume it's true

@pytest.fixture
def auth_headers():
    credentials = "admin:password"
    encoded_credentials = base64.b64encode(credentials.encode()).decode("utf-8")
    return {"Authorization": f"Basic {encoded_credentials}"}

def test_missing_credentials(client, monkeypatch):
    """If no environment variables are set, access should be denied securely."""
    monkeypatch.delenv('DASHBOARD_USERNAME', raising=False)
    monkeypatch.delenv('DASHBOARD_PASSWORD', raising=False)

    response = client.get('/api/account')
    assert response.status_code == 401
    assert b'Authentication configuration missing. Access denied.' in response.data

def test_no_auth_provided(client, monkeypatch):
    """If credentials exist but not provided in request, access denied."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'password')

    response = client.get('/api/account')
    assert response.status_code == 401
    assert b'Could not verify your access level' in response.data

def test_invalid_auth_provided(client, monkeypatch):
    """If wrong credentials provided, access denied."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'password')

    credentials = "admin:wrongpassword"
    encoded_credentials = base64.b64encode(credentials.encode()).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    response = client.get('/api/account', headers=headers)
    assert response.status_code == 401
    assert b'Could not verify your access level' in response.data

def test_valid_auth_provided(client, monkeypatch, auth_headers):
    """If correct credentials provided, access should proceed."""
    monkeypatch.setenv('DASHBOARD_USERNAME', 'admin')
    monkeypatch.setenv('DASHBOARD_PASSWORD', 'password')

    # Let's mock the alpaca_client since we don't have real credentials
    from unittest.mock import MagicMock

    import dashboard.app

    dashboard.app.alpaca_client = MagicMock()
    dashboard.app.alpaca_client.get_account.return_value = {
        'portfolio_value': '1000',
        'cash': '500',
        'buying_power': '500',
        'equity': '1000'
    }

    response = client.get('/api/account', headers=auth_headers)
    assert response.status_code == 200

def test_bypass_static_routes(client, monkeypatch):
    """Static routes should bypass authentication."""
    monkeypatch.delenv('DASHBOARD_USERNAME', raising=False)
    monkeypatch.delenv('DASHBOARD_PASSWORD', raising=False)

    # We may get 404 because the static file might not exist, but we shouldn't get 401.
    response = client.get('/static/css/style.css')
    assert response.status_code != 401

def test_options_cors_bypass(client, monkeypatch):
    """OPTIONS requests for CORS preflight should bypass auth."""
    monkeypatch.delenv('DASHBOARD_USERNAME', raising=False)
    monkeypatch.delenv('DASHBOARD_PASSWORD', raising=False)

    response = client.options('/api/account')
    assert response.status_code != 401
