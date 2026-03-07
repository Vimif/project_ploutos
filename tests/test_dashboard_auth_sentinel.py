import pytest

from dashboard.app import app


@pytest.fixture
def client():
    # Set testing mode so app catches exceptions, but disable app config TESTING
    # so our auth hook runs
    app.config["TESTING"] = False
    with app.test_client() as client:
        yield client
    # Restore config to not pollute other tests
    app.config["TESTING"] = True


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")


def test_no_auth_provided_returns_401(client, auth_env):
    """Test accessing protected route without auth returns 401."""
    response = client.get("/api/account")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == 'Basic realm="Login Required"'


def test_invalid_auth_returns_401(client, auth_env):
    """Test accessing protected route with wrong credentials returns 401."""
    # Encode 'admin:wrongpass' -> 'YWRtaW46d3JvbmdwYXNz'
    headers = {"Authorization": "Basic YWRtaW46d3JvbmdwYXNz"}
    response = client.get("/api/account", headers=headers)
    assert response.status_code == 401


def test_valid_auth_allowed(client, auth_env, monkeypatch):
    """Test accessing protected route with correct credentials."""
    # We mock init_alpaca or get_account so it doesn't try to make real calls
    monkeypatch.setattr("dashboard.app.init_alpaca", lambda: False)

    # Encode 'admin:secret' -> 'YWRtaW46c2VjcmV0'
    headers = {"Authorization": "Basic YWRtaW46c2VjcmV0"}
    response = client.get("/api/account", headers=headers)
    # Expected to return 500 because Alpaca init fails, but NOT 401
    assert response.status_code != 401


def test_excluded_paths_allowed_without_auth(client, auth_env):
    """Test paths that should be accessible without authentication."""
    # /static/
    response = client.get("/static/style.css")
    assert response.status_code != 401

    # /api/webhook
    response = client.post("/api/webhook")
    assert response.status_code != 401

    # /socket.io/ (Socket.IO endpoints often return 400 without proper payload,
    # but they shouldn't return our custom 401 from require_auth)
    response = client.get("/socket.io/?EIO=4&transport=polling")
    assert response.status_code != 401


def test_options_method_allowed_without_auth(client, auth_env):
    """Test CORS preflight requests bypass auth."""
    response = client.options("/api/account")
    assert response.status_code != 401


def test_missing_env_vars_denies_all(client, monkeypatch):
    """Test that missing env vars strictly denies access and does not fallback."""
    # Ensure env vars are missing
    monkeypatch.delenv("DASHBOARD_USERNAME", raising=False)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)

    # Encode 'admin:secret' -> 'YWRtaW46c2VjcmV0'
    headers = {"Authorization": "Basic YWRtaW46c2VjcmV0"}
    response = client.get("/api/account", headers=headers)
    assert response.status_code == 401
