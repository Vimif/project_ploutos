import base64
import os

import pytest

from dashboard.app import app, socketio


@pytest.fixture
def client():
    original_testing = app.config.get("TESTING")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    app.config["TESTING"] = original_testing


@pytest.fixture
def auth_client():
    original_testing = app.config.get("TESTING")
    app.config["TESTING"] = False
    with app.test_client() as client:
        yield client
    app.config["TESTING"] = original_testing


@pytest.fixture
def env_credentials():
    os.environ["DASHBOARD_USERNAME"] = "admin"
    os.environ["DASHBOARD_PASSWORD"] = "secretpassword"
    yield
    os.environ.pop("DASHBOARD_USERNAME", None)
    os.environ.pop("DASHBOARD_PASSWORD", None)


@pytest.fixture
def env_missing_credentials():
    os.environ.pop("DASHBOARD_USERNAME", None)
    os.environ.pop("DASHBOARD_PASSWORD", None)
    yield
    os.environ.pop("DASHBOARD_USERNAME", None)
    os.environ.pop("DASHBOARD_PASSWORD", None)


def get_basic_auth_headers(username, password):
    auth_string = f"{username}:{password}"
    base64_auth = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {base64_auth}"}


def test_dashboard_auth_missing_credentials_fails_securely(auth_client, env_missing_credentials):
    """S'assurer qu'il n'y a pas de fallback par défaut."""
    response = auth_client.get("/")
    assert response.status_code == 401

    headers = get_basic_auth_headers("admin", "admin")
    response = auth_client.get("/", headers=headers)
    assert response.status_code == 401


def test_dashboard_auth_valid_credentials(auth_client, env_credentials):
    """Accès autorisé avec les bons identifiants."""
    headers = get_basic_auth_headers("admin", "secretpassword")
    response = auth_client.get("/", headers=headers)
    assert response.status_code == 200


def test_dashboard_auth_invalid_credentials(auth_client, env_credentials):
    """Accès refusé avec de mauvais identifiants."""
    headers = get_basic_auth_headers("admin", "wrongpassword")
    response = auth_client.get("/", headers=headers)
    assert response.status_code == 401


def test_dashboard_auth_no_credentials(auth_client, env_credentials):
    """Accès refusé sans header Authorization."""
    response = auth_client.get("/")
    assert response.status_code == 401


def test_dashboard_auth_static_files_excluded(auth_client, env_missing_credentials):
    """Les fichiers statiques ne doivent pas nécessiter d'authentification."""
    response = auth_client.get("/static/css/style.css")
    assert (
        response.status_code != 401
    )  # Should ideally be 404 since it might not exist in tests, but definitely not 401


def test_dashboard_auth_webhook_excluded(auth_client, env_missing_credentials):
    """Les webhooks ne doivent pas nécessiter d'authentification."""
    response = auth_client.post("/api/webhook")
    assert response.status_code != 401


def test_dashboard_auth_options_excluded(auth_client, env_missing_credentials):
    """Les requêtes OPTIONS (CORS) ne doivent pas nécessiter d'authentification."""
    response = auth_client.options("/")
    assert response.status_code == 200


def test_dashboard_auth_testing_mode(client, env_missing_credentials):
    """En mode TESTING, l'authentification est ignorée."""
    response = client.get("/")
    assert response.status_code == 200


def test_dashboard_websocket_unauthorized(auth_client, env_credentials):
    """La connexion WebSocket doit être rejetée sans authentification."""
    from socketio.exceptions import ConnectionError

    app.config["TESTING"] = False
    try:
        # Flask-SocketIO test_client raises ConnectionRefusedError or ConnectionError if rejected
        socket_client = socketio.test_client(app, flask_test_client=auth_client)
        assert not socket_client.is_connected()
    except (ConnectionRefusedError, ConnectionError):
        pass  # Expected behavior
    finally:
        app.config["TESTING"] = True


def test_dashboard_websocket_authorized(auth_client, env_credentials):
    """La connexion WebSocket doit être acceptée avec authentification."""
    app.config["TESTING"] = False
    headers = get_basic_auth_headers("admin", "secretpassword")
    socket_client = socketio.test_client(app, flask_test_client=auth_client, headers=headers)
    assert socket_client.is_connected()
    app.config["TESTING"] = True
