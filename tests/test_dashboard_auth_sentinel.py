import pytest
import os
import sys
import base64
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True

    # Patch init_alpaca to avoid real connection attempts
    with patch('dashboard.app.init_alpaca') as mock_init:
        mock_init.return_value = True
        with app.test_client() as client:
            yield client

def test_dashboard_access_without_auth(client):
    """Test that accessing the dashboard without authentication returns 401."""
    response = client.get('/')
    assert response.status_code == 401
    assert b'Authentication required' in response.data

def test_dashboard_access_with_auth(client):
    """Test that accessing with correct credentials works."""
    with patch.dict(os.environ, {'DASHBOARD_USERNAME': 'admin', 'DASHBOARD_PASSWORD': 'secret'}):
        # Construct Basic Auth header
        auth_str = 'admin:secret'
        auth_bytes = auth_str.encode('ascii')
        base64_bytes = base64.b64encode(auth_bytes)
        base64_str = base64_bytes.decode('ascii')
        headers = {'Authorization': f'Basic {base64_str}'}

        response = client.get('/', headers=headers)
        assert response.status_code == 200

def test_socketio_exclusion(client):
    """Test that Socket.IO paths are excluded from authentication."""
    # This path mimics what a socket.io client might request
    response = client.get('/socket.io/?EIO=4&transport=polling')
    # Should NOT be 401.
    # It might be 404 if the socketio server isn't fully spinning in test client
    # or 200 if handled by socketio.
    # Crucially, it must NOT be 401 from our before_request hook.
    assert response.status_code != 401

def test_static_exclusion(client):
    """Test that static paths are excluded from authentication."""
    # Assuming there's a static folder or route, even if empty/missing file
    # We just want to ensure our before_request hook returns None (pass through)
    # If file missing, it's 404, not 401.
    response = client.get('/static/js/main.js')
    assert response.status_code != 401

def test_webhook_exclusion(client):
    """Test that webhook paths are excluded from authentication."""
    response = client.post('/api/webhook/alpaca', json={'test': 'data'})
    # Should not be 401
    assert response.status_code != 401
