import pytest
from unittest.mock import MagicMock, patch
import json
from dashboard.app import app, init_alpaca

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Mock socketio to avoid async/gevent issues in tests
    with patch('dashboard.app.socketio'), \
         patch('dashboard.app.alpaca_client'), \
         patch('dashboard.app.init_alpaca', return_value=True):
        with app.test_client() as client:
            yield client

def test_routes_exist(client):
    """Test that main routes return 200 OK."""
    # Index
    rv = client.get('/')
    assert rv.status_code == 200

    # New routes
    rv = client.get('/trades')
    assert rv.status_code == 200

    rv = client.get('/metrics')
    assert rv.status_code == 200

def test_api_routes(client):
    """Test API routes."""
    # Mock load_trades_from_json
    with patch('dashboard.app.load_trades_from_json') as mock_load:
        mock_load.return_value = [
            {
                'timestamp': '2023-01-01T12:00:00',
                'action': 'BUY',
                'symbol': 'AAPL',
                'price': 150.0,
                'shares': 10,
                'amount': 1500.0,
                'reason': 'Test'
            }
        ]

        # /api/db/trades
        rv = client.get('/api/db/trades')
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data['success'] is True
        assert len(data['data']) == 1

        # /api/db/statistics
        rv = client.get('/api/db/statistics')
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data['success'] is True
        assert data['data']['statistics']['total_volume'] == 1500.0

def test_api_account(client):
    """Test Alpaca account API."""
    with patch('dashboard.app.alpaca_client') as mock_alpaca:
        mock_alpaca.get_account.return_value = {
            'portfolio_value': 100000,
            'cash': 50000,
            'buying_power': 200000,
            'equity': 100000,
            'last_equity': 99000
        }

        rv = client.get('/api/account')
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data['success'] is True
        assert data['data']['portfolio_value'] == 100000.0
