import sys
import os
import pytest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might cause side effects or require dependencies
sys.modules['trading'] = MagicMock()
sys.modules['trading.alpaca_client'] = MagicMock()
sys.modules['trading.broker_interface'] = MagicMock()
sys.modules['core'] = MagicMock()
sys.modules['core.utils'] = MagicMock()

# Setup mocks for imports in dashboard/app.py
mock_alpaca_client = MagicMock()
sys.modules['trading.alpaca_client'].AlpacaClient = mock_alpaca_client
sys.modules['core.utils'].setup_logging = MagicMock(return_value=MagicMock())

# Import app after mocking
from dashboard.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # We need to add the route if it doesn't exist to test the template
    # But first we check if it exists
    with app.test_client() as client:
        yield client

def test_trades_route_exists(client):
    """Test if /trades route exists"""
    rv = client.get('/trades')
    # This assertion will likely fail if the route is missing
    # But if it fails, I will add the route to app.py as part of my fix?
    # Or I inject it for the test just to verify the template?
    assert rv.status_code == 200

def test_trades_template_content(client):
    """Test if trades template has accessibility improvements"""
    rv = client.get('/trades')
    assert rv.status_code == 200

    html = rv.data.decode('utf-8')

    # Check accessibility improvements
    assert 'id="filter-btn"' in html
    assert 'for="filter-days"' in html
    assert 'onkeydown' in html
