"""Unit tests for the eToro broker client."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)


class _FakeSession:
    def __init__(self):
        self.headers = {}

    def request(self, *args, **kwargs):
        raise NotImplementedError


requests_module = types.ModuleType("requests")
requests_module.Session = _FakeSession
requests_module.exceptions = types.SimpleNamespace(RequestException=Exception)
requests_module.Response = object
sys.modules.setdefault("requests", requests_module)

from trading import etoro_client as etoro_module
from trading.etoro_client import EToroClient


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ETORO_PUBLIC_API_KEY", "public-key")
    monkeypatch.setenv("ETORO_USER_KEY", "user-key")
    monkeypatch.delenv("ETORO_SUBSCRIPTION_KEY", raising=False)
    monkeypatch.delenv("ETORO_API_KEY", raising=False)

    def fake_authenticate(self):
        self._identity = {"username": "demo-user"}

    monkeypatch.setattr(EToroClient, "_authenticate", fake_authenticate)
    return EToroClient(paper_trading=True)


def test_init_requires_user_key(monkeypatch):
    monkeypatch.setenv("ETORO_PUBLIC_API_KEY", "public-key")
    monkeypatch.delenv("ETORO_USER_KEY", raising=False)
    monkeypatch.delenv("ETORO_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ETORO_USER_KEY"):
        EToroClient(paper_trading=True)


def test_request_reauths_after_401(client, monkeypatch):
    responses = [DummyResponse(status_code=401), DummyResponse(status_code=200)]
    client.session.request = MagicMock(side_effect=responses)
    reauths = {"count": 0}

    def reauthenticate():
        reauths["count"] += 1
        client._identity = {"username": "demo-user"}

    monkeypatch.setattr(client, "_authenticate", reauthenticate)

    response = client._request("GET", "/me")

    assert response.status_code == 200
    assert client.session.request.call_count == 2
    assert reauths["count"] == 1


def test_request_retries_after_rate_limit(client, monkeypatch):
    responses = [DummyResponse(status_code=429), DummyResponse(status_code=200)]
    client.session.request = MagicMock(side_effect=responses)
    sleeps = []
    monkeypatch.setattr(etoro_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    response = client._request("GET", "/me")

    assert response.status_code == 200
    assert sleeps == [1]


def test_get_positions_normalizes_current_portfolio_snapshot(client, monkeypatch):
    payload = {
        "clientPortfolio": {
            "positions": [
                {
                    "instrumentId": 1,
                    "initialAmountInDollars": 500.0,
                    "openRate": 100.0,
                    "closeRate": 110.0,
                    "pnL": 50.0,
                    "units": 5.0,
                    "positionId": "pos-1",
                    "openDateTime": "2026-04-01T10:00:00Z",
                }
            ]
        }
    }
    monkeypatch.setattr(
        client, "_request", lambda *args, **kwargs: DummyResponse(200, payload, text="ok")
    )
    monkeypatch.setattr(client, "_get_symbol", lambda instrument_id: "AAPL")

    positions = client.get_positions()

    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["qty"] == 5.0
    assert positions[0]["market_value"] == 550.0
    assert positions[0]["unrealized_plpc"] == 0.1


def test_place_market_order_uses_demo_by_amount_endpoint(client, monkeypatch):
    captured = {}

    def fake_request(method, path, json_data=None, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = json_data
        return DummyResponse(201, {"orderId": "order-1"}, text="ok")

    monkeypatch.setattr(client, "_get_instrument_id", lambda symbol: 123)
    monkeypatch.setattr(client, "get_current_price", lambda symbol: 200.0)
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "get_account", lambda: {"portfolio_value": 1_000.0})
    monkeypatch.setattr(etoro_module, "log_trade_to_json", lambda **kwargs: None)

    order = client.place_market_order("AAPL", 1.5, side="buy", reason="unit-test")

    assert captured["method"] == "POST"
    assert captured["path"] == "/trading/execution/demo/market-open-orders/by-amount"
    assert captured["payload"]["InstrumentId"] == 123
    assert captured["payload"]["Amount"] == 300.0
    assert order["qty"] == 1.5
    assert order["filled_avg_price"] == 200.0


def test_wait_for_order_fill_polls_until_position_present(client, monkeypatch):
    orders = [
        [{"id": "order-1", "status": "open"}],
        [{"id": "order-1", "status": "open"}],
        [],
    ]
    positions = [None, None, {"symbol": "AAPL"}]
    monkeypatch.setattr(client, "get_orders", lambda status="all": orders.pop(0) if orders else [])
    monkeypatch.setattr(
        client,
        "get_position",
        lambda symbol: positions.pop(0) if positions else {"symbol": symbol},
    )
    monkeypatch.setattr(etoro_module.time, "sleep", lambda seconds: None)

    assert client.wait_for_order_fill("order-1", symbol="AAPL", timeout=3, poll_interval=0.01)
