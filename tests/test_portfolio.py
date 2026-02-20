from unittest.mock import patch

import pytest

# Use patch.dict to mock sys.modules locally if needed, or rely on import logic
# If Portfolio imports something that imports torch, we need to mock it if torch is missing
# But since we installed torch in the previous step (for E2E tests), we might not need to mock it anymore?
# The logs showed "Installing ... torch-2.10.0".
# However, if we want to run tests in isolation or speed up, mocking is fine, but NOT globally.

@pytest.fixture
def portfolio():
    # We can mock torch here if we want to test Portfolio in isolation without torch
    # But Portfolio primarily uses core.utils, settings, etc.
    # If we need to mock imports, we should do it before importing Portfolio
    # BUT we can't un-import easily.
    # Given torch is installed, let's remove the global mock.

    # Check if we need to mock torch. Portfolio uses 'core.utils' which imports 'torch'.
    # If torch is installed, no issue. If not, core.utils handles it.

    # We'll just import Portfolio normally.
    from trading.portfolio import Portfolio
    return Portfolio(initial_capital=100000)


def test_initialization(portfolio):
    assert portfolio.initial_capital == 100000
    assert portfolio.cash == 100000
    assert portfolio.positions == {}
    assert portfolio.trades_history == []


def test_buy_success(portfolio):
    success = portfolio.buy("AAPL", 150.0, 1500.0)
    assert success is True
    assert portfolio.cash == 100000 - 1500.0
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"]["shares"] == 10
    assert portfolio.positions["AAPL"]["entry_price"] == 150.0
    assert len(portfolio.trades_history) == 1
    assert portfolio.trades_history[0]["action"] == "BUY"


def test_buy_insufficient_funds(portfolio):
    success = portfolio.buy("AAPL", 150.0, 200000.0)
    assert success is False
    assert portfolio.cash == 100000
    assert "AAPL" not in portfolio.positions


def test_buy_existing_position(portfolio):
    portfolio.buy("AAPL", 100.0, 1000.0)  # 10 shares @ 100
    portfolio.buy("AAPL", 200.0, 1000.0)  # 5 shares @ 200

    assert portfolio.positions["AAPL"]["shares"] == 15
    # (10 * 100 + 5 * 200) / 15 = (1000 + 1000) / 15 = 2000 / 15 = 133.333...
    assert pytest.approx(portfolio.positions["AAPL"]["entry_price"]) == 133.33333333333334
    assert portfolio.cash == 100000 - 2000.0


def test_sell_success_full(portfolio):
    portfolio.buy("AAPL", 100.0, 1000.0)
    success = portfolio.sell("AAPL", 120.0, 1.0)

    assert success is True
    assert "AAPL" not in portfolio.positions
    assert portfolio.cash == 99000 + 1200.0
    assert len(portfolio.trades_history) == 2
    assert portfolio.trades_history[1]["action"] == "SELL"
    assert portfolio.trades_history[1]["pnl"] == 200.0
    assert pytest.approx(portfolio.trades_history[1]["pnl_pct"]) == 20.0


def test_sell_success_partial(portfolio):
    portfolio.buy("AAPL", 100.0, 1000.0)  # 10 shares
    success = portfolio.sell("AAPL", 120.0, 0.5)  # 5 shares

    assert success is True
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"]["shares"] == 5
    assert portfolio.cash == 99000 + 600.0
    assert portfolio.trades_history[1]["pnl"] == 100.0


def test_sell_non_existent(portfolio):
    success = portfolio.sell("MSFT", 100.0)
    assert success is False


def test_get_total_value(portfolio):
    portfolio.buy("AAPL", 100.0, 1000.0)  # 10 shares
    current_prices = {"AAPL": 150.0}

    total_value = portfolio.get_total_value(current_prices)
    # 99000 (cash) + 10 * 150 (AAPL) = 99000 + 1500 = 100500
    assert total_value == 100500.0


def test_get_summary(portfolio):
    portfolio.buy("AAPL", 100.0, 1000.0)
    summary = portfolio.get_summary({"AAPL": 150.0})

    assert summary["cash"] == 99000.0
    assert summary["positions_count"] == 1
    assert summary["total_value"] == 100500.0
    assert summary["total_return"] == 500.0
    assert summary["total_return_pct"] == 0.5


@patch("trading.portfolio.TRADES_DIR")
def test_save_state(mock_trades_dir, portfolio, tmp_path):
    mock_trades_dir.__truediv__.return_value = tmp_path / "test_portfolio.json"

    portfolio.buy("AAPL", 100.0, 1000.0)
    portfolio.save_state("test_portfolio.json")

    saved_file = tmp_path / "test_portfolio.json"
    assert saved_file.exists()

    import json

    with open(saved_file) as f:
        data = json.load(f)

    assert data["initial_capital"] == 100000
    assert data["cash"] == 99000
    assert "AAPL" in data["positions"]


@patch("trading.portfolio.TRADES_DIR")
def test_load_state(mock_trades_dir, portfolio, tmp_path):
    test_file = tmp_path / "test_portfolio.json"
    mock_trades_dir.__truediv__.return_value = test_file

    import json

    state = {
        "initial_capital": 50000,
        "cash": 45000,
        "positions": {"MSFT": {"shares": 50, "entry_price": 100.0}},
        "trades_history": [],
    }
    with open(test_file, "w") as f:
        json.dump(state, f)

    success = portfolio.load_state("test_portfolio.json")
    assert success is True
    assert portfolio.initial_capital == 50000
    assert portfolio.cash == 45000
    assert "MSFT" in portfolio.positions
