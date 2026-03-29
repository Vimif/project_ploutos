sed -i 's/from dotenv import load_dotenv/# ruff: noqa: E402\nfrom dotenv import load_dotenv/' core/data_fetcher.py
sed -i 's/from core.utils import setup_logging/# ruff: noqa: E402\nfrom core.utils import setup_logging/' core/macro_data.py
sed -i 's/from .db import \*/from .db import init_database, log_trade, log_position, log_prediction, get_trade_history, get_position_history, get_daily_summary, save_daily_summary, get_trade_statistics, get_portfolio_evolution/' database/__init__.py
sed -i 's/for ticker, df in data.items():/for _ticker, df in data.items():/g' legacy/training/train_v7_sp500_sectors.py
sed -i 's/for ticker, df in data.items():/for _ticker, df in data.items():/g' scripts/backtest_ultimate.py
sed -i 's/({expected_obs} dims)//' scripts/backtest_ultimate.py
