import os

files = [
    "tests/test_ensemble.py",
    "tests/test_data_pipeline.py",
    "tests/test_trading_env_v8.py",
    "tests/test_portfolio.py",
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # We should avoid mocking torch in sys.modules because it causes reload issues for e2e test.
    # Actually, the e2e test imports torch, then these tests run, or vice versa.
    # Since torch is already installed and fast enough to import, mocking it is causing the issue.
    # Let's remove the sys.modules mocking.
