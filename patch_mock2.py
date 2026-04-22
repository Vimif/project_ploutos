import os

files_to_patch = [
    'tests/test_data_pipeline.py',
    'tests/test_trading_env_v8.py',
    'tests/test_ensemble.py',
    'tests/test_paper_trade.py',
]

for filepath in files_to_patch:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()

        # Remove torch mocking
        content = content.replace('sys.modules.setdefault("torch", MagicMock())\n', '')
        content = content.replace("sys.modules.setdefault('torch', MagicMock())\n", '')

        with open(filepath, 'w') as f:
            f.write(content)
