import os

files_to_patch = [
    'tests/e2e/test_training_flow.py'
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
