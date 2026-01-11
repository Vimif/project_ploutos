# Installation Guide

## 1. Prerequisites
*   **Python**: Version 3.10 or 3.11 recommended.
*   **OS**: Windows, Linux, or macOS.

## 2. Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing or you want to install manually:
```bash
pip install gymnasium stable-baselines3 pandas numpy torch pyyaml yfinance ta
```

## 3. Verify Setup
Run a quick check to ensure the environment is ready:

```bash
python -c "import gymnasium; import stable_baselines3; import torch; print(f'Ready! Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```
