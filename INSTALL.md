# Installation Guide

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.11 or higher
- **RAM**: 16GB minimum (32GB recommended for full S&P 500)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)

---

## Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/project_ploutos.git
cd project_ploutos
```

---

## Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies
| Package | Purpose |
|---------|---------|
| `torch>=2.0.0` | Deep learning backbone |
| `stable-baselines3>=2.0.0` | RL algorithms |
| `gymnasium>=0.29.0` | Environment interface |
| `pandas>=2.1.4` | Data processing |
| `yfinance>=0.2.33` | Stock data download |
| `scikit-learn>=1.3.0` | Correlation clustering |

---

## Step 4: GPU Setup (CUDA)

### Check CUDA Availability
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

### Install CUDA-enabled PyTorch
If the above returns False, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5: Download Data

```bash
python scripts/download.py --tickers SP500 --period 5y --output data/sp500.csv
```

This downloads 5 years of historical data for all S&P 500 constituents.

---

## Step 6: Verify Installation

```bash
python -c "from ploutos.env.environment import TradingEnvironment; print('OK')"
```

If you see `OK`, installation is complete!

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ploutos'`
Set PYTHONPATH:
```bash
# Windows PowerShell
$env:PYTHONPATH="src"

# Linux/macOS
export PYTHONPATH=src
```

### `UnicodeEncodeError` on Windows
Set console encoding:
```bash
chcp 65001
```
Or use `python -X utf8 script.py`

### `CUDA out of memory`
Reduce `n_envs` in `config/training.yaml`:
```yaml
n_envs: 16  # or lower
```

---

## Next Steps

See [TRAINING.md](TRAINING.md) for training instructions.
