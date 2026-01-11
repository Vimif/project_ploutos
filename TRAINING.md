# Training Guide

## 1. Prerequisites
Ensure you have the data file at `data/sp500.csv`.
If not, run the downloader:
```bash
python scripts/download.py
```

## 2. Launching Training
To start the training process:

### Windows (PowerShell)
**Crucial:** Set encoding before running to prevent errors.
```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="."
python scripts/train.py
```

### Linux / Mac
```bash
export PYTHONPATH="."
python scripts/train.py
```

## 3. Arguments
*   `--data`: Path to dataset (default: `data/sp500.csv`).
*   `--timesteps`: Total steps (default: `50,000,000`).
*   `--device`: `cuda:0` (requires GPU setup, see INSTALL.md) or `cpu`.
*   `--config`: Configuration file (default: `config/training.yaml`).

**Example (Training on CPU):**
```bash
python scripts/train.py --timesteps 1000000 --device cpu
```

## 4. Monitoring
*   **Console**: Live progress bars.
*   **Logs**: Saved in `logs/train.log`.
*   **TensorBoard**: Visualize learning curves.
    ```bash
    tensorboard --logdir logs/tensorboard
    ```
