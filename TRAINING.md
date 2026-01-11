# Training Guide

## 1. Prerequisites
Ensure you have the data file at `data/sp500.csv`.
If not, run the downloader:
```bash
python scripts/download.py
```

## 2. Launching Training
To start the training process with default settings:

```bash
# Windows (PowerShell)
$env:PYTHONPATH="."; python scripts/train.py
```

### Optional Arguments
*   `--data`: Path to custom dataset (default: `data/sp500.csv`).
*   `--timesteps`: Total steps (default: `50,000,000`).
*   `--device`: `cuda:0` or `cpu`.
*   `--config`: Configuration file (default: `config/training.yaml`).

**Example:**
```bash
$env:PYTHONPATH="."; python scripts/train.py --timesteps 1000000 --device cpu
```

## 3. Monitoring
*   **Console**: Live progress bars and metrics.
*   **Logs**: Saved in `logs/train.log`.
*   **TensorBoard**: Visualize learning curves.
    ```bash
    tensorboard --logdir logs/tensorboard
    ```

## 4. Models
Checkpoints are saved automatically in the `models/` directory.
