# Training Guide

## 1. Prerequisites
Ensure you have the data file `SnP_daily_update.csv` in the project root.

## 2. Launching Training
Use the following command to start training on your custom data:

```bash
# Windows (PowerShell) - Set PYTHONPATH to avoid import errors
$env:PYTHONPATH="."; python scripts/train_v6_final.py --data SnP_daily_update.csv --output models/v6_snp --timesteps 50000000
```

## 3. Monitoring
Logs are saved to `logs/train_v6_final.log`.
You can also view TensorBoard logs:
```bash
tensorboard --logdir logs/tensorboard
```

## 4. Key Parameters
- `--timesteps`: Total training steps (default: 50M).
- `--device`: GPU device (default: `cuda:0`).
- `--output`: Directory for saving models.
