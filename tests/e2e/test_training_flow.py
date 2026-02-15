import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from training.train import run_walk_forward

# Setup paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config_e2e.yaml"
OUTPUT_DIR = BASE_DIR / "models_e2e"


@pytest.fixture(scope="module")
def setup_config():
    # 1. Create Config
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    config = {
        "data": {"tickers": ["TEST"], "period": "2y", "interval": "1h"},
        "training": {
            "total_timesteps": 256,
            "learning_rate": 0.0003,
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "n_envs": 1,
            "xml_output": False,
        },
        "environment": {
            "initial_balance": 10000,
            "commission": 0.001,
            "reward_scaling": 1.0,
            "features_precomputed": True,
        },
        "walk_forward": {"train_years": 1, "test_months": 3, "step_months": 3},
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    yield

    if CONFIG_PATH.exists():
        os.remove(CONFIG_PATH)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


import os


@patch("training.train.download_data")
@patch("training.train.MacroDataFetcher")
def test_full_pipeline_execution(mock_macro_cls, mock_download, setup_config):
    """Lance un training complet avec Mock Data."""

    # 1. Mock Market Data
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="h")
    fake_df = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, len(dates)),
            "High": np.random.uniform(200, 210, len(dates)),
            "Low": np.random.uniform(90, 100, len(dates)),
            "Close": np.random.uniform(100, 200, len(dates)),
            "Volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )
    fake_df.index.name = "Date"
    mock_download.return_value = {"TEST": fake_df}

    # 2. Mock Macro Data
    mock_macro_instance = mock_macro_cls.return_value
    mock_macro_instance.fetch_all.return_value = pd.DataFrame()  # Empty macro

    print("\n--- STARTING E2E TRAINING (MOCKED) ---")

    # Run Training
    results = run_walk_forward(
        config_path=str(CONFIG_PATH),
        use_recurrent=False,
        n_ensemble=1,
        auto_scale=False,
        use_shared_memory=False,
    )

    # Assertions
    assert results is not None, "Training returned None"
    assert results["n_folds"] > 0

    # Check Index Preservation in Train Loop (indirectly via success)
    out_path = Path(results["output_dir"])
    assert (out_path / "walk_forward_results.json").exists()

    print("\n--- E2E TRAINING SUCCESS ---")
