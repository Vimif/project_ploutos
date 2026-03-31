#!/usr/bin/env python3
# scripts/run_pipeline.py
"""Pipeline complet : Walk-Forward Training -> Tests de Robustesse.

Orchestre l'entraînement et la validation en une seule commande.
Avec --auto-scale, détecte le hardware et optimise les paramètres.

Usage:
    python scripts/run_pipeline.py --config config/config.yaml --auto-scale
    python scripts/run_pipeline.py --config config/config.yaml --auto-scale --ensemble 3
    python scripts/run_pipeline.py --config config/config.yaml --auto-scale --ensemble 3 --mc-sims 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import time

from core.utils import setup_logging
from config.hardware import detect_hardware

logger = setup_logging(__name__, 'pipeline.log')


def main():
    parser = argparse.ArgumentParser(description='Ploutos Full Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--recurrent', action='store_true', help='Use RecurrentPPO (LSTM)')
    parser.add_argument('--ensemble', type=int, default=1, help='Ensemble size')
    parser.add_argument('--auto-scale', action='store_true', help='Auto-detect hardware')
    parser.add_argument('--shared-memory', action='store_true', help='Use Shared Memory for training (V9)')
    parser.add_argument(
        '--mc-sims', type=int, default=1000,
        help='Monte Carlo simulations (default: 1000)',
    )
    parser.add_argument('--skip-robustness', action='store_true', help='Skip robustness tests')

    args = parser.parse_args()

    t0 = time.time()

    # Log hardware
    hw = detect_hardware()
    logger.info("=" * 70)
    logger.info("PLOUTOS TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(
        f"Hardware: {hw['gpu_name'] or 'CPU'} | "
        f"{hw['gpu_vram_gb']} GB VRAM | "
        f"{hw['cpu_count']} CPUs | "
        f"{hw['ram_gb']:.0f} GB RAM"
    )
    logger.info(f"Config: {args.config}")
    logger.info(f"Ensemble: {args.ensemble} | Recurrent: {args.recurrent}")
    logger.info(f"Auto-scale: {args.auto_scale} | Shared Memory: {args.shared_memory}")

    # === Phase 1 : Walk-Forward Training ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: WALK-FORWARD TRAINING")
    logger.info("=" * 70)

    from training.train import run_walk_forward

    results = run_walk_forward(
        config_path=args.config,
        use_recurrent=args.recurrent,
        n_ensemble=args.ensemble,
        auto_scale=args.auto_scale,
        use_shared_memory=args.shared_memory,
    )

    if not results:
        logger.error("Training failed, aborting pipeline")
        return

    training_time = time.time() - t0
    logger.info(f"Training completed in {training_time / 60:.1f} min")

    # === Phase 2 : Robustness Tests ===
    if not args.skip_robustness:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: ROBUSTNESS TESTS")
        logger.info("=" * 70)

        output_dir = results.get('output_dir')
        if not output_dir:
            logger.error("No output_dir in results, skipping robustness")
            return

        # Trouver les modèles dans les folds
        model_files = sorted(glob.glob(f"{output_dir}/fold_*/model.zip"))
        if not model_files:
            logger.warning("No model.zip found in folds, skipping robustness")
            return

        from scripts.robustness_tests import (
            load_model, monte_carlo_test, stress_test_crash,
            _load_vecnormalize,
        )
        from config.hardware import compute_optimal_params
        from core.data_fetcher import download_data
        from core.features import FeatureEngineer
        from core.macro_data import MacroDataFetcher
        import pandas as pd
        import yaml
        import json
        import os

        # Load config and data
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        env_kwargs = {k: v for k, v in config.get('environment', {}).items()}

        data = download_data(
            tickers=config['data']['tickers'],
            period=config['data'].get('period', '5y'),
            interval=config['data'].get('interval', '1h'),
            dataset_path=config['data'].get('dataset_path'),
        )

        # Auto-scale workers
        n_workers = 1
        if args.auto_scale:
            params = compute_optimal_params(hw)
            n_workers = params["mc_workers"]

        # Test each fold with its own test period and VecNormalize
        for model_path in model_files:
            fold_dir = str(Path(model_path).parent)
            fold_name = Path(model_path).parent.name
            logger.info(f"\n--- Robustness: {fold_name} ---")

            # Load fold metadata for test period alignment
            metadata_path = os.path.join(fold_dir, 'fold_metadata.json')
            test_data = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                ts_start = pd.Timestamp(metadata['test_start'])
                ts_end = pd.Timestamp(metadata['test_end'])
                logger.info(f"  Using fold test period: {ts_start.date()} -> {ts_end.date()}")
                test_data = {}
                for ticker, df in data.items():
                    mask = (df.index >= ts_start) & (df.index < ts_end)
                    sliced = df.loc[mask]
                    if len(sliced) > 50:
                        test_data[ticker] = sliced

            if not test_data:
                logger.warning("  No fold metadata or empty slice, using DataSplitter")
                from core.data_pipeline import DataSplitter
                splits = DataSplitter.split(data)
                test_data = splits.test

            # Macro data for test period
            macro_data = None
            try:
                macro_fetcher = MacroDataFetcher()
                ref_df = test_data[list(test_data.keys())[0]]
                macro_data = macro_fetcher.fetch_all(
                    start_date=str(ref_df.index[0].date()),
                    end_date=str(ref_df.index[-1].date()),
                    interval=config['data'].get('interval', '1h'),
                )
                if macro_data.empty:
                    macro_data = None
            except Exception as e:
                logger.warning(f"  Failed to fetch macro data: {e}")

            # Pre-compute features for stress test
            fe = FeatureEngineer()
            precomputed_test = {}
            for ticker, df in test_data.items():
                original_idx = df.index
                feat_df = fe.calculate_all_features(df.copy())
                if len(feat_df) == len(original_idx):
                    feat_df.index = original_idx
                precomputed_test[ticker] = feat_df

            # Load VecNormalize per fold
            vecnorm_path = os.path.join(fold_dir, 'vecnormalize.pkl')
            vecnorm_env = None
            if os.path.exists(vecnorm_path):
                precomputed_env_kwargs = {**env_kwargs, 'features_precomputed': True}
                vecnorm_env = _load_vecnormalize(
                    vecnorm_path, precomputed_test, macro_data, precomputed_env_kwargs
                )
                logger.info(f"  Loaded VecNormalize: {vecnorm_path}")
            else:
                vecnorm_path = None
                logger.warning("  No VecNormalize found for this fold")

            model = load_model(model_path, use_recurrent=args.recurrent)

            mc_report = monte_carlo_test(
                model, test_data, macro_data,
                n_simulations=args.mc_sims,
                env_kwargs=env_kwargs,
                n_workers=n_workers,
                model_path=model_path,
                use_recurrent=args.recurrent,
                vecnorm_env=vecnorm_env,
                vecnorm_path=vecnorm_path,
            )

            st_report = stress_test_crash(
                model, test_data, macro_data,
                env_kwargs=env_kwargs,
                vecnorm_env=vecnorm_env,
            )

            # Save per fold
            report = {'monte_carlo': mc_report, 'stress_test': st_report}
            report_path = os.path.join(fold_dir, 'robustness_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"  Report: {report_path}")

    # === Résumé ===
    total_time = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time / 60:.1f} min")
    logger.info(f"Output: {results.get('output_dir', 'N/A')}")


if __name__ == '__main__':
    main()
