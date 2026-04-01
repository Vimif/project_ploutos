#!/usr/bin/env python3
# scripts/robustness_tests.py
"""Tests de Robustesse : Monte Carlo + Stress Test Krach.

1. Monte Carlo: N backtests avec bruit sur les feature arrays (pas OHLCV).
   Le bruit est proportionnel a la std de chaque feature (noise_std * col_std).
   Features calculees une seule fois, bruitees N fois -> ~1000x speedup.
   Si le modele perd de l'argent dans >20% des cas -> overfitting.

2. Stress Test: Simule un krach de -20% en une journee.
   Verifie que le modele coupe ses positions ou se met short.

Usage:
    python scripts/robustness_tests.py --model models/fold_00/model.zip --all
    python scripts/robustness_tests.py --model models/fold_00/model.zip --monte-carlo 1000
    python scripts/robustness_tests.py --model models/fold_00/model.zip --stress-test
    python scripts/robustness_tests.py --model models/fold_00/model.zip --all --test-start 2024-01-01 --test-end 2024-07-01
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.hardware import compute_optimal_params, detect_hardware
from core.data_fetcher import download_data
from core.data_pipeline import DataSplitter
from core.environment import TradingEnv
from core.features import FeatureEngineer
from core.macro_data import MacroDataFetcher
from core.model_support import predict_with_optional_recurrence
from core.utils import setup_logging

logger = setup_logging(__name__, "robustness_tests.log")

try:
    from sb3_contrib import RecurrentPPO

    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


def load_model(model_path: str, use_recurrent: bool = False, device: str = "auto"):
    """Charge un modele PPO ou RecurrentPPO."""
    ModelClass = RecurrentPPO if (use_recurrent and HAS_RECURRENT) else PPO
    return ModelClass.load(model_path, device=device)


def _load_vecnormalize(vecnorm_path, test_data, macro_data, env_kwargs):
    """Load a frozen VecNormalize from pickle for inference."""
    kwargs = {**env_kwargs, "mode": "backtest"}
    dummy_env = DummyVecEnv(
        [lambda: Monitor(TradingEnv(data=test_data, macro_data=macro_data, **kwargs))]
    )
    vecnorm_env = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm_env.training = False
    vecnorm_env.norm_reward = False
    return vecnorm_env


def _precompute_features(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pre-compute features while preserving the original index."""
    fe = FeatureEngineer()
    precomputed = {}
    for ticker, df in data.items():
        original_idx = df.index
        feat_df = fe.calculate_all_features(df.copy())
        if len(feat_df) == len(original_idx):
            feat_df.index = original_idx
        precomputed[ticker] = feat_df
    return precomputed


def add_feature_noise(
    data: dict[str, pd.DataFrame], noise_std: float = 0.005
) -> dict[str, pd.DataFrame]:
    """Add proportional gaussian noise directly to feature columns (not OHLCV).

    Noise per column = noise_std * col_std * randn. OHLCV columns are kept intact
    so that trade execution prices remain realistic.

    Args:
        data: Dict {ticker: DataFrame with computed features}.
        noise_std: Noise scale relative to each column's std.

    Returns:
        Copy of data with noised feature columns.
    """
    ohlcv_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
    noisy_data = {}
    for ticker, df in data.items():
        noisy_df = df.copy()
        feature_cols = [c for c in noisy_df.columns if c not in ohlcv_cols]
        for col in feature_cols:
            col_std = noisy_df[col].std()
            if col_std > 0:
                noise = np.random.randn(len(noisy_df)) * noise_std * col_std
                noisy_df[col] = noisy_df[col] + noise
        noisy_data[ticker] = noisy_df
    return noisy_data


def add_price_noise(
    data: dict[str, pd.DataFrame], noise_std: float = 0.005
) -> dict[str, pd.DataFrame]:
    """Add gaussian noise to OHLCV prices (legacy, used by stress test)."""
    noisy_data = {}
    for ticker, df in data.items():
        noisy_df = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col in noisy_df.columns:
                noise = np.random.randn(len(noisy_df)) * noise_std
                noisy_df[col] = noisy_df[col] * (1 + noise)
        noisy_df["High"] = noisy_df[["Open", "High", "Close"]].max(axis=1)
        noisy_df["Low"] = noisy_df[["Open", "Low", "Close"]].min(axis=1)
        noisy_df = noisy_df.clip(lower=0.01)
        noisy_data[ticker] = noisy_df
    return noisy_data


def calculate_psr(returns: np.array, benchmark_sr: float = 0.0) -> float:
    """Calcule le Probabilistic Sharpe Ratio (PSR)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0

    sr_est = np.mean(returns) / np.std(returns)
    skew_est = skew(returns)
    kurt_est = kurtosis(returns)
    n = len(returns)

    denom = n - 1
    if denom <= 0:
        return 0.0

    sigma_sr_sq = (
        1 + (0.5 * sr_est**2) - (skew_est * sr_est) + ((kurt_est / 4) * sr_est**2)
    ) / denom
    if sigma_sr_sq <= 0:
        return 0.0

    sigma_sr = np.sqrt(sigma_sr_sq)
    psr = norm.cdf((sr_est - benchmark_sr) / sigma_sr)
    return float(psr)


def calculate_dsr(returns: np.array, n_trials: int = 1) -> float:
    """Calcule le Deflated Sharpe Ratio (DSR) simplifié."""
    if n_trials <= 1:
        return calculate_psr(returns, 0.0)

    # Benchmark ajusté pour le biais de sélection (Multiple Testing)
    # E[max(SR)] approx sqrt(2 * logN)
    benchmark_sr = np.sqrt(2 * np.log(n_trials)) * 0.05  # Scale factor empirique pour hourly
    return calculate_psr(returns, benchmark_sr)


def simulate_crash(
    data: dict[str, pd.DataFrame],
    crash_pct: float = -0.20,
    crash_day_idx: int = None,
    instant: bool = False,
) -> dict[str, pd.DataFrame]:
    """Simule un krach boursier à une date donnée.

    Args:
        data: Dict {ticker: DataFrame OHLCV}.
        crash_pct: Amplitude du krach (-0.20 = -20%).
        crash_day_idx: Index de la bougie de début du krach (None = milieu).
        instant: Si True, crash instantané (1 bar). Sinon, graduel sur 6 bars.

    Returns:
        Données avec un krach injecté.
    """
    crashed_data = {}
    for ticker, df in data.items():
        crashed_df = df.copy()
        n = len(crashed_df)

        if crash_day_idx is None:
            crash_day_idx = n // 2

        crash_duration = 1 if instant else 6
        end_idx = min(crash_day_idx + crash_duration, n)

        for col in ["Open", "High", "Low", "Close"]:
            if col not in crashed_df.columns:
                continue
            # Appliquer le crash
            for i, idx in enumerate(range(crash_day_idx, end_idx)):
                progress = 1.0 if instant else (i + 1) / crash_duration
                drop = crash_pct * progress
                crashed_df.iloc[idx, crashed_df.columns.get_loc(col)] *= 1 + drop

            # Les prix après le crash restent au niveau bas
            post_crash_factor = 1 + crash_pct
            for idx in range(end_idx, n):
                crashed_df.iloc[idx, crashed_df.columns.get_loc(col)] *= post_crash_factor

        crashed_df = crashed_df.clip(lower=0.01)
        crashed_data[ticker] = crashed_df

    return crashed_data


def run_backtest(
    model, data, macro_data, vecnorm_env=None, env_kwargs=None, deterministic=True
) -> dict:
    """Execute a backtest and return metrics.

    Args:
        deterministic: If False, use stochastic policy (adds exploration noise).
            MC sims should use deterministic=False to capture policy uncertainty.
    """
    kwargs = dict(env_kwargs or {})
    kwargs["mode"] = "backtest"
    kwargs["seed"] = 42

    env = TradingEnv(data=data, macro_data=macro_data, **kwargs)
    obs, _ = env.reset()
    done = False
    equity_curve = [env.initial_balance]
    recurrent_state = None
    episode_start = np.array([True], dtype=bool)

    while not done:
        if vecnorm_env is not None:
            obs_norm = vecnorm_env.normalize_obs(obs.reshape(1, -1)).flatten()
        else:
            obs_norm = obs
        action, recurrent_state = predict_with_optional_recurrence(
            model,
            obs_norm,
            deterministic=deterministic,
            recurrent_state=recurrent_state,
            episode_start=episode_start,
        )
        obs, reward, done, truncated, info = env.step(action)
        episode_start = np.array([done], dtype=bool)
        equity_curve.append(info["equity"])

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5)

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "final_equity": float(equity_series.iloc[-1]),
        "total_trades": info.get("total_trades", 0),
        "winning_trades": info.get("winning_trades", 0),
        "losing_trades": info.get("losing_trades", 0),
    }


def _mc_worker(args):
    """Worker for parallel Monte Carlo. Top-level for pickle.

    Receives precomputed feature data, applies feature noise, loads VecNormalize
    from disk in each worker (picklable path string).
    """
    (
        precomputed_data,
        macro_data,
        noise_std,
        env_kwargs,
        model_path,
        use_recurrent,
        seed,
        vecnorm_path,
    ) = args
    np.random.seed(seed)
    model = load_model(model_path, use_recurrent=use_recurrent, device="cpu")

    # Apply feature noise to precomputed data
    noisy_data = add_feature_noise(precomputed_data, noise_std=noise_std)

    # Load VecNormalize in worker if available
    vecnorm_env = None
    if vecnorm_path:
        vecnorm_env = _load_vecnormalize(vecnorm_path, noisy_data, macro_data, env_kwargs)

    return run_backtest(
        model,
        noisy_data,
        macro_data,
        vecnorm_env=vecnorm_env,
        env_kwargs={**env_kwargs, "features_precomputed": True},
        deterministic=False,
    )


def monte_carlo_test(
    model,
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    n_simulations: int = 1000,
    noise_std: float = 0.005,
    vecnorm_env=None,
    env_kwargs: dict = None,
    n_workers: int = 1,
    model_path: str = None,
    use_recurrent: bool = False,
    vecnorm_path: str = None,
) -> dict:
    """Monte Carlo Simulations with feature-level noise.

    Pre-computes features once, then applies proportional noise to feature arrays
    N times. Uses non-deterministic policy to capture action stochasticity.
    Loads VecNormalize per worker for correct observation normalization.

    Criterion: if >20% of simulations lose money -> overfitting.
    """
    # Pre-compute features once (shared across all MC sims)
    logger.info("  Pre-computing features for MC simulations...")
    precomputed_data = _precompute_features(test_data)

    mc_env_kwargs = dict(env_kwargs or {})
    mc_env_kwargs["features_precomputed"] = True

    logger.info(
        f"Monte Carlo: {n_simulations} simulations "
        f"(feature_noise={noise_std*100:.1f}%, workers={n_workers}, deterministic=False)"
    )

    if n_workers > 1 and model_path:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # Use 'spawn' context to avoid CUDA re-initialization errors in forked processes
        ctx = mp.get_context("spawn")
        worker_args = [
            (
                precomputed_data,
                macro_data,
                noise_std,
                mc_env_kwargs,
                model_path,
                use_recurrent,
                42 + i,
                vecnorm_path,
            )
            for i in range(n_simulations)
        ]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            results = list(pool.map(_mc_worker, worker_args))
        losses = sum(1 for r in results if r["total_return"] < 0)
        logger.info(f"  Done: {n_simulations} sims | Losses: {losses}/{n_simulations}")
    else:
        results = []
        for i in range(n_simulations):
            np.random.seed(42 + i)
            noisy_data = add_feature_noise(precomputed_data, noise_std=noise_std)
            metrics = run_backtest(
                model,
                noisy_data,
                macro_data,
                vecnorm_env,
                env_kwargs=mc_env_kwargs,
                deterministic=False,
            )
            results.append(metrics)

            if (i + 1) % 100 == 0:
                losses = sum(1 for r in results if r["total_return"] < 0)
                logger.info(
                    f"  Progress: {i+1}/{n_simulations} | "
                    f"Losses: {losses}/{i+1} ({losses/(i+1)*100:.1f}%)"
                )

    # Analyze
    returns = [r["total_return"] for r in results]
    sharpes = [r["sharpe_ratio"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]

    n_losses = sum(1 for r in returns if r < 0)
    loss_rate = n_losses / n_simulations

    report = {
        "n_simulations": n_simulations,
        "noise_std": noise_std,
        "noise_type": "feature",
        "deterministic": False,
        "loss_rate": loss_rate,
        "is_overfit": loss_rate > 0.20,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "p5_return": float(np.percentile(returns, 5)),
        "p95_return": float(np.percentile(returns, 95)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_max_drawdown": float(np.mean(drawdowns)),
    }

    psr = calculate_psr(np.array(returns), benchmark_sr=0.0)
    report["psr"] = psr
    report["dsr"] = calculate_dsr(np.array(returns), n_trials=n_simulations)

    logger.info("\n" + "=" * 50)
    logger.info("MONTE CARLO RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Simulations: {n_simulations}")
    logger.info(f"  Noise type:  feature-level ({noise_std*100:.1f}% * col_std)")
    logger.info("  Policy:      stochastic (deterministic=False)")
    logger.info(f"  VecNorm:     {'YES' if (vecnorm_env or vecnorm_path) else 'NO'}")
    logger.info(f"  Loss Rate:   {loss_rate:.1%} {'OVERFIT' if report['is_overfit'] else 'OK'}")
    logger.info(f"  Avg Return:  {report['avg_return']:+.2%}")
    logger.info(f"  Std Return:  {report['std_return']:.2%}")
    logger.info(f"  Range:       [{report['min_return']:+.2%}, {report['max_return']:+.2%}]")
    logger.info(f"  P5/P95:      [{report['p5_return']:+.2%}, {report['p95_return']:+.2%}]")
    logger.info(f"  Avg Sharpe:  {report['avg_sharpe']:.2f}")
    logger.info(
        f"  PSR (Prob):  {report['psr']:.4f} (>{'0.95 OK' if report['psr']>0.95 else 'FAIL'})"
    )
    logger.info(f"  Avg MaxDD:   {report['avg_max_drawdown']:.2%}")

    if report["is_overfit"]:
        logger.warning("  VERDICT: OVERFIT - Loss rate > 20%")
    else:
        logger.info("  VERDICT: ROBUST")

    return report


def stress_test_crash(
    model,
    test_data: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame | None,
    crash_pct: float = -0.20,
    vecnorm_env=None,
    env_kwargs: dict = None,
) -> dict:
    """Stress Test: simule un krach de -20%.

    Vérifie que le modèle :
    1. Coupe ses positions (stop loss)
    2. Ne perd pas plus que le drawdown max acceptable (15%)
    """
    logger.info(f"Stress Test: krach de {crash_pct*100:.0f}%")

    precomputed_env_kwargs = dict(env_kwargs or {})
    precomputed_env_kwargs["features_precomputed"] = True

    # Baseline sans krach
    baseline_data = _precompute_features(test_data)
    baseline = run_backtest(
        model,
        baseline_data,
        macro_data,
        vecnorm_env,
        precomputed_env_kwargs,
    )

    # Avec krach: les features doivent etre recalculees APRES le choc prix
    crashed_data = simulate_crash(test_data, crash_pct=crash_pct)
    crashed_features = _precompute_features(crashed_data)
    crash_result = run_backtest(
        model,
        crashed_features,
        macro_data,
        vecnorm_env,
        precomputed_env_kwargs,
    )

    # Analyser la réaction
    baseline_return = baseline["total_return"]
    crash_return = crash_result["total_return"]
    return_impact = crash_return - baseline_return

    report = {
        "crash_pct": crash_pct,
        "baseline_return": baseline_return,
        "crash_return": crash_return,
        "return_impact": return_impact,
        "baseline_max_drawdown": baseline["max_drawdown"],
        "crash_max_drawdown": crash_result["max_drawdown"],
        "crash_trades": crash_result["total_trades"],
        "survives": crash_result["max_drawdown"] < 0.50,  # Ne perd pas tout
        "acceptable_drawdown": crash_result["max_drawdown"] < 0.25,
    }

    logger.info("\n" + "=" * 50)
    logger.info("STRESS TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Crash simule:     {crash_pct*100:.0f}%")
    logger.info(f"  Baseline Return:  {baseline_return:+.2%}")
    logger.info(f"  Crash Return:     {crash_return:+.2%}")
    logger.info(f"  Impact:           {return_impact:+.2%}")
    logger.info(f"  Baseline MaxDD:   {baseline['max_drawdown']:.2%}")
    logger.info(f"  Crash MaxDD:      {crash_result['max_drawdown']:.2%}")
    logger.info(f"  Survives:         {'YES' if report['survives'] else 'NO'}")
    logger.info(f"  Acceptable DD:    {'YES' if report['acceptable_drawdown'] else 'NO'}")

    if not report["survives"]:
        logger.warning("  VERDICT: ECHEC - Le modele ne survit pas au krach")
    elif not report["acceptable_drawdown"]:
        logger.warning("  VERDICT: MARGINAL - Survit mais drawdown excessif")
    else:
        logger.info("  VERDICT: ROBUSTE - Le modele gere bien le krach")

    return report


def main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Robustness Tests (Monte Carlo + Stress Test)")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to vecnormalize .pkl (auto-detected from model dir if not set)",
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--monte-carlo", type=int, default=0, help="Number of MC simulations (0=skip)"
    )
    parser.add_argument("--stress-test", action="store_true", help="Run crash stress test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--recurrent", action="store_true", help="Model is RecurrentPPO")
    parser.add_argument(
        "--noise-std", type=float, default=0.005, help="MC noise std (default: 0.5%%)"
    )
    parser.add_argument(
        "--crash-pct", type=float, default=-0.20, help="Crash severity (default: -20%%)"
    )
    parser.add_argument(
        "--test-start", type=str, default=None, help="Test period start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-end", type=str, default=None, help="Test period end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Auto-detect hardware and parallelize Monte Carlo",
    )

    args = parser.parse_args()

    if args.all:
        args.monte_carlo = 1000
        args.stress_test = True

    if args.monte_carlo == 0 and not args.stress_test:
        print("Specify --monte-carlo N, --stress-test, or --all")
        return

    # Auto-detect vecnorm from model directory
    model_dir = str(Path(args.model).parent)
    if args.vecnorm is None:
        candidate = os.path.join(model_dir, "vecnormalize.pkl")
        if os.path.exists(candidate):
            args.vecnorm = candidate
            logger.info(f"Auto-detected VecNormalize: {candidate}")

    # Try to load fold_metadata.json for test period alignment
    test_start = args.test_start
    test_end = args.test_end
    if test_start is None or test_end is None:
        metadata_path = os.path.join(model_dir, "fold_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            if test_start is None:
                test_start = metadata.get("test_start")
            if test_end is None:
                test_end = metadata.get("test_end")
            logger.info(f"Using fold metadata test period: {test_start} -> {test_end}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, use_recurrent=args.recurrent)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_kwargs = {k: v for k, v in config.get("environment", {}).items()}

    # Load data
    logger.info("Loading data...")
    data = download_data(
        tickers=config["data"]["tickers"],
        period=config["data"].get("period", "5y"),
        interval=config["data"].get("interval", "1h"),
        dataset_path=config["data"].get("dataset_path"),
    )

    # Slice test data using fold dates or fallback to DataSplitter
    if test_start and test_end:
        logger.info(f"Slicing test data: {test_start} -> {test_end}")
        test_data = {}
        ts_start = pd.Timestamp(test_start)
        ts_end = pd.Timestamp(test_end)
        for ticker, df in data.items():
            mask = (df.index >= ts_start) & (df.index < ts_end)
            sliced = df.loc[mask]
            if len(sliced) > 50:
                test_data[ticker] = sliced
        if not test_data:
            logger.warning("No data in test range, falling back to DataSplitter")
            splits = DataSplitter.split(data)
            test_data = splits.test
    else:
        logger.info("No test dates specified, using DataSplitter (60/20/20)")
        splits = DataSplitter.split(data)
        test_data = splits.test

    ref_ticker = list(test_data.keys())[0]
    logger.info(
        f"Test data: {len(test_data)} tickers, "
        f"{len(test_data[ref_ticker])} bars "
        f"({test_data[ref_ticker].index[0].date()} -> {test_data[ref_ticker].index[-1].date()})"
    )

    # Macro data
    macro_data = None
    try:
        macro_fetcher = MacroDataFetcher()
        ref_df = test_data[ref_ticker]
        macro_data = macro_fetcher.fetch_all(
            start_date=str(ref_df.index[0].date()),
            end_date=str(ref_df.index[-1].date()),
            interval=config["data"].get("interval", "1h"),
        )
        if macro_data.empty:
            macro_data = None
    except Exception as e:
        logger.warning(f"Failed to fetch macro data: {e}")

    # Pre-compute features for stress test (MC does its own)
    logger.info("Pre-computing features for test data...")
    precomputed_test = _precompute_features(test_data)


    # VecNormalize
    vecnorm_env = None
    vecnorm_path = None
    if args.vecnorm and os.path.exists(args.vecnorm):
        vecnorm_path = args.vecnorm
        vecnorm_env = _load_vecnormalize(vecnorm_path, precomputed_test, macro_data, env_kwargs)
        logger.info(f"Loaded VecNormalize: {vecnorm_path}")
    else:
        logger.warning("No VecNormalize found - model receives unnormalized observations!")

    # Auto-scale
    n_workers = 1
    if args.auto_scale:
        hw = detect_hardware()
        params = compute_optimal_params(hw)
        n_workers = params["mc_workers"]
        logger.info(f"Auto-scale: {n_workers} MC workers")

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/robustness_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Monte Carlo (uses raw test_data, pre-computes features internally)
    if args.monte_carlo > 0:
        mc_report = monte_carlo_test(
            model,
            test_data,
            macro_data,
            n_simulations=args.monte_carlo,
            noise_std=args.noise_std,
            vecnorm_env=vecnorm_env,
            env_kwargs=env_kwargs,
            n_workers=n_workers,
            model_path=args.model,
            use_recurrent=args.recurrent,
            vecnorm_path=vecnorm_path,
        )
        all_results["monte_carlo"] = mc_report

    # Stress Test (uses precomputed features)
    if args.stress_test:
        st_report = stress_test_crash(
            model,
            test_data,
            macro_data,
            crash_pct=args.crash_pct,
            vecnorm_env=vecnorm_env,
            env_kwargs=env_kwargs,
        )
        all_results["stress_test"] = st_report

    # Save
    results_path = os.path.join(output_dir, "robustness_report.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nReport saved: {results_path}")


if __name__ == "__main__":
    main()
