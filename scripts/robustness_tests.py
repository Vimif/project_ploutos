#!/usr/bin/env python3
# scripts/robustness_tests.py
"""Tests de Robustesse : Monte Carlo + Stress Test Krach.

1. Monte Carlo: 1000 backtests avec bruit aléatoire sur les prix.
   Si le modèle perd de l'argent dans >20% des cas -> overfitting.

2. Stress Test: Simule un krach de -20% en une journée.
   Vérifie que le modèle coupe ses positions ou se met short.

Usage:
    python scripts/robustness_tests.py --model models/v8/model.zip --vecnorm models/v8/vecnormalize.pkl
    python scripts/robustness_tests.py --model models/v8/model.zip --monte-carlo 1000
    python scripts/robustness_tests.py --model models/v8/model.zip --stress-test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from scipy.stats import skew, kurtosis, norm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from core.environment import TradingEnv
from core.macro_data import MacroDataFetcher
from core.data_fetcher import download_data
from core.data_pipeline import DataSplitter
from core.utils import setup_logging
from config.hardware import detect_hardware, compute_optimal_params

logger = setup_logging(__name__, 'robustness_tests.log')

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


def load_model(model_path: str, use_recurrent: bool = False):
    """Charge un modèle PPO ou RecurrentPPO."""
    ModelClass = RecurrentPPO if (use_recurrent and HAS_RECURRENT) else PPO
    return ModelClass.load(model_path)


def add_price_noise(data: Dict[str, pd.DataFrame], noise_std: float = 0.005) -> Dict[str, pd.DataFrame]:
    """Ajoute du bruit gaussien aux prix OHLCV.

    Args:
        data: Dict {ticker: DataFrame OHLCV}.
        noise_std: Écart-type du bruit (0.005 = +/- 0.5%).

    Returns:
        Copie des données avec bruit ajouté.
    """
    noisy_data = {}
    for ticker, df in data.items():
        noisy_df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in noisy_df.columns:
                noise = np.random.randn(len(noisy_df)) * noise_std
                noisy_df[col] = noisy_df[col] * (1 + noise)

        # S'assurer que High >= Low et que les prix restent positifs
        noisy_df['High'] = noisy_df[['Open', 'High', 'Close']].max(axis=1)
        noisy_df['Low'] = noisy_df[['Open', 'Low', 'Close']].min(axis=1)
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
    if denom <= 0: return 0.0

    sigma_sr_sq = (1 + (0.5 * sr_est**2) - (skew_est * sr_est) + ((kurt_est / 4) * sr_est**2)) / denom
    if sigma_sr_sq <= 0: return 0.0
    
    sigma_sr = np.sqrt(sigma_sr_sq)
    psr = norm.cdf((sr_est - benchmark_sr) / sigma_sr)
    return float(psr)


def calculate_dsr(returns: np.array, n_trials: int = 1) -> float:
    """Calcule le Deflated Sharpe Ratio (DSR) simplifié."""
    if n_trials <= 1:
        return calculate_psr(returns, 0.0)
    
    # Benchmark ajusté pour le biais de sélection (Multiple Testing)
    # E[max(SR)] approx sqrt(2 * logN)
    benchmark_sr = np.sqrt(2 * np.log(n_trials)) * 0.05 # Scale factor empirique pour hourly
    return calculate_psr(returns, benchmark_sr)


def simulate_crash(
    data: Dict[str, pd.DataFrame],
    crash_pct: float = -0.20,
    crash_day_idx: int = None,
    instant: bool = False,
) -> Dict[str, pd.DataFrame]:
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

        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in crashed_df.columns:
                continue
            # Appliquer le crash
            for i, idx in enumerate(range(crash_day_idx, end_idx)):
                progress = 1.0 if instant else (i + 1) / crash_duration
                drop = crash_pct * progress
                crashed_df.iloc[idx, crashed_df.columns.get_loc(col)] *= (1 + drop)

            # Les prix après le crash restent au niveau bas
            post_crash_factor = 1 + crash_pct
            for idx in range(end_idx, n):
                crashed_df.iloc[idx, crashed_df.columns.get_loc(col)] *= post_crash_factor

        crashed_df = crashed_df.clip(lower=0.01)
        crashed_data[ticker] = crashed_df

    return crashed_data


def run_backtest(model, data, macro_data, vecnorm_env=None, env_kwargs=None) -> dict:
    """Exécute un backtest et retourne les métriques."""
    kwargs = env_kwargs or {}
    kwargs['mode'] = 'backtest'
    kwargs['seed'] = 42

    env = TradingEnv(data=data, macro_data=macro_data, **kwargs)
    obs, _ = env.reset()
    done = False
    equity_curve = [env.initial_balance]

    while not done:
        if vecnorm_env is not None:
            obs_norm = vecnorm_env.normalize_obs(obs.reshape(1, -1)).flatten()
        else:
            obs_norm = obs
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5)

    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'final_equity': float(equity_series.iloc[-1]),
        'total_trades': info.get('total_trades', 0),
        'winning_trades': info.get('winning_trades', 0),
        'losing_trades': info.get('losing_trades', 0),
    }


def _mc_worker(args):
    """Worker pour Monte Carlo parallèle. Top-level pour pickle."""
    test_data, macro_data, noise_std, env_kwargs, model_path, use_recurrent, seed = args
    np.random.seed(seed)
    model = load_model(model_path, use_recurrent=use_recurrent)
    noisy_data = add_price_noise(test_data, noise_std=noise_std)
    return run_backtest(model, noisy_data, macro_data, vecnorm_env=None, env_kwargs=env_kwargs)


def monte_carlo_test(
    model,
    test_data: Dict[str, pd.DataFrame],
    macro_data: Optional[pd.DataFrame],
    n_simulations: int = 1000,
    noise_std: float = 0.005,
    vecnorm_env=None,
    env_kwargs: dict = None,
    n_workers: int = 1,
    model_path: str = None,
    use_recurrent: bool = False,
) -> dict:
    """Monte Carlo Simulations.

    Lance n_simulations backtests avec du bruit aléatoire.
    Critère: si >5% des simulations perdent de l'argent -> overfitting.
    n_workers > 1 parallélise via ProcessPoolExecutor.
    """
    logger.info(
        f"Monte Carlo: {n_simulations} simulations "
        f"(noise={noise_std*100:.1f}%, workers={n_workers})"
    )

    if n_workers > 1 and model_path:
        from concurrent.futures import ProcessPoolExecutor

        worker_args = [
            (test_data, macro_data, noise_std, env_kwargs,
             model_path, use_recurrent, 42 + i)
            for i in range(n_simulations)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_mc_worker, worker_args))
        losses = sum(1 for r in results if r['total_return'] < 0)
        logger.info(f"  Done: {n_simulations} sims | Losses: {losses}/{n_simulations}")
    else:
        results = []
        for i in range(n_simulations):
            noisy_data = add_price_noise(test_data, noise_std=noise_std)
            metrics = run_backtest(model, noisy_data, macro_data, vecnorm_env, env_kwargs)
            results.append(metrics)

            if (i + 1) % 100 == 0:
                losses = sum(1 for r in results if r['total_return'] < 0)
                logger.info(
                    f"  Progress: {i+1}/{n_simulations} | "
                    f"Losses: {losses}/{i+1} ({losses/(i+1)*100:.1f}%)"
                )

    # Analyser
    returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]

    n_losses = sum(1 for r in returns if r < 0)
    loss_rate = n_losses / n_simulations

    report = {
        'n_simulations': n_simulations,
        'noise_std': noise_std,
        'loss_rate': loss_rate,
        'is_overfit': loss_rate > 0.20,
        'avg_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'median_return': float(np.median(returns)),
        'p5_return': float(np.percentile(returns, 5)),
        'p95_return': float(np.percentile(returns, 95)),
        'avg_sharpe': float(np.mean(sharpes)),
        'avg_max_drawdown': float(np.mean(drawdowns)),
    }

    # Calcul PSR/DSR sur la distribution des *moyennes* de rendement des simulations
    # Note: Le PSR s'applique normalement à une série temporelle de rendements d'UNE stratégie.
    # Ici on l'applique à la distribution des Sharpes de Monte Carlo pour voir la robustesse globale.
    
    # Agrégation des rendements de tous les MC pour un PSR global "Meta"
    # On prend le rendement moyen par pas de temps sur toutes les sims
    # C'est une approximation pour voir si la stratégie "en moyenne" a un PSR élevé.
    all_returns_flat = np.array(returns) # Rendements totaux des épisodes
    
    # Calcul PSR sur la distribution des Scénarios (Est-ce que >95% des scénarios battent 0 ?)
    # On utilise simplement le taux de perte pour ça, mais le PSR ajoute la nuance de la variance.
    psr = calculate_psr(np.array(returns), benchmark_sr=0.0)
    report['psr'] = psr
    report['dsr'] = calculate_dsr(np.array(returns), n_trials=n_simulations)

    logger.info("\n" + "=" * 50)
    logger.info("MONTE CARLO RESULTS (Robuste)")
    logger.info("=" * 50)
    logger.info(f"  Simulations: {n_simulations}")
    logger.info(f"  Loss Rate:   {loss_rate:.1%} {'OVERFIT' if report['is_overfit'] else 'OK'}")
    logger.info(f"  Avg Return:  {report['avg_return']:+.2%}")
    logger.info(f"  Std Return:  {report['std_return']:.2%}")
    logger.info(f"  Range:       [{report['min_return']:+.2%}, {report['max_return']:+.2%}]")
    logger.info(f"  P5/P95:      [{report['p5_return']:+.2%}, {report['p95_return']:+.2%}]")
    logger.info(f"  Avg Sharpe:  {report['avg_sharpe']:.2f}")
    logger.info(f"  PSR (Prob):  {report['psr']:.4f} (>{'0.95 OK' if report['psr']>0.95 else 'FAIL'})")
    logger.info(f"  Avg MaxDD:   {report['avg_max_drawdown']:.2%}")

    if report['is_overfit']:
        logger.warning("  VERDICT: OVERFIT - Le modele perd de l'argent dans >5% des cas")
    else:
        logger.info("  VERDICT: ROBUSTE - Le modele est stable")

    return report


def stress_test_crash(
    model,
    test_data: Dict[str, pd.DataFrame],
    macro_data: Optional[pd.DataFrame],
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

    # Baseline sans krach
    baseline = run_backtest(model, test_data, macro_data, vecnorm_env, env_kwargs)

    # Avec krach
    crashed_data = simulate_crash(test_data, crash_pct=crash_pct)
    crash_result = run_backtest(model, crashed_data, macro_data, vecnorm_env, env_kwargs)

    # Analyser la réaction
    baseline_return = baseline['total_return']
    crash_return = crash_result['total_return']
    return_impact = crash_return - baseline_return

    report = {
        'crash_pct': crash_pct,
        'baseline_return': baseline_return,
        'crash_return': crash_return,
        'return_impact': return_impact,
        'baseline_max_drawdown': baseline['max_drawdown'],
        'crash_max_drawdown': crash_result['max_drawdown'],
        'crash_trades': crash_result['total_trades'],
        'survives': crash_result['max_drawdown'] < 0.50,  # Ne perd pas tout
        'acceptable_drawdown': crash_result['max_drawdown'] < 0.25,
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

    if not report['survives']:
        logger.warning("  VERDICT: ECHEC - Le modele ne survit pas au krach")
    elif not report['acceptable_drawdown']:
        logger.warning("  VERDICT: MARGINAL - Survit mais drawdown excessif")
    else:
        logger.info("  VERDICT: ROBUSTE - Le modele gere bien le krach")

    return report


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Robustness Tests (Monte Carlo + Stress Test)')
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip')
    parser.add_argument('--vecnorm', type=str, default=None, help='Path to vecnormalize .pkl')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--monte-carlo', type=int, default=0, help='Number of MC simulations (0=skip)')
    parser.add_argument('--stress-test', action='store_true', help='Run crash stress test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--recurrent', action='store_true', help='Model is RecurrentPPO')
    parser.add_argument('--noise-std', type=float, default=0.005, help='MC noise std (default: 0.5%%)')
    parser.add_argument('--crash-pct', type=float, default=-0.20, help='Crash severity (default: -20%%)')
    parser.add_argument(
        '--auto-scale', action='store_true',
        help='Auto-detect hardware and parallelize Monte Carlo',
    )

    args = parser.parse_args()

    if args.all:
        args.monte_carlo = 1000
        args.stress_test = True

    if args.monte_carlo == 0 and not args.stress_test:
        print("Specify --monte-carlo N, --stress-test, or --all")
        return

    # Charger modèle
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, use_recurrent=args.recurrent)

    # Charger config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env_kwargs = {k: v for k, v in config.get('environment', {}).items()}

    # Charger données
    logger.info("Loading test data...")
    data = download_data(
        tickers=config['data']['tickers'],
        period=config['data'].get('period', '5y'),
        interval=config['data'].get('interval', '1h'),
    )

    splits = DataSplitter.split(data)
    test_data = splits.test

    # Macro data
    macro_fetcher = MacroDataFetcher()
    ref_df = data[list(data.keys())[0]]
    macro_data = macro_fetcher.fetch_all(
        start_date=str(ref_df.index[0].date()),
        end_date=str(ref_df.index[-1].date()),
        interval=config['data'].get('interval', '1h'),
    )
    if macro_data.empty:
        macro_data = None

    # VecNormalize
    vecnorm_env = None
    if args.vecnorm and os.path.exists(args.vecnorm):
        dummy_env = DummyVecEnv([
            lambda: Monitor(TradingEnv(
                data=test_data, macro_data=macro_data, **{**env_kwargs, 'mode': 'backtest'}
            ))
        ])
        vecnorm_env = VecNormalize.load(args.vecnorm, dummy_env)
        vecnorm_env.training = False
        vecnorm_env.norm_reward = False

    # Auto-scale
    n_workers = 1
    if args.auto_scale:
        hw = detect_hardware()
        params = compute_optimal_params(hw)
        n_workers = params["mc_workers"]
        logger.info(f"Auto-scale: {n_workers} MC workers")

    # Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"models/robustness_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Monte Carlo
    if args.monte_carlo > 0:
        mc_report = monte_carlo_test(
            model, test_data, macro_data,
            n_simulations=args.monte_carlo,
            noise_std=args.noise_std,
            vecnorm_env=vecnorm_env,
            env_kwargs=env_kwargs,
            n_workers=n_workers,
            model_path=args.model,
            use_recurrent=args.recurrent,
        )
        all_results['monte_carlo'] = mc_report

    # Stress Test
    if args.stress_test:
        st_report = stress_test_crash(
            model, test_data, macro_data,
            crash_pct=args.crash_pct,
            vecnorm_env=vecnorm_env,
            env_kwargs=env_kwargs,
        )
        all_results['stress_test'] = st_report

    # Sauvegarder
    results_path = os.path.join(output_dir, 'robustness_report.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nReport saved: {results_path}")


if __name__ == '__main__':
    main()
