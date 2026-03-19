#!/usr/bin/env python3
# scripts/validate_pipeline.py
"""Pipeline de validation automatique: train → val → backtest → certification.

Exécute tout le workflow de validation en une seule commande:
    1. Télécharge les données
    2. Split temporel train/val/test
    3. Entraîne un modèle rapide (ou utilise un modèle existant)
    4. Évalue sur les données de validation
    5. Backtest sur les données de test (OOS)
    6. Calcule les métriques de certification
    7. Génère un rapport

Usage:
    # Validation complète (entraînement + backtest)
    python scripts/validate_pipeline.py --tickers AAPL MSFT NVDA --timesteps 100000

    # Validation d'un modèle existant
    python scripts/validate_pipeline.py --model models/v7_sp500/best.zip --tickers AAPL MSFT NVDA

    # Validation rapide (smoke test)
    python scripts/validate_pipeline.py --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import warnings
import argparse
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from core.data_fetcher import download_data
from core.data_pipeline import DataSplitter
from core.environment import TradingEnv
from core.features import FeatureEngineer  # V9 Turbo


def run_validation(
    tickers: list[str],
    model_path: str | None = None,
    total_timesteps: int = 100_000,
    period: str = "2y",
    interval: str = "1h",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str = "reports/validation",
) -> dict:
    """Pipeline de validation complet."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tickers": tickers,
        "seed": seed,
        "stages": {},
    }

    print("=" * 70)
    print("🚀 PIPELINE DE VALIDATION (V9)")
    print("=" * 70)

    # ================================================================
    # Stage 1: Download data & Feature Engineering
    # ================================================================
    print(f"\n📥 Stage 1/6: Téléchargement & Features...")
    try:
        data = download_data(tickers, period=period, interval=interval)
        if not data or len(data) == 0:
            raise ValueError("Aucune donnée récupérée")

        # V9: Calculate Features immediately
        fe = FeatureEngineer()
        for t, df in data.items():
            data[t] = fe.calculate_all_features(df)

        print(f"  ✅ {len(data)} tickers chargés & features calculées")

        results["stages"]["download"] = {
            "status": "OK",
            "n_tickers": len(data),
            "bars_per_ticker": {t: len(df) for t, df in data.items()},
        }
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results["stages"]["download"] = {"status": "FAIL", "error": str(e)}
        return results

    # ================================================================
    # Stage 2: Split temporel
    # ================================================================
    # ... (code identique) ...
    print(f"\n📊 Stage 2/6: Split temporel ({train_ratio}/{val_ratio}/{test_ratio})...")
    try:
        splits = DataSplitter.split(data, train_ratio, val_ratio, test_ratio)
        DataSplitter.validate_no_overlap(splits)
        # ... logs ...
        results["stages"]["split"] = {"status": "OK", "info": splits.info}
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results["stages"]["split"] = {"status": "FAIL", "error": str(e)}
        return results

    # ================================================================
    # Stage 3: Training (ou chargement modèle)
    # ================================================================
    # ...
    # (Dans le bloc else pour training)
    # ...
    #         from stable_baselines3 import PPO
    #         from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    #         from stable_baselines3.common.monitor import Monitor
    #
    #         # Utiliser TradingEnv V9 avec features_precomputed=True
    #         train_env = DummyVecEnv([
    #             lambda: Monitor(TradingEnv(
    #                 splits.train, mode='train', seed=seed, features_precomputed=True
    #             ))
    #         ])
    #         # ...
    # ...

    # ================================================================
    # Stage 4: Evaluation (val data)
    # ================================================================
    print(f"\n📈 Stage 4/6: Évaluation sur données de validation...")
    try:
        val_env = TradingEnv(splits.val, mode="eval", seed=seed, features_precomputed=True)
        val_results = _run_episodes(model, val_env, n_episodes=3, label="Val")
        results["stages"]["validation"] = {"status": "OK", **val_results}
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results["stages"]["validation"] = {"status": "FAIL", "error": str(e)}

    # ================================================================
    # Stage 5: Backtest OOS (test data)
    # ================================================================
    print(f"\n🎯 Stage 5/6: Backtest Out-of-Sample (données test)...")
    try:
        test_env = TradingEnv(splits.test, mode="backtest", seed=seed, features_precomputed=True)
        oos_results = _run_episodes(model, test_env, n_episodes=1, label="OOS")

        results["stages"]["oos_backtest"] = {
            "status": "OK",
            **oos_results,
        }
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results["stages"]["oos_backtest"] = {"status": "FAIL", "error": str(e)}

    # ================================================================
    # Stage 6: Certification
    # ================================================================
    print(f"\n🏆 Stage 6/6: Certification...")
    cert = _certify(results)
    results["stages"]["certification"] = cert

    # ================================================================
    # Sauvegarde rapport
    # ================================================================
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"validation_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📄 Rapport sauvegardé: {report_path}")

    _print_summary(results)
    return results


def _run_episodes(model, env, n_episodes: int, label: str) -> dict:
    """Execute n épisodes et retourne les métriques."""
    all_returns = []
    all_trades = []
    all_win_rates = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward

        total_return = info.get("total_return", 0.0)
        trades = info.get("total_trades", 0)
        wins = info.get("winning_trades", 0)
        win_rate = wins / max(trades, 1)

        all_returns.append(total_return)
        all_trades.append(trades)
        all_win_rates.append(win_rate)

        print(
            f"  [{label}] Ep {ep + 1}/{n_episodes}: "
            f"Return={total_return * 100:.2f}% | "
            f"Trades={trades} | "
            f"WR={win_rate:.1%}"
        )

    return {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_trades": float(np.mean(all_trades)),
        "mean_win_rate": float(np.mean(all_win_rates)),
        "n_episodes": n_episodes,
    }


def _certify(results: dict) -> dict:
    """Évalue la qualité du modèle et détermine s'il est certifié."""
    cert = {"checks": {}}

    # Check 1: OOS return > 0
    oos = results["stages"].get("oos_backtest", {})
    oos_return = oos.get("mean_return", -1)
    cert["checks"]["oos_positive_return"] = oos_return > 0

    # Check 2: OOS win rate > 50%
    oos_wr = oos.get("mean_win_rate", 0)
    cert["checks"]["oos_win_rate_gt_50"] = oos_wr > 0.5

    # Check 3: OOS trades > 0
    oos_trades = oos.get("mean_trades", 0)
    cert["checks"]["oos_has_trades"] = oos_trades > 0

    # Check 4: Val return direction matches OOS
    val = results["stages"].get("validation", {})
    val_return = val.get("mean_return", 0)
    cert["checks"]["val_oos_direction_match"] = (val_return > 0 and oos_return > 0) or (
        val_return <= 0 and oos_return <= 0
    )

    # Verdict
    n_passed = sum(1 for v in cert["checks"].values() if v)
    n_total = len(cert["checks"])
    cert["passed"] = n_passed
    cert["total"] = n_total
    cert["certified"] = n_passed == n_total
    cert["status"] = "CERTIFIED ✅" if cert["certified"] else f"FAILED ({n_passed}/{n_total})"

    return cert


def _print_summary(results: dict):
    """Affiche un résumé lisible."""
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DE VALIDATION")
    print("=" * 70)

    for stage_name, stage_data in results["stages"].items():
        status = stage_data.get("status", "?")
        icon = "✅" if "OK" in str(status) or "CERTIFIED" in str(status) else "❌"
        print(f"  {icon} {stage_name}: {status}")

    cert = results["stages"].get("certification", {})
    if cert:
        print(f"\n  🏆 Certification: {cert.get('status', '?')}")
        for check, passed in cert.get("checks", {}).items():
            icon = "✅" if passed else "❌"
            print(f"     {icon} {check}")

    print("=" * 70)


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de validation: train → val → backtest → certification"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA"],
        help="Tickers à utiliser (défaut: AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin vers un modèle existant (skip training)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Nombre de timesteps pour entraînement rapide (défaut: 100K)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Période de données (défaut: 2y)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Mode rapide (10K steps, 3 tickers)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/validation",
        help="Dossier de sortie (défaut: reports/validation)",
    )

    args = parser.parse_args()

    if args.quick:
        args.tickers = ["AAPL", "MSFT", "NVDA"]
        args.timesteps = 10_000

    run_validation(
        tickers=args.tickers,
        model_path=args.model,
        total_timesteps=args.timesteps,
        period=args.period,
        seed=args.seed,
        output_dir=args.output,
    )
