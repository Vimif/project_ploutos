#!/usr/bin/env python3
"""
===============================================================================
  BACKTEST ULTIMATE — Validation Complete des Modeles Ploutos
===============================================================================

11 modules d'analyse:
  1. Trade Journal         — Log detaille de chaque trade + timing quality
  2. Walk-Forward          — Fenetres glissantes pour tester la stabilite
  3. Benchmarks            — Buy & Hold, Random Agent, Hold Cash
  4. Metriques exhaustives — Sharpe, Sortino, Calmar, Profit Factor, etc.
  5. Monte Carlo           — Test statistique: la strategie bat-elle le hasard?
  6. Stress Test           — Robustesse sous couts/slippage augmentes
  7. Rapport Certification — Verdict PASS/FAIL + score 0-100 + export JSON
  8. OOS Validation        — Detection chevauchement training/backtest
  9. Bootstrap CI          — Intervalles de confiance 95% sur les metriques
 10. Market Regime         — Performance par regime (bull/bear/range)
 11. Stress Scenarios      — Crash, range-bound, gap simulations

Usage:
  python scripts/backtest_ultimate.py --model data/models/brain_tech.zip
  python scripts/backtest_ultimate.py --model data/models/brain_tech.zip --quick
  python scripts/backtest_ultimate.py --model models/v7_sp500/ploutos_v7_sp500.zip
  python scripts/backtest_ultimate.py --model ... --oos-only
"""

import sys
import os
from pathlib import Path

# Fix Windows UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import yaml
except ImportError:
    yaml = None

from core.data_fetcher import UniversalDataFetcher
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS_15 = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'SPY', 'QQQ', 'VOO', 'VTI', 'XLE', 'XLF', 'XLK', 'XLV'
]
TICKERS_10 = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'SPY', 'QQQ', 'VOO', 'XLE', 'XLF']

INITIAL_BALANCE = 100_000

# Pass/Fail thresholds
THRESHOLDS = {
    'return_min': 0.0,
    'sharpe_min': 0.5,
    'sortino_min': 0.7,
    'max_drawdown_max': 0.25,
    'calmar_min': 0.5,
    'win_rate_min': 0.45,
    'profit_factor_min': 1.0,
    'buy_quality_min': 0.35,
    'avg_trade_duration_min': 3,
    'recovery_factor_min': 1.0,
    'monte_carlo_p_max': 0.05,
}

# V6 environment defaults
ENV_PARAMS_V6 = dict(
    initial_balance=INITIAL_BALANCE,
    commission=0.0,
    sec_fee=0.0000221,
    finra_taf=0.000145,
    max_steps=5000,
    buy_pct=0.2,
    slippage_model='realistic',
    spread_bps=2.0,
    max_position_pct=0.25,
    max_trades_per_day=10,
    min_holding_period=2,
    reward_scaling=1.5,
    use_sharpe_penalty=True,
    use_drawdown_penalty=True,
    reward_trade_success=0.5,
    penalty_overtrading=0.005,
    drawdown_penalty_factor=3.0,
)

# V7 environment defaults (same env class, adjusted params for ~22 tickers)
ENV_PARAMS_V7 = dict(
    initial_balance=INITIAL_BALANCE,
    commission=0.0,
    sec_fee=0.0000221,
    finra_taf=0.000145,
    max_steps=2500,
    buy_pct=0.15,
    slippage_model='realistic',
    spread_bps=2.0,
    max_position_pct=0.20,
    max_trades_per_day=15,
    min_holding_period=2,
    reward_scaling=1.5,
    use_sharpe_penalty=True,
    use_drawdown_penalty=True,
    reward_trade_success=0.5,
    penalty_overtrading=0.005,
    drawdown_penalty_factor=3.0,
)


# ============================================================================
# V7 METADATA & CONFIG LOADING
# ============================================================================

def load_v7_metadata(model_path):
    """Charge les metadata V7 (tickers, secteurs, config) a cote du modele."""
    p = Path(model_path)
    metadata_path = p.with_name(p.stem + '_metadata.json')
    config_path = p.with_name(p.stem + '_config.json')
    vecnorm_path = p.with_name(p.stem + '_vecnormalize.pkl')

    metadata = None
    config = None

    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"  Metadata V7 chargee: {metadata_path.name}")
        logger.info(f"    Version: {metadata.get('version', '?')}")
        logger.info(f"    Tickers: {metadata.get('n_tickers', '?')} ({', '.join(metadata.get('tickers', [])[:5])}...)")

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"  Config V7 chargee: {config_path.name}")

    vecnorm_exists = vecnorm_path.exists()
    if vecnorm_exists:
        logger.info(f"  VecNormalize: {vecnorm_path.name}")

    return metadata, config, vecnorm_path if vecnorm_exists else None


def load_yaml_config(config_path):
    """Charge une config YAML et extrait les params environnement."""
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# ENVIRONMENT AUTO-DETECTION
# ============================================================================

def detect_environment(model, metadata=None, config=None):
    """Auto-detecte l'environnement compatible avec le modele.

    Si metadata V7 est fournie, utilise ses tickers et params.
    Sinon, auto-detection par taille d'observation.
    """
    obs_size = model.observation_space.shape[0]
    logger.info(f"  Observation space du modele: {obs_size} dims")

    from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

    # --- V7: metadata fournie avec tickers specifiques ---
    if metadata and metadata.get('version', '').startswith('v7'):
        tickers = metadata.get('tickers', [])
        n_tickers = len(tickers)

        # Use config env params if available, else V7 defaults
        if config and 'environment' in config:
            env_cfg = config['environment']
            params = dict(
                initial_balance=env_cfg.get('initial_balance', INITIAL_BALANCE),
                commission=env_cfg.get('commission', 0.0),
                sec_fee=env_cfg.get('sec_fee', 0.0000221),
                finra_taf=env_cfg.get('finra_taf', 0.000145),
                max_steps=env_cfg.get('max_steps', 2500),
                buy_pct=env_cfg.get('buy_pct', 0.15),
                slippage_model=env_cfg.get('slippage_model', 'realistic'),
                spread_bps=env_cfg.get('spread_bps', 2.0),
                max_position_pct=env_cfg.get('max_position_pct', 0.20),
                max_trades_per_day=env_cfg.get('max_trades_per_day', 15),
                min_holding_period=env_cfg.get('min_holding_period', 2),
                reward_scaling=env_cfg.get('reward_scaling', 1.5),
                use_sharpe_penalty=env_cfg.get('use_sharpe_penalty', True),
                use_drawdown_penalty=env_cfg.get('use_drawdown_penalty', True),
                reward_trade_success=env_cfg.get('reward_trade_success', 0.5),
                penalty_overtrading=env_cfg.get('penalty_overtrading', 0.005),
                drawdown_penalty_factor=env_cfg.get('drawdown_penalty_factor', 3.0),
            )
        else:
            params = dict(ENV_PARAMS_V7)

        logger.info(f"  -> V7 S&P 500 Sectors: {n_tickers} tickers")
        return 'V7', n_tickers, tickers, UniversalTradingEnvV6BetterTiming, params

    # --- V6/V7 sans metadata: detection par obs_size ---
    # Formula: obs = n_tickers * 85 + n_tickers + 3 = n_tickers * 86 + 3
    for n_tickers in [22, 20, 15, 10, 5, 1]:
        expected_obs = n_tickers * 85 + n_tickers + 3
        if abs(expected_obs - obs_size) < 20:
            is_v7 = n_tickers > 15
            version = 'V7' if is_v7 else 'V6'
            params = dict(ENV_PARAMS_V7) if is_v7 else dict(ENV_PARAMS_V6)
            logger.info(f"  -> Match probable: {version} BetterTiming avec {n_tickers} tickers (~{expected_obs} dims)")
            return version, n_tickers, None, UniversalTradingEnvV6BetterTiming, params

    # Fallback
    logger.info(f"  -> Detection par essai direct avec V6...")
    return 'V6_AUTO', None, None, UniversalTradingEnvV6BetterTiming, dict(ENV_PARAMS_V6)


def create_env_with_check(env_class, data, env_params, model_obs_size):
    """Cree l'environnement et verifie la compatibilite."""
    env = env_class(data=data, **env_params)
    obs, _ = env.reset()

    if obs.shape[0] != model_obs_size:
        raise ValueError(
            f"MISMATCH: Env produit {obs.shape[0]} dims, modele attend {model_obs_size}. "
            f"Ce modele a probablement ete entraine avec un environnement different."
        )

    return env


def create_vec_env_normalized(env_class, data, env_params, vecnorm_path=None):
    """Cree un DummyVecEnv avec VecNormalize (pour modeles V7)."""
    def _make():
        return env_class(data=data, **env_params)

    vec_env = DummyVecEnv([_make])

    if vecnorm_path:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False  # No stat updates during inference
        vec_env.norm_reward = False
        logger.info(f"  VecNormalize charge depuis {vecnorm_path}")
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False,
            clip_obs=10.0, training=False,
        )
        logger.info(f"  VecNormalize cree (sans stats pre-calculees)")

    return vec_env


# ============================================================================
# TRADE JOURNAL — Module 1
# ============================================================================

class TradeJournal:
    """Enregistre et analyse chaque trade via delta portfolio avant/apres step."""

    def __init__(self, tickers):
        self.tickers = tickers
        self.trades = []
        self.open_positions = {}  # ticker -> {buy_step, buy_price, buy_sma20}

    def snapshot_portfolio(self, env):
        """Capture l'etat du portfolio AVANT le step."""
        return {t: env.portfolio.get(t, 0.0) for t in self.tickers}

    def on_step_done(self, step, actions, env, data_dict, portfolio_before):
        """Appelee APRES env.step() — compare avant/apres pour detecter trades."""
        for i, ticker in enumerate(self.tickers):
            qty_before = portfolio_before.get(ticker, 0.0)
            qty_after = env.portfolio.get(ticker, 0.0)
            action = actions[i] if i < len(actions) else 0

            current_price = self._safe_price(env, ticker, step)
            if current_price <= 0:
                continue

            sma20 = self._compute_sma(data_dict, ticker, step, window=20)

            # DETECT BUY: qty increased
            if qty_after > qty_before + 1e-6 and ticker not in self.open_positions:
                entry_price = env.entry_prices.get(ticker, current_price) if hasattr(env, 'entry_prices') else current_price
                self.open_positions[ticker] = {
                    'buy_step': step,
                    'buy_price': entry_price,
                    'buy_sma20': sma20,
                    'ticker': ticker,
                    'qty': qty_after,
                }

            # DETECT SELL: qty went to ~0
            elif qty_before > 1e-6 and qty_after < 1e-6 and ticker in self.open_positions:
                pos = self.open_positions.pop(ticker)
                pnl_pct = (current_price - pos['buy_price']) / pos['buy_price'] if pos['buy_price'] > 0 else 0

                self.trades.append({
                    'ticker': pos['ticker'],
                    'buy_step': pos['buy_step'],
                    'sell_step': step,
                    'duration': step - pos['buy_step'],
                    'buy_price': pos['buy_price'],
                    'sell_price': current_price,
                    'pnl_pct': pnl_pct,
                    'is_good_buy': pos['buy_price'] < pos['buy_sma20'] if pos['buy_sma20'] > 0 else False,
                    'is_good_sell': current_price > sma20 if sma20 > 0 else False,
                    'is_winner': pnl_pct > 0,
                })

    def _safe_price(self, env, ticker, step):
        try:
            df = env.processed_data[ticker]
            idx = min(step, len(df) - 1)
            price = df.iloc[idx]['Close']
            if np.isnan(price) or np.isinf(price) or price <= 0:
                return df['Close'].median()
            return price
        except Exception:
            return 0.0

    def _compute_sma(self, data_dict, ticker, step, window=20):
        try:
            df = data_dict[ticker]
            end = min(step + 1, len(df))
            start = max(0, end - window)
            if end - start < 5:
                return 0.0
            return df['Close'].iloc[start:end].mean()
        except Exception:
            return 0.0

    def get_metrics(self):
        if not self.trades:
            return {k: 0.0 for k in [
                'total_completed_trades', 'buy_quality', 'sell_quality',
                'avg_trade_duration', 'avg_pnl_pct', 'win_rate',
                'profit_factor', 'best_trade', 'worst_trade'
            ]}

        n = len(self.trades)
        good_buys = sum(1 for t in self.trades if t['is_good_buy'])
        good_sells = sum(1 for t in self.trades if t['is_good_sell'])
        winners = sum(1 for t in self.trades if t['is_winner'])
        gains = sum(t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0)
        losses = abs(sum(t['pnl_pct'] for t in self.trades if t['pnl_pct'] < 0))

        return {
            'total_completed_trades': n,
            'buy_quality': good_buys / n if n > 0 else 0.0,
            'sell_quality': good_sells / n if n > 0 else 0.0,
            'avg_trade_duration': float(np.mean([t['duration'] for t in self.trades])),
            'avg_pnl_pct': float(np.mean([t['pnl_pct'] for t in self.trades])),
            'win_rate': winners / n if n > 0 else 0.0,
            'profit_factor': gains / losses if losses > 0 else (10.0 if gains > 0 else 0.0),
            'best_trade': float(max(t['pnl_pct'] for t in self.trades)),
            'worst_trade': float(min(t['pnl_pct'] for t in self.trades)),
        }


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Execute un modele sur un environnement et collecte les resultats."""

    def __init__(self, model, tickers, env_class, env_params, model_obs_size):
        self.model = model
        self.tickers = tickers
        self.env_class = env_class
        self.env_params = env_params
        self.model_obs_size = model_obs_size

    def run_episode(self, data, track_trades=True, record_actions=False):
        """Execute un episode complet."""
        env = create_env_with_check(self.env_class, data, self.env_params, self.model_obs_size)
        journal = TradeJournal(self.tickers) if track_trades else None

        obs, info = env.reset()
        done = False
        truncated = False
        step = 0

        initial_balance = self.env_params.get('initial_balance', INITIAL_BALANCE)
        portfolio_history = [initial_balance]
        actions_log = [] if record_actions else None

        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)

            # Snapshot portfolio BEFORE step
            portfolio_before = None
            if journal:
                portfolio_before = journal.snapshot_portfolio(env)

            if record_actions:
                actions_log.append(action.copy())

            obs, reward, done, truncated, info = env.step(action)

            # Track trades AFTER step (compare before/after)
            if journal:
                journal.on_step_done(step, action, env, data, portfolio_before)

            equity = info.get('equity', info.get('portfolio_value', initial_balance))
            portfolio_history.append(equity)
            step += 1

        final_val = portfolio_history[-1]
        return {
            'portfolio_history': np.array(portfolio_history),
            'final_value': final_val,
            'total_return': (final_val - initial_balance) / initial_balance,
            'total_trades': info.get('total_trades', 0),
            'winning_trades': info.get('winning_trades', 0),
            'losing_trades': info.get('losing_trades', 0),
            'steps': step,
            'journal': journal,
            'actions_log': actions_log,
        }


# ============================================================================
# METRIQUES — Module 4
# ============================================================================

def compute_portfolio_metrics(portfolio_history, initial_balance, annualize_factor=252 * 6.5):
    """Calcule toutes les metriques financieres."""
    pv = np.array(portfolio_history, dtype=np.float64)
    returns = np.diff(pv) / pv[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return {k: 0.0 for k in [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'recovery_factor', 'annualized_return', 'volatility', 'n_steps'
        ]}

    total_return = (pv[-1] - initial_balance) / initial_balance
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(annualize_factor) if std_ret > 0 else 0.0

    neg_returns = returns[returns < 0]
    downside_std = np.std(neg_returns, ddof=1) if len(neg_returns) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(annualize_factor) if downside_std > 0 else 0.0

    cumulative = np.maximum.accumulate(pv)
    drawdowns = (cumulative - pv) / cumulative
    max_drawdown = float(np.max(drawdowns))

    ann_return = total_return * (annualize_factor / len(returns)) if len(returns) > 0 else 0.0
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0
    recovery = total_return / max_drawdown if max_drawdown > 0 else 0.0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'recovery_factor': recovery,
        'annualized_return': ann_return,
        'volatility': std_ret * np.sqrt(annualize_factor),
        'n_steps': len(returns),
    }


# ============================================================================
# BENCHMARKS — Module 3
# ============================================================================

def benchmark_buy_and_hold(data, initial_balance):
    amount_per = initial_balance / len(data)
    total_final = 0.0
    for ticker, df in data.items():
        if len(df) < 2:
            total_final += amount_per
            continue
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]
        if initial_price > 0:
            total_final += (amount_per / initial_price) * final_price
        else:
            total_final += amount_per
    return (total_final - initial_balance) / initial_balance


def benchmark_random_agent(env_class, data, env_params, model_obs_size, n_assets, n_runs=5):
    returns = []
    ib = env_params.get('initial_balance', INITIAL_BALANCE)
    for _ in range(n_runs):
        try:
            env = create_env_with_check(env_class, data, env_params, model_obs_size)
            obs, _ = env.reset()
            done = truncated = False
            while not (done or truncated):
                action = np.random.choice([0, 1, 2], size=n_assets, p=[0.89, 0.055, 0.055])
                obs, _, done, truncated, info = env.step(action)
            final_val = info.get('equity', info.get('portfolio_value', ib))
            returns.append((final_val - ib) / ib)
        except Exception:
            pass
    return float(np.mean(returns)) if returns else 0.0


# ============================================================================
# WALK-FORWARD — Module 2
# ============================================================================

def walk_forward_validation(engine, data_full, n_windows=6, window_days=30, shift_days=15, episodes_per_window=3):
    logger.info(f"\n{'='*60}")
    logger.info(f"  WALK-FORWARD VALIDATION ({n_windows} fenetres x {window_days}j)")
    logger.info(f"{'='*60}")

    min_len = min(len(df) for df in data_full.values())
    bars_per_day = 7
    window_bars = window_days * bars_per_day
    shift_bars = shift_days * bars_per_day

    if min_len < window_bars + shift_bars:
        logger.warning(f"  Donnees insuffisantes ({min_len} barres). Reduction des fenetres.")
        n_windows = max(2, (min_len - window_bars) // shift_bars + 1)

    window_results = []
    for w in range(n_windows):
        start = w * shift_bars
        end = start + window_bars
        if end > min_len:
            break

        wdata = {t: df.iloc[start:end].reset_index(drop=True) for t, df in data_full.items()}
        rets = []
        for ep in range(episodes_per_window):
            try:
                result = engine.run_episode(wdata, track_trades=False)
                rets.append(result['total_return'])
            except Exception as e:
                logger.warning(f"  Fenetre {w+1} ep {ep+1}: {e}")

        if rets:
            avg = np.mean(rets)
            window_results.append(avg)
            logger.info(f"  Fenetre {w+1}/{n_windows}: return={avg:+.2%}")

    if not window_results:
        return {'mean_return': 0.0, 'std_return': 1.0, 'windows': [], 'positive_pct': 0.0, 'n_windows': 0}

    return {
        'mean_return': float(np.mean(window_results)),
        'std_return': float(np.std(window_results)),
        'windows': window_results,
        'positive_pct': sum(1 for r in window_results if r > 0) / len(window_results),
        'n_windows': len(window_results),
    }


# ============================================================================
# MONTE CARLO — Module 5
# ============================================================================

def monte_carlo_test(env_class, data, env_params, model_obs_size, n_assets, real_return, n_perms=100):
    logger.info(f"\n{'='*60}")
    logger.info(f"  MONTE CARLO PERMUTATION TEST ({n_perms} permutations)")
    logger.info(f"{'='*60}")

    ib = env_params.get('initial_balance', INITIAL_BALANCE)
    random_returns = []

    for i in range(n_perms):
        try:
            env = create_env_with_check(env_class, data, env_params, model_obs_size)
            obs, _ = env.reset()
            done = truncated = False
            while not (done or truncated):
                action = np.random.choice([0, 1, 2], size=n_assets, p=[0.89, 0.055, 0.055])
                obs, _, done, truncated, info = env.step(action)
            fv = info.get('equity', info.get('portfolio_value', ib))
            random_returns.append((fv - ib) / ib)
            if (i + 1) % 25 == 0:
                logger.info(f"  {i+1}/{n_perms}...")
        except Exception:
            pass

    if not random_returns:
        return {'p_value': 1.0, 'percentile': 0, 'beats_random': False}

    arr = np.array(random_returns)
    p_value = float(np.mean(arr >= real_return))
    percentile = float(np.mean(real_return > arr) * 100)

    logger.info(f"  Return modele: {real_return:+.2%}")
    logger.info(f"  Return random: {np.mean(arr):+.2%} (mean)")
    logger.info(f"  Percentile:    {percentile:.0f}e | p-value: {p_value:.4f}")

    return {
        'p_value': p_value, 'percentile': percentile,
        'beats_random': p_value < 0.05,
        'random_mean': float(np.mean(arr)), 'random_std': float(np.std(arr)),
    }


# ============================================================================
# STRESS TEST — Module 6 (Enhanced)
# ============================================================================

def _apply_crash_to_data(data, crash_pct=-0.10, crash_point=0.5):
    """Simule un crash soudain dans les donnees."""
    stressed = {}
    for ticker, df in data.items():
        df2 = df.copy()
        crash_idx = int(len(df2) * crash_point)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df2.columns:
                df2.loc[df2.index[crash_idx:], col] *= (1 + crash_pct)
        stressed[ticker] = df2
    return stressed


def _apply_low_volatility(data, vol_factor=0.3):
    """Reduit la volatilite pour simuler un marche range-bound."""
    stressed = {}
    for ticker, df in data.items():
        df2 = df.copy()
        mean_close = df2['Close'].mean()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df2.columns:
                df2[col] = mean_close + (df2[col] - mean_close) * vol_factor
        stressed[ticker] = df2
    return stressed


def _apply_gaps(data, n_gaps=5, gap_pct=0.03):
    """Ajoute des gaps overnight aleatoires."""
    stressed = {}
    for ticker, df in data.items():
        df2 = df.copy()
        gap_indices = np.random.choice(range(50, len(df2) - 10), size=min(n_gaps, max(1, len(df2) // 50)), replace=False)
        for idx in gap_indices:
            direction = np.random.choice([-1, 1])
            factor = 1 + direction * gap_pct
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df2.columns:
                    df2.loc[df2.index[idx:], col] *= factor
        stressed[ticker] = df2
    return stressed


def stress_test(engine, data, base_params):
    logger.info(f"\n{'='*60}")
    logger.info(f"  STRESS TEST")
    logger.info(f"{'='*60}")

    # --- Cost/spread scenarios ---
    cost_scenarios = [
        ('Normal', {}),
        ('Couts 2x', {'sec_fee': base_params.get('sec_fee', 0) * 2, 'finra_taf': base_params.get('finra_taf', 0) * 2}),
        ('Couts 5x', {'sec_fee': base_params.get('sec_fee', 0) * 5, 'finra_taf': base_params.get('finra_taf', 0) * 5}),
        ('Spread 2x', {'spread_bps': base_params.get('spread_bps', 2) * 2}),
        ('Spread 5x', {'spread_bps': base_params.get('spread_bps', 2) * 5}),
    ]

    results = {}
    for name, overrides in cost_scenarios:
        params = {**base_params, **overrides}
        stress_engine = BacktestEngine(engine.model, engine.tickers, engine.env_class, params, engine.model_obs_size)
        try:
            r = stress_engine.run_episode(data, track_trades=False)
            results[name] = r['total_return']
            status = "OK" if r['total_return'] > 0 else "FAIL"
            logger.info(f"  {name:15s}: {r['total_return']:+.2%}  [{status}]")
        except Exception as e:
            results[name] = None
            logger.info(f"  {name:15s}: ERREUR ({e})")

    # --- Market condition scenarios ---
    logger.info(f"\n  Stress Scenarios Marche:")
    market_scenarios = [
        ('Crash -10%', lambda d: _apply_crash_to_data(d, -0.10)),
        ('Crash -20%', lambda d: _apply_crash_to_data(d, -0.20)),
        ('Range-bound', lambda d: _apply_low_volatility(d, 0.3)),
        ('Gaps overnight', lambda d: _apply_gaps(d, n_gaps=5)),
    ]

    for name, transform_fn in market_scenarios:
        try:
            stressed_data = transform_fn(data)
            r = engine.run_episode(stressed_data, track_trades=False)
            results[name] = r['total_return']
            status = "OK" if r['total_return'] > -0.15 else "DANGER"
            logger.info(f"  {name:15s}: {r['total_return']:+.2%}  [{status}]")
        except Exception as e:
            results[name] = None
            logger.info(f"  {name:15s}: ERREUR ({e})")

    return results


# ============================================================================
# OOS VALIDATION — Module 8
# ============================================================================

def check_oos_validity(metadata, backtest_start_date, backtest_end_date):
    """Verifie si les donnees de backtest chevauchent l'entrainement.

    Returns:
        dict with 'is_oos', 'overlap_pct', 'training_end', 'warnings'
    """
    result = {
        'is_oos': True,
        'overlap_pct': 0.0,
        'training_data_end': None,
        'backtest_start': str(backtest_start_date),
        'backtest_end': str(backtest_end_date),
        'warnings': [],
    }

    if not metadata:
        result['warnings'].append("Pas de metadata — impossible de verifier OOS")
        result['is_oos'] = False
        return result

    training_end_str = metadata.get('training_data_end')
    if not training_end_str:
        result['warnings'].append("Metadata sans 'training_data_end' — impossible de verifier OOS")
        result['is_oos'] = False
        return result

    try:
        # Parse training end date (handle multiple formats)
        training_end_str_clean = training_end_str.split('+')[0].split('T')[0] if 'T' in training_end_str else training_end_str[:10]
        training_end = datetime.strptime(training_end_str_clean, '%Y-%m-%d')
    except (ValueError, TypeError):
        try:
            training_end = pd.to_datetime(training_end_str).to_pydatetime().replace(tzinfo=None)
        except Exception:
            result['warnings'].append(f"Format date non reconnu: {training_end_str}")
            result['is_oos'] = False
            return result

    result['training_data_end'] = training_end.strftime('%Y-%m-%d')

    bt_start = backtest_start_date
    if hasattr(bt_start, 'to_pydatetime'):
        bt_start = bt_start.to_pydatetime()
    if bt_start.tzinfo:
        bt_start = bt_start.replace(tzinfo=None)

    bt_end = backtest_end_date
    if hasattr(bt_end, 'to_pydatetime'):
        bt_end = bt_end.to_pydatetime()
    if bt_end.tzinfo:
        bt_end = bt_end.replace(tzinfo=None)

    if bt_start >= training_end:
        result['is_oos'] = True
        result['overlap_pct'] = 0.0
        logger.info(f"  ✅ OOS VALIDE: backtest ({bt_start.date()}) commence apres training ({training_end.date()})")
    else:
        total_bt_days = (bt_end - bt_start).days
        overlap_days = (training_end - bt_start).days if training_end > bt_start else 0
        overlap_pct = min(100.0, (overlap_days / max(1, total_bt_days)) * 100)

        result['is_oos'] = False
        result['overlap_pct'] = overlap_pct

        if overlap_pct >= 100:
            result['warnings'].append(
                f"⛔ ALERTE CRITIQUE: 100% des donnees de backtest etaient dans le training! "
                f"Training: ->  {training_end.date()} | Backtest: {bt_start.date()} -> {bt_end.date()}"
            )
        else:
            result['warnings'].append(
                f"⚠️  CHEVAUCHEMENT: {overlap_pct:.0f}% des donnees backtest dans le training. "
                f"Training: -> {training_end.date()} | Backtest: {bt_start.date()} -> {bt_end.date()}"
            )

    return result


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS — Module 9
# ============================================================================

def bootstrap_confidence_intervals(all_histories, initial_balance, n_bootstrap=1000, ci_level=0.95):
    """Calcule des intervalles de confiance bootstrap sur les metriques cles.

    Args:
        all_histories: liste de portfolio histories (arrays)
        initial_balance: capital initial
        n_bootstrap: nombre de reechantillonnages
        ci_level: niveau de confiance (0.95 = 95%)

    Returns:
        dict de {metric_name: {'mean': ..., 'ci_low': ..., 'ci_high': ..., 'std': ...}}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap} resamples, {ci_level:.0%} CI)")
    logger.info(f"{'='*60}")

    if len(all_histories) < 2:
        logger.warning("  Pas assez d'episodes pour bootstrap (min 2)")
        return {}

    alpha = (1 - ci_level) / 2

    # Bootstrap resamples
    boot_returns = []
    boot_sharpes = []
    boot_drawdowns = []
    boot_sortinos = []

    n_episodes = len(all_histories)

    for _ in range(n_bootstrap):
        # Resample episodes with replacement
        indices = np.random.choice(n_episodes, size=n_episodes, replace=True)
        sampled_history = all_histories[indices[0]]  # use first sampled

        m = compute_portfolio_metrics(sampled_history, initial_balance)
        boot_returns.append(m['total_return'])
        boot_sharpes.append(m['sharpe_ratio'])
        boot_drawdowns.append(m['max_drawdown'])
        boot_sortinos.append(m['sortino_ratio'])

    results = {}
    for name, values in [
        ('total_return', boot_returns),
        ('sharpe_ratio', boot_sharpes),
        ('max_drawdown', boot_drawdowns),
        ('sortino_ratio', boot_sortinos),
    ]:
        arr = np.array(values)
        results[name] = {
            'mean': float(np.mean(arr)),
            'ci_low': float(np.percentile(arr, alpha * 100)),
            'ci_high': float(np.percentile(arr, (1 - alpha) * 100)),
            'std': float(np.std(arr)),
        }

    for name, r in results.items():
        if 'return' in name or 'drawdown' in name:
            logger.info(f"  {name:20s}: {r['mean']:+.2%}  [{r['ci_low']:+.2%}, {r['ci_high']:+.2%}]")
        else:
            logger.info(f"  {name:20s}: {r['mean']:.2f}  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")

    return results


# ============================================================================
# MARKET REGIME ANALYSIS — Module 10
# ============================================================================

def detect_market_regime(data, window_bars=100):
    """Classifie les fenetres de donnees en regimes bull/bear/range.

    Returns:
        list of (start, end, regime) tuples
    """
    # Use first available ticker as market proxy
    first_ticker = list(data.keys())[0]
    df = data[first_ticker]
    closes = df['Close'].values

    regimes = []
    i = 0
    while i + window_bars <= len(closes):
        segment = closes[i:i + window_bars]
        ret = (segment[-1] - segment[0]) / segment[0] if segment[0] > 0 else 0
        volatility = np.std(np.diff(segment) / segment[:-1]) if len(segment) > 1 else 0

        if ret > 0.05:
            regime = 'BULL'
        elif ret < -0.05:
            regime = 'BEAR'
        else:
            regime = 'RANGE'

        regimes.append((i, i + window_bars, regime, ret, volatility))
        i += window_bars // 2  # 50% overlap

    return regimes


def market_regime_analysis(engine, data, metadata=None):
    """Analyse la performance du modele par regime de marche.

    Returns:
        dict with per-regime stats
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  MARKET REGIME ANALYSIS")
    logger.info(f"{'='*60}")

    regimes = detect_market_regime(data)

    if not regimes:
        logger.warning("  Pas assez de donnees pour analyser les regimes")
        return {'regimes': [], 'per_regime': {}}

    regime_counts = {}
    for _, _, regime, _, _ in regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    logger.info(f"  Regimes detectes: {regime_counts}")

    # Test per regime
    per_regime = {}
    for regime_type in ['BULL', 'BEAR', 'RANGE']:
        regime_windows = [(s, e) for s, e, r, _, _ in regimes if r == regime_type]
        if not regime_windows:
            continue

        returns = []
        for start, end in regime_windows[:5]:  # Max 5 windows per regime
            wdata = {t: df.iloc[start:end].reset_index(drop=True) for t, df in data.items() if end <= len(df)}
            if len(wdata) < len(data) // 2:
                continue
            try:
                result = engine.run_episode(wdata, track_trades=False)
                returns.append(result['total_return'])
            except Exception:
                pass

        if returns:
            per_regime[regime_type] = {
                'mean_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)) if len(returns) > 1 else 0.0,
                'n_windows': len(returns),
                'positive_pct': sum(1 for r in returns if r > 0) / len(returns),
            }
            logger.info(f"  {regime_type:6s}: return={np.mean(returns):+.2%}  "
                         f"(n={len(returns)}, {per_regime[regime_type]['positive_pct']:.0%} positif)")

    return {'regimes': [(s, e, r) for s, e, r, _, _ in regimes], 'per_regime': per_regime}


# ============================================================================
# CERTIFICATION — Module 7
# ============================================================================

def generate_certification(metrics, journal_metrics, walk_forward, monte_carlo, stress_results, benchmarks, thresholds):
    checks = {}

    def _check(name, value, threshold_key, compare, weight):
        thr = thresholds[threshold_key]
        if compare == '>':
            passed = value > thr
            thr_str = f"> {thr}"
        else:
            passed = value < thr
            thr_str = f"< {thr}"
        checks[name] = {'value': value, 'threshold': thr_str, 'pass': passed, 'weight': weight}

    _check('return',              metrics['total_return'],                     'return_min',              '>', 15)
    _check('sharpe',              metrics['sharpe_ratio'],                     'sharpe_min',              '>', 12)
    _check('sortino',             metrics['sortino_ratio'],                    'sortino_min',             '>', 8)
    _check('max_drawdown',        metrics['max_drawdown'],                     'max_drawdown_max',        '<', 12)
    _check('calmar',              metrics['calmar_ratio'],                     'calmar_min',              '>', 5)
    _check('win_rate',            journal_metrics.get('win_rate', 0),          'win_rate_min',            '>', 10)
    _check('profit_factor',       journal_metrics.get('profit_factor', 0),     'profit_factor_min',       '>', 10)
    _check('buy_quality',         journal_metrics.get('buy_quality', 0),       'buy_quality_min',         '>', 15)
    _check('avg_trade_duration',  journal_metrics.get('avg_trade_duration', 0),'avg_trade_duration_min',  '>', 3)
    _check('monte_carlo',         monte_carlo.get('p_value', 1.0),             'monte_carlo_p_max',       '<', 10)

    total_weight = sum(c['weight'] for c in checks.values())
    earned = sum(c['weight'] for c in checks.values() if c['pass'])
    score = int(100 * earned / total_weight) if total_weight > 0 else 0

    n_pass = sum(1 for c in checks.values() if c['pass'])
    n_total = len(checks)
    global_pass = score >= 60 and checks['return']['pass'] and checks['max_drawdown']['pass']

    return {
        'checks': checks, 'score': score,
        'n_pass': n_pass, 'n_total': n_total,
        'global_pass': global_pass, 'verdict': 'PASS' if global_pass else 'FAIL',
    }


# ============================================================================
# DISPLAY
# ============================================================================

def print_full_report(cert, metrics, jm, wf, mc, stress, benchmarks, model_path, env_version, elapsed,
                      oos_result=None, bootstrap_ci=None, regime_results=None):
    W = 70
    print(f"\n{'='*W}")
    print(f"  RAPPORT DE CERTIFICATION — BACKTEST ULTIMATE")
    print(f"{'='*W}")
    print(f"  Modele    : {model_path}")
    print(f"  Env       : {env_version}")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Duree     : {elapsed:.0f}s")
    print(f"{'='*W}")

    # OOS Validation (show prominently at top if issues)
    if oos_result:
        print(f"\n  [OOS VALIDATION]")
        if oos_result.get('is_oos'):
            print(f"  {'Statut':<25s}: {'✅ OUT-OF-SAMPLE VALIDE':>30s}")
        else:
            overlap = oos_result.get('overlap_pct', 0)
            print(f"  {'Statut':<25s}: {'⛔ IN-SAMPLE (BIAISE)':>30s}")
            print(f"  {'Chevauchement':<25s}: {overlap:>10.0f}%")
        if oos_result.get('training_data_end'):
            print(f"  {'Fin donnees training':<25s}: {oos_result['training_data_end']:>30s}")
        print(f"  {'Debut backtest':<25s}: {oos_result.get('backtest_start', '?'):>30s}")
        for w in oos_result.get('warnings', []):
            print(f"  {w}")

    print(f"\n  [PERFORMANCE]")
    for label, key, fmt in [
        ('Return total', 'total_return', '+.2%'), ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
        ('Sortino Ratio', 'sortino_ratio', '.2f'), ('Max Drawdown', 'max_drawdown', '.2%'),
        ('Calmar Ratio', 'calmar_ratio', '.2f'), ('Volatilite ann.', 'volatility', '.2%'),
        ('Return annualise', 'annualized_return', '+.2%'),
    ]:
        val = metrics.get(key, 0)
        val_str = f"{val:{fmt}}"
        # Add CI if available
        if bootstrap_ci and key in bootstrap_ci:
            ci = bootstrap_ci[key]
            if 'return' in key or 'drawdown' in key:
                ci_str = f"  CI95: [{ci['ci_low']:+.2%}, {ci['ci_high']:+.2%}]"
            else:
                ci_str = f"  CI95: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
            print(f"  {label:<25s}: {val_str:>12s}{ci_str}")
        else:
            print(f"  {label:<25s}: {val_str:>12s}")

    print(f"\n  [TRADE JOURNAL]")
    print(f"  {'Trades completes':<25s}: {int(jm.get('total_completed_trades', 0)):>10d}")
    print(f"  {'Win Rate':<25s}: {jm.get('win_rate', 0):>10.1%}")
    print(f"  {'Profit Factor':<25s}: {jm.get('profit_factor', 0):>10.2f}")
    print(f"  {'BUY Quality (buy low)':<25s}: {jm.get('buy_quality', 0):>10.1%}")
    print(f"  {'SELL Quality (sell high)':<25s}: {jm.get('sell_quality', 0):>10.1%}")
    print(f"  {'Duree moy. trades':<25s}: {jm.get('avg_trade_duration', 0):>10.1f} steps")
    print(f"  {'Meilleur trade':<25s}: {jm.get('best_trade', 0):>+10.2%}")
    print(f"  {'Pire trade':<25s}: {jm.get('worst_trade', 0):>+10.2%}")

    print(f"\n  [BENCHMARKS]")
    bot = metrics['total_return']
    for name, ret in benchmarks.items():
        diff = bot - ret
        m = "+" if diff > 0 else "-"
        print(f"  {'vs ' + name:<25s}: {ret:>+10.2%}  ({m}{abs(diff):.2%})")

    print(f"\n  [WALK-FORWARD]")
    print(f"  {'Fenetres testees':<25s}: {wf.get('n_windows', 0):>10d}")
    print(f"  {'Return moyen':<25s}: {wf.get('mean_return', 0):>+10.2%}")
    print(f"  {'Ecart-type returns':<25s}: {wf.get('std_return', 0):>10.2%}")
    print(f"  {'% fenetres positives':<25s}: {wf.get('positive_pct', 0):>10.0%}")

    print(f"\n  [MONTE CARLO]")
    print(f"  {'p-value':<25s}: {mc.get('p_value', 1.0):>10.4f}")
    print(f"  {'Percentile':<25s}: {mc.get('percentile', 0):>10.0f}e")
    beats = "OUI" if mc.get('beats_random', False) else "NON"
    print(f"  {'Bat le hasard (p<0.05)':<25s}: {beats:>10s}")

    print(f"\n  [STRESS TEST]")
    for name, ret in stress.items():
        if ret is not None:
            if name.startswith('Crash') or name.startswith('Range') or name.startswith('Gaps'):
                status = 'OK' if ret > -0.15 else 'DANGER'
            else:
                status = 'OK' if ret > 0 else 'FAIL'
            print(f"  {name:<25s}: {ret:>+10.2%}  [{status}]")
        else:
            print(f"  {name:<25s}: {'ERREUR':>10s}")

    # Market regime analysis
    if regime_results and regime_results.get('per_regime'):
        print(f"\n  [MARKET REGIME]")
        for regime, stats in regime_results['per_regime'].items():
            emoji = {'BULL': '📈', 'BEAR': '📉', 'RANGE': '➡️'}.get(regime, '')
            print(f"  {emoji} {regime:<22s}: {stats['mean_return']:>+10.2%}  "
                  f"({stats['n_windows']} fenetres, {stats['positive_pct']:.0%} positif)")

    print(f"\n{'='*W}")
    print(f"  CERTIFICATION — CRITERES PASS/FAIL")
    print(f"{'='*W}")
    for name, c in cert['checks'].items():
        icon = "PASS" if c['pass'] else "FAIL"
        val = c['value']
        if isinstance(val, float):
            if abs(val) < 1 and name not in ('sharpe', 'sortino', 'calmar', 'profit_factor', 'avg_trade_duration', 'monte_carlo'):
                vs = f"{val:+.2%}"
            else:
                vs = f"{val:.3f}"
        else:
            vs = str(val)
        print(f"  [{icon:4s}] {name:<25s}  {vs:>10s}  (seuil: {c['threshold']})")

    # OOS penalty in verdict
    oos_penalty = ""
    if oos_result and not oos_result.get('is_oos'):
        oos_penalty = "  ⚠️  ATTENTION: Resultats IN-SAMPLE — non fiables pour production!"

    print(f"\n{'='*W}")
    print(f"  VERDICT FINAL:  {cert['verdict']}  |  Score: {cert['score']}/100  |  {cert['n_pass']}/{cert['n_total']} criteres")
    if oos_penalty:
        print(oos_penalty)
    print(f"{'='*W}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest Ultimate — Validation Complete')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modele (.zip)')
    parser.add_argument('--days', type=int, default=90, help='Periode de backtest (defaut: 90)')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes pour le run principal (defaut: 5)')
    parser.add_argument('--quick', action='store_true', help='Mode rapide')
    parser.add_argument('--output', type=str, default=None, help='Chemin du rapport JSON')
    parser.add_argument('--config', type=str, default=None, help='Chemin config YAML (override auto-detection)')
    parser.add_argument('--oos-only', action='store_true',
                        help='Mode OOS strict: ne telecharge que les donnees post-training')
    parser.add_argument('--force', action='store_true',
                        help='Force le backtest meme si les donnees chevauchent le training')
    args = parser.parse_args()

    start_time = time.time()

    # ── 0. LOAD MODEL ──
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Modele introuvable: {model_path}")
        sys.exit(1)

    logger.info(f"Chargement modele: {model_path.name}")
    model = PPO.load(model_path)
    model_obs_size = model.observation_space.shape[0]

    # ── 0b. LOAD V7 METADATA (if exists) ──
    metadata, model_config, vecnorm_path = load_v7_metadata(model_path)

    # Override config from CLI if specified
    if args.config:
        yaml_config = load_yaml_config(args.config)
        if yaml_config:
            model_config = yaml_config
            logger.info(f"  Config YAML chargee: {args.config}")

    # ── 1. DETECT ENVIRONMENT ──
    env_version, n_tickers, meta_tickers, env_class, env_params = detect_environment(
        model, metadata=metadata, config=model_config
    )

    # Resolve tickers: metadata > config > fallback
    if meta_tickers:
        tickers = meta_tickers
    elif model_config and model_config.get('data', {}).get('tickers'):
        tickers = model_config['data']['tickers']
    elif n_tickers and n_tickers <= 10:
        tickers = TICKERS_10[:n_tickers]
    elif n_tickers:
        tickers = TICKERS_15[:min(n_tickers, 15)]
    else:
        tickers = TICKERS_15

    use_vecnorm = vecnorm_path is not None or (env_version == 'V7')
    logger.info(f"  Environnement: {env_version} | Tickers: {len(tickers)} | VecNormalize: {use_vecnorm}")

    # ── 2. FETCH DATA (with OOS support) ──
    end_date = datetime.now()

    # OOS-only mode: start from training end date
    if args.oos_only and metadata and metadata.get('training_data_end'):
        training_end_str = metadata['training_data_end']
        try:
            training_end = pd.to_datetime(training_end_str)
            start_date = training_end + timedelta(days=1)
            logger.info(f"\n⚡ MODE OOS-ONLY: donnees uniquement apres {start_date.strftime('%Y-%m-%d')}")
            logger.info(f"  (training_data_end = {training_end_str})")
        except Exception:
            logger.warning(f"  Format date non reconnu pour OOS: {training_end_str}")
            start_date = end_date - timedelta(days=args.days + 120)
    else:
        start_date = end_date - timedelta(days=args.days + 120)

    logger.info(f"\nTelechargement donnees ({start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}, {len(tickers)} tickers)...")
    fetcher = UniversalDataFetcher()

    data = {}
    for ticker in tickers:
        try:
            df = fetcher.fetch(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval='1h')
            if df is not None and len(df) > 100:
                data[ticker] = df
                logger.info(f"  {ticker}: {len(df)} barres")
        except Exception as e:
            logger.warning(f"  {ticker}: ERREUR ({e})")

    if len(data) < max(1, len(tickers) // 2):
        logger.error("Pas assez de donnees. Abandon.")
        sys.exit(1)

    logger.info(f"{len(data)}/{len(tickers)} tickers charges")

    # Adjust tickers to actually loaded ones
    actual_tickers = list(data.keys())
    n_assets = len(actual_tickers)

    # ── 2b. OOS VALIDATION ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  OOS VALIDATION")
    logger.info(f"{'='*60}")

    # Get actual date range of backtest data
    all_bt_starts = []
    all_bt_ends = []
    for ticker, df in data.items():
        if len(df) > 0:
            all_bt_starts.append(df.index[0])
            all_bt_ends.append(df.index[-1])

    bt_start = min(all_bt_starts) if all_bt_starts else start_date
    bt_end = max(all_bt_ends) if all_bt_ends else end_date

    oos_result = check_oos_validity(metadata, bt_start, bt_end)

    if not oos_result['is_oos'] and not args.force:
        for w in oos_result.get('warnings', []):
            logger.warning(w)
        if oos_result.get('overlap_pct', 0) >= 100:
            logger.error("⛔ 100% in-sample! Utilisez --oos-only pour un vrai test OOS, ou --force pour continuer quand meme.")
            if not args.force:
                logger.error("Ajoutez --force pour continuer malgre le chevauchement.")
                sys.exit(1)

    # ── 3. VERIFY ENV COMPATIBILITY ──
    logger.info(f"\nVerification compatibilite modele/environnement...")
    try:
        test_env = create_env_with_check(env_class, data, env_params, model_obs_size)
        logger.info(f"  MATCH! Env produit {model_obs_size} dims")
        del test_env
    except ValueError as e:
        logger.error(f"  {e}")
        logger.error(f"  Ce modele n'est pas compatible avec l'environnement {env_version} disponible.")
        logger.error(f"  Verifiez que le bon environnement est installe ou utilisez un modele V6/V7.")
        sys.exit(1)

    # ── 4. MAIN BACKTEST ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKTEST PRINCIPAL ({args.episodes} episodes)")
    logger.info(f"{'='*60}")

    engine = BacktestEngine(model, actual_tickers, env_class, env_params, model_obs_size)
    all_histories = []
    all_journals = []
    first_return = 0.0

    for ep in range(args.episodes):
        result = engine.run_episode(data, track_trades=True, record_actions=(ep == 0))
        all_histories.append(result['portfolio_history'])
        if result['journal']:
            all_journals.append(result['journal'])
        logger.info(f"  Ep {ep+1}/{args.episodes}: return={result['total_return']:+.2%}, "
                     f"trades={result['total_trades']}, steps={result['steps']}")
        if ep == 0:
            first_return = result['total_return']

    best_idx = int(np.argmax([h[-1] for h in all_histories]))
    main_history = all_histories[best_idx]
    main_metrics = compute_portfolio_metrics(main_history, INITIAL_BALANCE)

    all_trades = []
    for j in all_journals:
        all_trades.extend(j.trades)
    agg_journal = TradeJournal(actual_tickers)
    agg_journal.trades = all_trades
    journal_metrics = agg_journal.get_metrics()

    # ── 5. BENCHMARKS ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  BENCHMARKS")
    logger.info(f"{'='*60}")

    bh = benchmark_buy_and_hold(data, INITIAL_BALANCE)
    logger.info(f"  Buy & Hold:    {bh:+.2%}")

    rnd = benchmark_random_agent(env_class, data, env_params, model_obs_size, n_assets, n_runs=(3 if args.quick else 5))
    logger.info(f"  Random Agent:  {rnd:+.2%}")
    logger.info(f"  Hold Cash:     +0.00%")

    benchmarks = {'Buy & Hold': bh, 'Random Agent': rnd, 'Hold Cash': 0.0}

    # ── 6. WALK-FORWARD ──
    wf_params = dict(
        n_windows=(3 if args.quick else 6), window_days=30,
        shift_days=15, episodes_per_window=(2 if args.quick else 3),
    )
    wf = walk_forward_validation(engine, data, **wf_params)

    # ── 7. MONTE CARLO ──
    if args.quick:
        logger.info(f"\n  [SKIP] Monte Carlo (mode --quick)")
        mc = {'p_value': 0.0, 'percentile': 100, 'beats_random': True}
    else:
        mc = monte_carlo_test(env_class, data, env_params, model_obs_size, n_assets, first_return, n_perms=100)

    # ── 8. STRESS TEST ──
    stress = stress_test(engine, data, env_params)

    # ── 9. BOOTSTRAP CONFIDENCE INTERVALS ──
    if args.quick:
        logger.info(f"\n  [SKIP] Bootstrap CI (mode --quick)")
        bootstrap_ci = {}
    else:
        bootstrap_ci = bootstrap_confidence_intervals(all_histories, INITIAL_BALANCE, n_bootstrap=1000)

    # ── 10. MARKET REGIME ANALYSIS ──
    if args.quick:
        logger.info(f"\n  [SKIP] Market Regime (mode --quick)")
        regime_results = {}
    else:
        regime_results = market_regime_analysis(engine, data, metadata)

    # ── 11. CERTIFICATION ──
    cert = generate_certification(main_metrics, journal_metrics, wf, mc, stress, benchmarks, THRESHOLDS)

    elapsed = time.time() - start_time

    # ── DISPLAY ──
    print_full_report(cert, main_metrics, journal_metrics, wf, mc, stress, benchmarks,
                      model_path.name, env_version, elapsed,
                      oos_result=oos_result, bootstrap_ci=bootstrap_ci, regime_results=regime_results)

    # ── EXPORT JSON ──
    report_dir = Path('logs/backtest_reports')
    report_dir.mkdir(parents=True, exist_ok=True)

    report_name = args.output or f"{model_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = report_dir / report_name

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    def _safe(v):
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, (bool,)): return v
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    def _deep_safe(obj):
        if isinstance(obj, dict):
            return {k: _deep_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_deep_safe(v) for v in obj]
        return _safe(obj)

    report_data = _deep_safe({
        'model': str(model_path),
        'env_version': env_version,
        'date': datetime.now().isoformat(),
        'duration_seconds': elapsed,
        'config': {'days': args.days, 'episodes': args.episodes, 'quick_mode': args.quick,
                   'oos_only': args.oos_only, 'tickers': actual_tickers},
        'oos_validation': {
            'is_oos': oos_result.get('is_oos', False),
            'overlap_pct': oos_result.get('overlap_pct', 0),
            'training_data_end': oos_result.get('training_data_end'),
            'warnings': oos_result.get('warnings', []),
        },
        'metrics': dict(main_metrics),
        'bootstrap_ci': bootstrap_ci if bootstrap_ci else None,
        'journal': dict(journal_metrics),
        'benchmarks': dict(benchmarks),
        'walk_forward': {k: v for k, v in wf.items() if k != 'windows'},
        'monte_carlo': dict(mc),
        'stress_test': dict(stress),
        'market_regime': {k: v for k, v in (regime_results.get('per_regime', {}) if regime_results else {}).items()},
        'certification': {
            'score': cert['score'], 'verdict': cert['verdict'],
            'n_pass': cert['n_pass'], 'n_total': cert['n_total'],
            'checks': {n: {'value': c['value'], 'threshold': c['threshold'], 'pass': c['pass']}
                       for n, c in cert['checks'].items()},
        },
    })

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"\nRapport JSON: {report_path}")
    logger.info(f"Duree totale: {elapsed:.0f}s")

    sys.exit(0 if cert['global_pass'] else 1)


if __name__ == '__main__':
    main()

