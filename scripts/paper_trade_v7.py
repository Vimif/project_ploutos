#!/usr/bin/env python3
"""
===============================================================================
  PAPER TRADER V7 — Simulation Live des Modeles Ploutos
===============================================================================

Pipeline de paper trading pour valider un modele V7 en conditions reelles.
Supporte deux modes:
  - Alpaca Paper Trading (avec cles API)
  - Simulation locale (sans API, yfinance only)

Fonctionnalites:
  - Chargement V7 model + VecNormalize + metadata
  - Boucle de trading live avec interval configurable
  - Kill switch automatique (drawdown, inactivite, perte max)
  - Journal des trades en temps reel
  - Export equity curve + rapport JSON
  - Comparaison avec performance OOS attendue

Usage:
  python scripts/paper_trade_v7.py --model models/v7_sp500/ploutos_v7.zip --mode simulate
  python scripts/paper_trade_v7.py --model models/v7_sp500/ploutos_v7.zip --mode alpaca \\
      --api-key YOUR_KEY --api-secret YOUR_SECRET
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
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import PPO

logger = logging.getLogger('PaperTrader')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_INITIAL_BALANCE = 100_000
KILL_SWITCH_MAX_DRAWDOWN = 0.15    # 15% drawdown -> stop
KILL_SWITCH_MAX_DAILY_LOSS = 0.05  # 5% daily loss -> stop
KILL_SWITCH_INACTIVITY_HOURS = 4   # No trades in 4h -> alert
TRADE_INTERVAL_MINUTES = 60        # Check every hour


# ============================================================================
# KILL SWITCH
# ============================================================================

class KillSwitch:
    """Monitore les conditions de stop automatique."""

    def __init__(self, initial_balance, max_drawdown=KILL_SWITCH_MAX_DRAWDOWN,
                 max_daily_loss=KILL_SWITCH_MAX_DAILY_LOSS,
                 inactivity_hours=KILL_SWITCH_INACTIVITY_HOURS):
        self.initial_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.inactivity_hours = inactivity_hours

        self.peak_equity = initial_balance
        self.daily_start_equity = initial_balance
        self.last_trade_time = datetime.now()
        self.last_daily_reset = datetime.now().date()
        self.triggered = False
        self.trigger_reason = None
        self.alerts = []

    def check(self, current_equity):
        """Verifie les conditions de kill switch. Returns (is_triggered, reason)."""
        now = datetime.now()

        # Reset daily tracking
        if now.date() != self.last_daily_reset:
            self.daily_start_equity = current_equity
            self.last_daily_reset = now.date()

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check max drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= self.max_drawdown:
            self.triggered = True
            self.trigger_reason = f"MAX DRAWDOWN: {drawdown:.1%} >= {self.max_drawdown:.1%}"
            return True, self.trigger_reason

        # Check daily loss
        daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if daily_loss >= self.max_daily_loss:
            self.triggered = True
            self.trigger_reason = f"MAX DAILY LOSS: {daily_loss:.1%} >= {self.max_daily_loss:.1%}"
            return True, self.trigger_reason

        # Check inactivity (alert only, no kill)
        hours_since_trade = (now - self.last_trade_time).total_seconds() / 3600
        if hours_since_trade >= self.inactivity_hours:
            alert = f"INACTIVITY: {hours_since_trade:.1f}h since last trade"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning(f"  ⚠️  {alert}")

        return False, None

    def record_trade(self):
        """Enregistre un trade pour le tracking d'inactivite."""
        self.last_trade_time = datetime.now()


# ============================================================================
# TRADE JOURNAL (LIVE)
# ============================================================================

class LiveTradeJournal:
    """Journal des trades en temps reel."""

    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.start_time = datetime.now()

    def record_trade(self, ticker, side, price, qty, total_value):
        """Enregistre un trade."""
        self.trades.append({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'side': side,
            'price': price,
            'qty': qty,
            'total_value': total_value,
        })
        logger.info(f"  📊 TRADE: {side} {qty:.2f}x {ticker} @ ${price:.2f} = ${total_value:.2f}")

    def record_equity(self, equity):
        """Enregistre un point equity."""
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
        })

    def get_summary(self, initial_balance):
        """Genere un resume."""
        if not self.equity_curve:
            return {'n_trades': 0, 'return': 0.0}

        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - initial_balance) / initial_balance
        buys = sum(1 for t in self.trades if t['side'] == 'BUY')
        sells = sum(1 for t in self.trades if t['side'] == 'SELL')

        return {
            'n_trades': len(self.trades),
            'n_buys': buys,
            'n_sells': sells,
            'total_return': total_return,
            'final_equity': final_equity,
            'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
        }

    def export(self, path):
        """Exporte le journal en JSON."""
        data = {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'start_time': self.start_time.isoformat(),
            'export_time': datetime.now().isoformat(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Journal exporte: {path}")


# ============================================================================
# SIMULATED BROKER (Local Mode)
# ============================================================================

class SimulatedBroker:
    """Broker simule pour paper trading sans API externe."""

    def __init__(self, initial_balance):
        self.cash = initial_balance
        self.positions = {}   # {ticker: {'qty': float, 'avg_price': float}}
        self.initial_balance = initial_balance

    def get_equity(self, prices):
        """Calcule l'equity totale."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos['avg_price'])
            equity += pos['qty'] * price
        return equity

    def buy(self, ticker, price, amount_usd):
        """Achete pour un montant en USD."""
        if amount_usd > self.cash:
            amount_usd = self.cash * 0.95

        if amount_usd < 10:
            return 0

        qty = amount_usd / price
        cost = qty * price

        self.cash -= cost

        if ticker in self.positions:
            old = self.positions[ticker]
            new_qty = old['qty'] + qty
            self.positions[ticker] = {
                'qty': new_qty,
                'avg_price': (old['avg_price'] * old['qty'] + price * qty) / new_qty,
            }
        else:
            self.positions[ticker] = {'qty': qty, 'avg_price': price}

        return qty

    def sell(self, ticker, price, qty=None):
        """Vend une position (tout ou partie)."""
        if ticker not in self.positions:
            return 0

        pos = self.positions[ticker]
        sell_qty = qty or pos['qty']
        sell_qty = min(sell_qty, pos['qty'])

        proceeds = sell_qty * price
        self.cash += proceeds

        remaining = pos['qty'] - sell_qty
        if remaining < 0.001:
            del self.positions[ticker]
        else:
            self.positions[ticker]['qty'] = remaining

        return sell_qty

    def get_positions(self):
        return dict(self.positions)


# ============================================================================
# ALPACA BROKER (Paper Trading Mode)
# ============================================================================

class AlpacaBroker:
    """Interface Alpaca paper trading."""

    def __init__(self, api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            account = self.api.get_account()
            self.initial_balance = float(account.equity)
            logger.info(f"  ✅ Alpaca connecte | Equity: ${self.initial_balance:,.2f}")
        except ImportError:
            logger.error("alpaca-trade-api non installe. pip install alpaca-trade-api")
            raise
        except Exception as e:
            logger.error(f"Erreur connexion Alpaca: {e}")
            raise

    def get_equity(self, prices=None):
        account = self.api.get_account()
        return float(account.equity)

    def buy(self, ticker, price, amount_usd):
        qty = int(amount_usd / price)
        if qty < 1:
            return 0
        try:
            self.api.submit_order(
                symbol=ticker, qty=qty, side='buy',
                type='market', time_in_force='day'
            )
            return qty
        except Exception as e:
            logger.warning(f"  Alpaca BUY {ticker} erreur: {e}")
            return 0

    def sell(self, ticker, price, qty=None):
        try:
            if qty:
                self.api.submit_order(
                    symbol=ticker, qty=int(qty), side='sell',
                    type='market', time_in_force='day'
                )
            else:
                self.api.close_position(ticker)
            return qty or 0
        except Exception as e:
            logger.warning(f"  Alpaca SELL {ticker} erreur: {e}")
            return 0

    def get_positions(self):
        positions = {}
        for p in self.api.list_positions():
            positions[p.symbol] = {
                'qty': float(p.qty),
                'avg_price': float(p.avg_entry_price),
            }
        return positions


# ============================================================================
# DATA FETCHER (Live)
# ============================================================================

def fetch_live_data(tickers, period='5d', interval='1h'):
    """Telecharge les donnees recentes pour tous les tickers."""
    from data.universal_data_fetcher import UniversalDataFetcher
    fetcher = UniversalDataFetcher()

    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    for ticker in tickers:
        try:
            df = fetcher.fetch(ticker, start_date.strftime('%Y-%m-%d'),
                             end_date.strftime('%Y-%m-%d'), interval=interval)
            if df is not None and len(df) > 50:
                data[ticker] = df
        except Exception as e:
            logger.warning(f"  {ticker}: fetch erreur ({e})")

    return data


def get_current_prices(data):
    """Extrait les prix actuels des donnees."""
    prices = {}
    for ticker, df in data.items():
        if len(df) > 0:
            prices[ticker] = float(df['Close'].iloc[-1])
    return prices


# ============================================================================
# MODEL DECISION ENGINE
# ============================================================================

def get_model_actions(model, data, tickers, env_class, env_params, model_obs_size, vecnorm_path=None):
    """Fait tourner le modele sur les donnees actuelles et retourne les actions."""
    from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        env = env_class(data_dict=data, tickers=tickers, **env_params)

        if env.observation_space.shape[0] != model_obs_size:
            logger.warning(f"  Obs mismatch: env={env.observation_space.shape[0]} vs model={model_obs_size}")
            return None

        # Wrap in VecNormalize if needed
        if vecnorm_path and Path(vecnorm_path).exists():
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            obs = vec_env.reset()
        else:
            obs = env.reset()
            vec_env = None

        # Run model for a few steps to get its current decision
        action, _ = model.predict(obs, deterministic=True)
        return action

    except Exception as e:
        logger.error(f"  Erreur model: {e}")
        return None


# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def run_paper_trading(model_path, mode='simulate', api_key=None, api_secret=None,
                      initial_balance=DEFAULT_INITIAL_BALANCE,
                      interval_minutes=TRADE_INTERVAL_MINUTES,
                      max_hours=24, buy_pct=0.15):
    """Boucle principale de paper trading."""

    model_path = Path(model_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"  PAPER TRADER V7")
    logger.info(f"{'='*60}")
    logger.info(f"  Modele     : {model_path.name}")
    logger.info(f"  Mode       : {mode}")
    logger.info(f"  Balance    : ${initial_balance:,.0f}")
    logger.info(f"  Interval   : {interval_minutes} min")
    logger.info(f"  Max heures : {max_hours}")
    logger.info(f"{'='*60}\n")

    # Load model
    model = PPO.load(model_path)
    model_obs_size = model.observation_space.shape[0]

    # Load V7 metadata
    from scripts.backtest_ultimate import load_v7_metadata, detect_environment
    metadata, model_config, vecnorm_path = load_v7_metadata(model_path)

    env_version, n_tickers, meta_tickers, env_class, env_params = detect_environment(
        model, metadata=metadata, config=model_config
    )

    tickers = meta_tickers if meta_tickers else []
    if not tickers:
        logger.error("Pas de tickers dans metadata!")
        return

    logger.info(f"  {len(tickers)} tickers: {', '.join(tickers[:10])}...")

    # Initialize broker
    if mode == 'alpaca':
        broker = AlpacaBroker(api_key, api_secret)
        initial_balance = broker.initial_balance
    else:
        broker = SimulatedBroker(initial_balance)

    # Initialize components
    kill_switch = KillSwitch(initial_balance)
    journal = LiveTradeJournal()
    start_time = datetime.now()

    # Graceful shutdown
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("\n⚡ Arret demande (Ctrl+C)")
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    iteration = 0

    while running:
        iteration += 1
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600

        if elapsed_hours >= max_hours:
            logger.info(f"\n⏰ Duree max atteinte ({max_hours}h)")
            break

        logger.info(f"\n--- Iteration {iteration} | {datetime.now().strftime('%H:%M:%S')} | {elapsed_hours:.1f}h ---")

        # 1. Fetch live data
        data = fetch_live_data(tickers)
        if len(data) < len(tickers) // 2:
            logger.warning(f"  Pas assez de donnees ({len(data)}/{len(tickers)}). Skip.")
            time.sleep(interval_minutes * 60)
            continue

        prices = get_current_prices(data)

        # 2. Check kill switch
        equity = broker.get_equity(prices)
        journal.record_equity(equity)

        triggered, reason = kill_switch.check(equity)
        if triggered:
            logger.error(f"\n🛑 KILL SWITCH: {reason}")
            logger.error(f"  Equity: ${equity:,.2f} | Peak: ${kill_switch.peak_equity:,.2f}")
            break

        # 3. Get model actions
        actions = get_model_actions(model, data, list(data.keys()), env_class, env_params,
                                     model_obs_size, vecnorm_path)

        if actions is None:
            logger.warning("  Model n'a pas retourne d'actions. Skip.")
            time.sleep(interval_minutes * 60)
            continue

        # 4. Execute trades based on model actions
        # Actions: 0 = hold, 1 = buy, 2 = sell (for each ticker)
        n_actions = len(actions) if hasattr(actions, '__len__') else 1
        if not hasattr(actions, '__len__'):
            actions = [actions]

        actual_tickers = list(data.keys())
        for i, ticker in enumerate(actual_tickers):
            if i >= n_actions:
                break

            action = int(actions[i]) if i < len(actions) else 0
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            if action == 1:  # BUY
                amount = equity * buy_pct
                qty = broker.buy(ticker, price, amount)
                if qty > 0:
                    journal.record_trade(ticker, 'BUY', price, qty, qty * price)
                    kill_switch.record_trade()

            elif action == 2:  # SELL
                positions = broker.get_positions()
                if ticker in positions:
                    qty = broker.sell(ticker, price)
                    if qty > 0:
                        journal.record_trade(ticker, 'SELL', price, qty, qty * price)
                        kill_switch.record_trade()

        # 5. Status
        equity = broker.get_equity(prices)
        ret = (equity - initial_balance) / initial_balance
        drawdown = (kill_switch.peak_equity - equity) / kill_switch.peak_equity
        positions = broker.get_positions()

        logger.info(f"  Equity: ${equity:,.2f} ({ret:+.2%}) | DD: {drawdown:.1%} | Positions: {len(positions)}")

        # Sleep until next iteration
        if running:
            logger.info(f"  Prochain check dans {interval_minutes} min...")
            time.sleep(interval_minutes * 60)

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info(f"  FIN DU PAPER TRADING")
    logger.info(f"{'='*60}")

    summary = journal.get_summary(initial_balance)
    logger.info(f"  Trades      : {summary['n_trades']}")
    logger.info(f"  Return      : {summary.get('total_return', 0):+.2%}")
    logger.info(f"  Equity fin  : ${summary.get('final_equity', initial_balance):,.2f}")
    logger.info(f"  Duree       : {summary.get('duration_hours', 0):.1f}h")

    if kill_switch.triggered:
        logger.warning(f"  Kill Switch : {kill_switch.trigger_reason}")

    # Export
    report_dir = Path('logs/paper_trading')
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    journal.export(report_dir / f"journal_{ts}.json")

    report = {
        'model': str(model_path),
        'mode': mode,
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'initial_balance': initial_balance,
        'summary': summary,
        'kill_switch': {
            'triggered': kill_switch.triggered,
            'reason': kill_switch.trigger_reason,
            'alerts': kill_switch.alerts,
        },
        'tickers': tickers,
    }

    report_path = report_dir / f"report_{ts}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"  Rapport: {report_path}")

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Paper Trader V7 — Live Simulation')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modele V7 (.zip)')
    parser.add_argument('--mode', type=str, default='simulate', choices=['simulate', 'alpaca'],
                        help='Mode: simulate (local) ou alpaca (paper trading)')
    parser.add_argument('--api-key', type=str, default=None, help='Alpaca API key')
    parser.add_argument('--api-secret', type=str, default=None, help='Alpaca API secret')
    parser.add_argument('--balance', type=float, default=DEFAULT_INITIAL_BALANCE,
                        help=f'Balance initiale (defaut: ${DEFAULT_INITIAL_BALANCE:,.0f})')
    parser.add_argument('--interval', type=int, default=TRADE_INTERVAL_MINUTES,
                        help=f'Intervalle entre checks (minutes, defaut: {TRADE_INTERVAL_MINUTES})')
    parser.add_argument('--max-hours', type=float, default=24,
                        help='Duree max du paper trading (heures, defaut: 24)')
    parser.add_argument('--buy-pct', type=float, default=0.15,
                        help='Pourcentage equity par achat (defaut: 0.15)')
    parser.add_argument('--max-drawdown', type=float, default=KILL_SWITCH_MAX_DRAWDOWN,
                        help=f'Kill switch drawdown max (defaut: {KILL_SWITCH_MAX_DRAWDOWN:.0%})')
    parser.add_argument('--max-daily-loss', type=float, default=KILL_SWITCH_MAX_DAILY_LOSS,
                        help=f'Kill switch perte journaliere max (defaut: {KILL_SWITCH_MAX_DAILY_LOSS:.0%})')
    args = parser.parse_args()

    if args.mode == 'alpaca' and (not args.api_key or not args.api_secret):
        parser.error("Mode alpaca requiert --api-key et --api-secret")

    run_paper_trading(
        model_path=args.model,
        mode=args.mode,
        api_key=args.api_key,
        api_secret=args.api_secret,
        initial_balance=args.balance,
        interval_minutes=args.interval,
        max_hours=args.max_hours,
        buy_pct=args.buy_pct,
    )


if __name__ == '__main__':
    main()
