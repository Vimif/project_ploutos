# dashboard/analytics.py
"""Module d'analytics avancé pour le dashboard Ploutos"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
import pandas as pd

from core.utils import setup_logging

logger = setup_logging(__name__, "analytics.log")


class PortfolioAnalytics:
    """Calculateur de métriques financières avancées"""

    def __init__(self, trades: list[dict], daily_summaries: list[dict] = None):
        """
        Initialiser l'analyseur

        Args:
            trades: Liste des trades (depuis BDD ou JSON)
            daily_summaries: Résumés quotidiens (optionnel)
        """
        self.trades = trades
        self.daily_summaries = daily_summaries or []
        self.df_trades = self._trades_to_dataframe()
        self.df_daily = self._daily_to_dataframe()

    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convertir trades en DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        return df

    def _daily_to_dataframe(self) -> pd.DataFrame:
        """Convertir résumés quotidiens en DataFrame"""
        if not self.daily_summaries:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_summaries)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df

    def calculate_returns(self) -> pd.Series:
        """
        Calculer les rendements quotidiens

        Returns:
            Series avec les rendements quotidiens
        """
        if self.df_daily.empty or "portfolio_value" not in self.df_daily.columns:
            logger.warning("⚠️  Pas de données daily pour calcul returns")
            return pd.Series(dtype=float)

        returns = self.df_daily["portfolio_value"].pct_change().dropna()
        return returns

    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculer le ratio de Sharpe (annualisé)

        Args:
            risk_free_rate: Taux sans risque annuel (défaut 5%)

        Returns:
            Sharpe ratio (float)
        """
        returns = self.calculate_returns()

        if returns.empty or len(returns) < 2:
            logger.warning("⚠️  Pas assez de données pour Sharpe")
            return 0.0

        # Annualiser (252 jours de trading)
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

        return float(sharpe) if not np.isnan(sharpe) else 0.0

    def sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculer le ratio de Sortino (annualisé)
        Comme Sharpe mais ne pénalise que la volatilité baissière

        Args:
            risk_free_rate: Taux sans risque annuel

        Returns:
            Sortino ratio (float)
        """
        returns = self.calculate_returns()

        if returns.empty or len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        sortino = np.sqrt(252) * (excess_returns.mean() / downside_std)

        return float(sortino) if not np.isnan(sortino) else 0.0

    def max_drawdown(self) -> tuple[float, str, str]:
        """
        Calculer le drawdown maximum

        Returns:
            Tuple (max_dd%, date_début, date_fin)
        """
        if self.df_daily.empty or "portfolio_value" not in self.df_daily.columns:
            return (0.0, "", "")

        portfolio_values = self.df_daily["portfolio_value"].values
        dates = self.df_daily["date"].dt.strftime("%Y-%m-%d").values

        # Calculer running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Drawdown en %
        drawdowns = (portfolio_values - running_max) / running_max * 100

        # Trouver le max drawdown
        max_dd_idx = np.argmin(drawdowns)
        max_dd = drawdowns[max_dd_idx]

        # Trouver la date du pic précédent
        peak_idx = np.argmax(running_max[: max_dd_idx + 1])

        return (
            float(max_dd),
            dates[peak_idx] if peak_idx < len(dates) else "",
            dates[max_dd_idx] if max_dd_idx < len(dates) else "",
        )

    def calmar_ratio(self) -> float:
        """
        Calculer le ratio de Calmar
        = Rendement annualisé / Max Drawdown

        Returns:
            Calmar ratio (float)
        """
        if self.df_daily.empty or "portfolio_value" not in self.df_daily.columns:
            return 0.0

        # Rendement total sur la période
        total_return = (
            self.df_daily["portfolio_value"].iloc[-1] / self.df_daily["portfolio_value"].iloc[0] - 1
        )

        # Annualiser
        days = (self.df_daily["date"].iloc[-1] - self.df_daily["date"].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # Max DD
        max_dd, _, _ = self.max_drawdown()

        if max_dd == 0:
            return 0.0

        calmar = annualized_return / abs(max_dd / 100)

        return float(calmar) if not np.isnan(calmar) else 0.0

    def win_rate(self) -> dict[str, float]:
        """
        Calculer le win rate (paires BUY->SELL rentables)

        Returns:
            Dict avec wins, losses, total, win_rate%
        """
        if self.df_trades.empty:
            return {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0}

        # Grouper par symbole
        symbols = self.df_trades["symbol"].unique()

        wins = 0
        losses = 0

        for symbol in symbols:
            symbol_trades = self.df_trades[self.df_trades["symbol"] == symbol].copy()
            symbol_trades = symbol_trades.sort_values("timestamp")

            # Trouver les paires BUY->SELL
            buy_price = None
            for _, trade in symbol_trades.iterrows():
                if trade["action"] == "BUY":
                    buy_price = float(trade["price"])
                elif trade["action"] == "SELL" and buy_price is not None:
                    sell_price = float(trade["price"])

                    if sell_price > buy_price:
                        wins += 1
                    else:
                        losses += 1

                    buy_price = None

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0

        return {"wins": wins, "losses": losses, "total": total, "win_rate": win_rate}

    def avg_win_loss(self) -> dict[str, float]:
        """
        Calculer le gain moyen vs perte moyenne

        Returns:
            Dict avec avg_win, avg_loss, profit_factor
        """
        if self.df_trades.empty:
            return {"avg_win": 0, "avg_loss": 0, "profit_factor": 0}

        symbols = self.df_trades["symbol"].unique()

        wins = []
        losses = []

        for symbol in symbols:
            symbol_trades = self.df_trades[self.df_trades["symbol"] == symbol].copy()
            symbol_trades = symbol_trades.sort_values("timestamp")

            buy_price = None
            buy_qty = None

            for _, trade in symbol_trades.iterrows():
                if trade["action"] == "BUY":
                    buy_price = float(trade["price"])
                    buy_qty = float(trade.get("quantity", 1))

                elif trade["action"] == "SELL" and buy_price is not None:
                    sell_price = float(trade["price"])
                    sell_qty = float(trade.get("quantity", buy_qty))

                    pnl = (sell_price - buy_price) * min(buy_qty, sell_qty)

                    if pnl > 0:
                        wins.append(pnl)
                    elif pnl < 0:
                        losses.append(abs(pnl))

                    buy_price = None
                    buy_qty = None

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 0

        return {
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
        }

    def trades_by_symbol(self) -> dict[str, dict]:
        """
        Statistiques par symbole

        Returns:
            Dict[symbol, stats]
        """
        if self.df_trades.empty:
            return {}

        stats_by_symbol = {}

        for symbol in self.df_trades["symbol"].unique():
            symbol_trades = self.df_trades[self.df_trades["symbol"] == symbol]

            buy_trades = symbol_trades[symbol_trades["action"] == "BUY"]
            sell_trades = symbol_trades[symbol_trades["action"] == "SELL"]

            total_volume = symbol_trades["amount"].sum()

            stats_by_symbol[symbol] = {
                "total_trades": len(symbol_trades),
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "total_volume": float(total_volume),
                "avg_price": float(symbol_trades["price"].mean()),
            }

        return stats_by_symbol

    def get_all_metrics(self) -> dict:
        """
        Calculer toutes les métriques en une seule fois

        Returns:
            Dict avec toutes les métriques
        """
        logger.info("📊 Calcul des métriques avancées...")

        # Ratios de performance
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()
        calmar = self.calmar_ratio()

        # Drawdown
        max_dd, dd_start, dd_end = self.max_drawdown()

        # Win/Loss
        win_loss = self.win_rate()
        avg_wl = self.avg_win_loss()

        # Par symbole
        by_symbol = self.trades_by_symbol()

        metrics = {
            "performance_ratios": {
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "calmar_ratio": round(calmar, 2),
            },
            "risk_metrics": {
                "max_drawdown_pct": round(max_dd, 2),
                "max_drawdown_start": dd_start,
                "max_drawdown_end": dd_end,
            },
            "win_loss": {
                "wins": win_loss["wins"],
                "losses": win_loss["losses"],
                "total_trades": win_loss["total"],
                "win_rate_pct": round(win_loss["win_rate"], 2),
                "avg_win": round(avg_wl["avg_win"], 2),
                "avg_loss": round(avg_wl["avg_loss"], 2),
                "profit_factor": round(avg_wl["profit_factor"], 2),
            },
            "by_symbol": by_symbol,
        }

        logger.info("✅ Métriques calculées")
        return metrics


def calculate_benchmark_comparison(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> dict:
    """
    Comparer le portfolio à un benchmark (ex: SPY)

    Args:
        portfolio_returns: Rendements du portfolio
        benchmark_returns: Rendements du benchmark

    Returns:
        Dict avec alpha, beta, correlation
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {"alpha": 0, "beta": 0, "correlation": 0}

    # Aligner les dates
    combined = pd.DataFrame(
        {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
    ).dropna()

    if len(combined) < 2:
        return {"alpha": 0, "beta": 0, "correlation": 0}

    # Beta (sensibilité au marché)
    covariance = combined["portfolio"].cov(combined["benchmark"])
    benchmark_var = combined["benchmark"].var()
    beta = covariance / benchmark_var if benchmark_var > 0 else 0

    # Alpha (rendement excédentaire)
    portfolio_mean = combined["portfolio"].mean()
    benchmark_mean = combined["benchmark"].mean()
    alpha = portfolio_mean - (beta * benchmark_mean)

    # Annualiser alpha
    alpha_annualized = alpha * 252

    # Corrélation
    correlation = combined["portfolio"].corr(combined["benchmark"])

    return {
        "alpha": float(alpha_annualized) if not np.isnan(alpha_annualized) else 0,
        "beta": float(beta) if not np.isnan(beta) else 0,
        "correlation": float(correlation) if not np.isnan(correlation) else 0,
    }
