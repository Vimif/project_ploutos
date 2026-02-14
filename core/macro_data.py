# core/macro_data.py
"""Récupération des données macroéconomiques (VIX, TNX, DXY).

Ces indicateurs fournissent le contexte de marché :
- VIX (^VIX) : Volatilité implicite S&P 500 → quand être défensif
- TNX (^TNX) : Taux 10 ans US → impact Tech/Growth
- DXY (DX-Y.NYB) : Dollar Index → impact matières premières

Usage:
    from core.macro_data import MacroDataFetcher

    macro = MacroDataFetcher()
    macro_data = macro.fetch_all(start_date='2023-01-01')
    # Returns: pd.DataFrame with columns [vix, tnx, dxy, vix_ma20, ...]
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from core.utils import setup_logging

logger = setup_logging(__name__)

# Tickers Yahoo Finance pour les données macro
MACRO_TICKERS = {
    'vix': '^VIX',
    'tnx': '^TNX',
    'dxy': 'DX-Y.NYB',
}


class MacroDataFetcher:
    """Récupère et transforme les données macroéconomiques."""

    def fetch_all(
        self,
        start_date: str = None,
        end_date: str = None,
        interval: str = '1h',
    ) -> pd.DataFrame:
        """Récupère VIX, TNX, DXY et calcule les features dérivées.

        Args:
            start_date: Date début (str 'YYYY-MM-DD' ou None pour 730j).
            end_date: Date fin (str ou None pour aujourd'hui).
            interval: '1h' ou '1d'.

        Returns:
            DataFrame avec colonnes macro (vix, tnx, dxy + dérivées).
        """
        import yfinance as yf

        if end_date is None:
            end_dt = datetime.now()
        elif isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date

        if start_date is None:
            start_dt = end_dt - timedelta(days=729)
        elif isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date

        # Yahoo limite les données horaires à 730 jours
        if interval in ['1h', '30m', '15m'] and (end_dt - start_dt).days > 729:
            start_dt = end_dt - timedelta(days=729)
            logger.warning(f"Limite Yahoo 730j pour {interval}, ajusté à {start_dt.date()}")

        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        logger.info(f"Fetch macro data ({start_str} -> {end_str}, {interval})")

        raw = {}
        for name, ticker in MACRO_TICKERS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                # Gestion du MultiIndex (nouveau yfinance)
                if isinstance(df.columns, pd.MultiIndex):
                    # Essayer de récupérer 'Close' pour ce ticker s'il est présent en niveau 1
                    try:
                         # Si format : Price | Ticker
                         #             Close | ^VIX
                         series = df.xs('Close', axis=1, level=0)
                         if isinstance(series, pd.DataFrame):
                             series = series.iloc[:, 0] # Prendre la première colonne si encore DataFrame
                    except KeyError:
                         # Si format plat ou autre, on prend juste la colonne 'Close'
                         series = df['Close']
                else:
                    series = df['Close']

                if len(series) > 0:
                    series.name = name # Renommer la Série
                    raw[name] = series
                    logger.info(f"  {name} ({ticker}): {len(series)} bars")
                else:
                    logger.warning(f"  {name} ({ticker}): aucune donnée")
            except Exception:
                logger.warning(f"  {name} ({ticker}): fetch échoué", exc_info=True)

        if not raw:
            logger.error("Aucune donnée macro récupérée")
            return pd.DataFrame()

        # Combiner sur le même index
        macro_df = pd.DataFrame(raw)

        # Forward-fill pour aligner les timestamps (marchés différents)
        macro_df = macro_df.ffill().bfill()

        # Retirer timezone si présente
        if hasattr(macro_df.index, 'tz') and macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)

        # Calculer features dérivées
        macro_df = self._compute_features(macro_df)

        # Cleanup
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.ffill().fillna(0)

        logger.info(f"Macro data: {len(macro_df)} bars, {len(macro_df.columns)} features")
        return macro_df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features dérivées des données macro."""
        for col in ['vix', 'tnx', 'dxy']:
            if col not in df.columns:
                continue

            series = df[col]

            # Moyennes mobiles
            df[f'{col}_ma20'] = series.rolling(20, min_periods=1).mean()
            df[f'{col}_ma50'] = series.rolling(50, min_periods=1).mean()

            # Variation
            df[f'{col}_pct_1'] = series.pct_change(1)
            df[f'{col}_pct_5'] = series.pct_change(5)

            # Z-score (distance à la moyenne)
            ma = df[f'{col}_ma20']
            std = series.rolling(20, min_periods=1).std()
            df[f'{col}_zscore'] = (series - ma) / (std + 1e-8)

        # VIX spécifique : régimes de peur
        if 'vix' in df.columns:
            df['vix_fear'] = (df['vix'] > 25).astype(np.float32)
            df['vix_extreme_fear'] = (df['vix'] > 35).astype(np.float32)
            df['vix_complacent'] = (df['vix'] < 15).astype(np.float32)

        # TNX spécifique : environnement de taux
        if 'tnx' in df.columns:
            df['tnx_rising'] = (df['tnx_pct_5'] > 0.02).astype(np.float32)
            df['tnx_falling'] = (df['tnx_pct_5'] < -0.02).astype(np.float32)

        # DXY spécifique : force du dollar
        if 'dxy' in df.columns:
            df['dxy_strong'] = (df['dxy_zscore'] > 1.0).astype(np.float32)
            df['dxy_weak'] = (df['dxy_zscore'] < -1.0).astype(np.float32)

        return df

    def align_to_ticker(
        self, macro_df: pd.DataFrame, ticker_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aligne les données macro sur l'index d'un ticker.

        Utilise merge_asof pour aligner sur le timestamp le plus proche
        (les marchés macro et actions n'ont pas exactement les mêmes horaires).

        Args:
            macro_df: DataFrame des données macro (output de fetch_all).
            ticker_df: DataFrame d'un ticker (OHLCV).

        Returns:
            DataFrame macro réindexé sur ticker_df.index.
        """
        if macro_df.empty:
            return pd.DataFrame(index=ticker_df.index)

        # Réindexer sur l'index du ticker, forward-fill
        aligned = macro_df.reindex(ticker_df.index, method='ffill')
        aligned = aligned.bfill().fillna(0)

        return aligned
