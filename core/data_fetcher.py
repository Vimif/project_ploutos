"""
Data Fetcher Multi-Sources avec Fallback Automatique
Priorise Alpaca → yfinance → Polygon.io
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ✅ AJOUT : Charger .env
from dotenv import load_dotenv
load_dotenv()

from core.utils import setup_logging
logger = setup_logging(__name__)

class UniversalDataFetcher:
    """
    Récupère les données de marché depuis plusieurs sources
    avec fallback automatique pour une robustesse maximale
    """
    
    def __init__(self):
        """Initialise toutes les APIs disponibles"""
        self.alpaca_api = self._init_alpaca()
        self.polygon_api = self._init_polygon()
        
        # Ordre de priorité des sources
        self.sources = []
        if self.alpaca_api:
            self.sources.append('alpaca')
        self.sources.append('yfinance')  # Toujours disponible
        if self.polygon_api:
            self.sources.append('polygon')
        
        logger.info(f"📡 Data Fetcher initialisé : {', '.join(self.sources)}")
        
    def _init_alpaca(self):
        """Initialise Alpaca avec la nouvelle API alpaca-py"""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta
            
            # ✅ FIX : Nom correct des variables
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')  # ❌ Était ALPACA_SECRET_KEY
            
            # Essayer aussi avec le nom alternatif
            if not api_secret:
                api_secret = os.getenv('ALPACA_API_SECRET')  # ✅ Alternative
            
            if not api_key or not api_secret:
                logger.warning("⚠️ Variables ALPACA_API_KEY/SECRET non définies")
                return None
            
            # Créer client
            client = StockHistoricalDataClient(api_key, api_secret)
            
            # ✅ FIX : Test AVANT le return
            try:
                test_request = StockBarsRequest(
                    symbol_or_symbols="SPY",
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=2)
                )
                client.get_stock_bars(test_request)
                logger.info("✅ Alpaca connecté (alpaca-py)")
                return client
            except Exception:
                logger.exception("⚠️ Alpaca test échec")
                return None
            
        except ImportError:
            logger.warning("⚠️ alpaca-py non installé (pip install alpaca-py)")
            return None
        except Exception:
            logger.exception("⚠️ Alpaca échec")
            return None

    
    def _init_polygon(self):
        """Initialise Polygon.io (optionnel, payant mais excellent)"""
        try:
            from polygon import RESTClient
            
            api_key = os.getenv('POLYGON_API_KEY')
            
            if not api_key:
                return None
            
            client = RESTClient(api_key=api_key)
            logger.info("✅ Polygon connecté")
            return client
            
        except ImportError:
            return None
        except Exception:
            logger.exception("⚠️ Polygon échec")
            return None
    
    def fetch(self, ticker, start_date=None, end_date=None, interval='1h', max_retries=2):
        """
        Récupère les données avec fallback automatique
        
        Args:
            ticker: Symbol (ex: 'NVDA')
            start_date: Date début (datetime, str 'YYYY-MM-DD', ou None)
            end_date: Date fin (datetime, str, ou None)
            interval: '1m', '5m', '15m', '1h', '1d'
            max_retries: Nombre de tentatives par source
            
        Returns:
            pd.DataFrame avec colonnes [Open, High, Low, Close, Volume]
        """
        
        # Dates par défaut : 730 derniers jours
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)
        
        # Convertir en strings si nécessaire
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = start_date
            
        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = end_date
        
        logger.info(f"📥 Fetch {ticker} ({start_str} → {end_str}, {interval})")
        
        # Essayer chaque source dans l'ordre
        for source in self.sources:
            for attempt in range(max_retries):
                try:
                    if source == 'alpaca' and self.alpaca_api:
                        df = self._fetch_alpaca(ticker, start_str, end_str, interval)
                        
                    elif source == 'yfinance':
                        df = self._fetch_yfinance(ticker, start_str, end_str, interval)
                        
                    elif source == 'polygon' and self.polygon_api:
                        df = self._fetch_polygon(ticker, start_str, end_str, interval)
                        
                    else:
                        continue
                    
                    # Validation
                    if df is not None and len(df) > 100:
                        df_normalized = self._normalize_dataframe(df)
                        logger.info(f"✅ {source} : {len(df_normalized)} bougies")
                        return df_normalized
                    
                except Exception:
                    if attempt == max_retries - 1:
                        logger.warning(f"⚠️ {source} échec (tentative {attempt+1}/{max_retries})", exc_info=True)
                    continue
        
        # Si toutes les sources ont échoué
        raise ValueError(f"❌ Impossible de récupérer {ticker} depuis aucune source")
    
    def _fetch_alpaca(self, ticker, start_date, end_date, interval):
        """Récupère depuis Alpaca (nouvelle API alpaca-py)"""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime
        
        # Mapping interval
        timeframe_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, "Min"),
            '15m': TimeFrame(15, "Min"),
            '30m': TimeFrame(30, "Min"),
            '1h': TimeFrame.Hour,
            '4h': TimeFrame(4, "Hour"),
            '1d': TimeFrame.Day
        }
        timeframe = timeframe_map.get(interval, TimeFrame.Hour)
        
        # Convertir dates en datetime
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
        
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        # Requête
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt
        )
        
        bars = self.alpaca_api.get_stock_bars(request)
        
        # Convertir en DataFrame
        df = bars.df
        
        # Si MultiIndex (plusieurs symboles), extraire le ticker
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol')
        
        # Renommer colonnes pour uniformité
        df.columns = [col.title() for col in df.columns]
        
        return df
    
    def _fetch_yfinance(self, ticker, start_date, end_date, interval):
        """Récupère depuis yfinance (fallback universel)"""
        import yfinance as yf
        
        # ✅ FIX : Gérer la limite 730 jours pour les intervalles horaires
        if interval in ['1h', '30m', '15m', '5m', '1m']:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            
            delta_days = (end_dt - start_dt).days
            
            if delta_days > 729:
                # Ajuster automatiquement
                start_date = (end_dt - timedelta(days=729)).strftime('%Y-%m-%d')
                logger.warning(f"⚠️ Yahoo limite 730j pour {interval} : ajusté à {start_date}")
        
        # Mapping interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d'
        }
        yf_interval = interval_map.get(interval, '1h')
        
        # Télécharger
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=yf_interval,
            progress=False,
            auto_adjust=True  # Prix ajustés
        )
        
        # Flatten MultiIndex si présent
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        return df
    
    def _fetch_polygon(self, ticker, start_date, end_date, interval):
        """Récupère depuis Polygon.io (premium mais très fiable)"""
        
        # Mapping interval
        if interval == '1d':
            multiplier, timespan = 1, 'day'
        elif interval == '1h':
            multiplier, timespan = 1, 'hour'
        elif interval == '5m':
            multiplier, timespan = 5, 'minute'
        elif interval == '15m':
            multiplier, timespan = 15, 'minute'
        else:
            multiplier, timespan = 1, 'hour'
        
        # Requête
        aggs = []
        for agg in self.polygon_api.list_aggs(
            ticker,
            multiplier,
            timespan,
            start_date,
            end_date,
            limit=50000
        ):
            aggs.append({
                'timestamp': agg.timestamp,
                'Open': agg.open,
                'High': agg.high,
                'Low': agg.low,
                'Close': agg.close,
                'Volume': agg.volume
            })
        
        # Conversion DataFrame
        df = pd.DataFrame(aggs)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        
        return df
    
    def _normalize_dataframe(self, df):
        """
        Normalise le DataFrame pour un format uniforme
        
        Returns:
            DataFrame avec [Open, High, Low, Close, Volume] et index DatetimeIndex
        """
        
        # 1. Nettoyer l'index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index)
        
        # Retirer timezone (uniformiser en UTC naïve)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 2. Mapping possible des noms de colonnes
        col_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
            'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume',
            'adj close': 'Close'  # Utiliser adjusted close
        }
        
        # Renommer (case-insensitive)
        df.columns = df.columns.str.lower()
        df = df.rename(columns=col_mapping)
        
        # 3. Garder seulement OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in required_cols if c in df.columns]
        
        if len(available_cols) < 5:
            raise ValueError(f"Colonnes manquantes : {set(required_cols) - set(available_cols)}")
        
        df = df[available_cols]
        
        # 4. Forcer types numériques
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Supprimer NaN et doublons
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        
        # 6. Trier par date
        df = df.sort_index()
        
        return df
    
    def bulk_fetch(self, tickers, start_date=None, end_date=None, interval='1h', save_to_cache=True):
        """
        Récupère plusieurs tickers en parallèle
        
        Args:
            tickers: Liste de symbols
            save_to_cache: Si True, sauvegarde dans data_cache/
            
        Returns:
            dict: {ticker: DataFrame}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"📦 Bulk fetch : {len(tickers)} tickers")
        
        results = {}
        failed = []
        
        def fetch_one(ticker):
            try:
                df = self.fetch(ticker, start_date, end_date, interval)
                
                if save_to_cache:
                    self.save_to_cache(ticker, df)
                
                return (ticker, df)
            except Exception:
                logger.error(f"❌ {ticker} : Échec du fetch", exc_info=True)
                return (ticker, None)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fetch_one, ticker) for ticker in tickers]
            
            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    results[ticker] = df
                else:
                    failed.append(ticker)
        
        logger.info(f"✅ Succès: {len(results)}/{len(tickers)}")
        if failed:
            logger.error(f"❌ Échecs: {', '.join(failed)}")
        
        return results
    
    def save_to_cache(self, ticker, df, cache_dir='data_cache'):
        """Sauvegarde dans le cache local"""
        os.makedirs(cache_dir, exist_ok=True)
        path = f"{cache_dir}/{ticker}.csv"
        df.to_csv(path)
    
    def load_from_cache(self, ticker, cache_dir='data_cache', max_age_days=7):
        """
        Charge depuis le cache si récent
        
        Args:
            max_age_days: Age maximum du cache en jours
            
        Returns:
            DataFrame ou None si cache trop vieux/absent
        """
        path = f"{cache_dir}/{ticker}.csv"
        
        if not os.path.exists(path):
            return None
        
        # Vérifier l'âge du fichier
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        
        if file_age.days > max_age_days:
            logger.warning(f"⏰ Cache {ticker} trop vieux ({file_age.days} jours)")
            return None
        
        try:
            from core.data_loader import load_market_data
            df = load_market_data(path)
            logger.info(f"💾 {ticker} chargé depuis cache")
            return df
        except Exception:
            logger.exception(f"❌ Erreur lors du chargement du cache pour {ticker}")
            return None


# ✅ AJOUT : Fonction wrapper pour compatibilité
def download_data(tickers, period='2y', interval='1h'):
    """
    Fonction wrapper simple pour télécharger des données
    
    Args:
        tickers: Liste de tickers ou ticker unique (str)
        period: Période ('1y', '2y', '5y', etc.)
        interval: Intervalle ('1h', '1d', etc.)
    
    Returns:
        dict: {ticker: DataFrame} ou DataFrame si ticker unique
    """
    # Convertir period en dates
    end_date = datetime.now()
    
    period_map = {
        '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650
    }
    
    days = period_map.get(period, 730)  # Défaut 2 ans
    start_date = end_date - timedelta(days=days)
    
    # Si interval horaire, limiter à 730 jours max
    if interval in ['1h', '30m', '15m', '5m'] and days > 730:
        days = 729
        start_date = end_date - timedelta(days=days)
        logger.warning(f"⚠️ Limite Yahoo 730j pour {interval}, ajusté à 2 ans")
    
    # Créer fetcher
    fetcher = UniversalDataFetcher()
    
    # Si ticker unique
    if isinstance(tickers, str):
        return fetcher.fetch(tickers, start_date, end_date, interval)
    
    # Si liste de tickers
    return fetcher.bulk_fetch(tickers, start_date, end_date, interval, save_to_cache=False)
