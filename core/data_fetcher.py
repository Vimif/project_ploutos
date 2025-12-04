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
        
        print(f"📡 Data Fetcher initialisé : {', '.join(self.sources)}")
        
    def _init_alpaca(self):
        """Initialise Alpaca (données gratuites et illimitées)"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Récupérer les clés depuis variables d'environnement
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not api_secret:
                print("⚠️ Variables ALPACA_API_KEY/SECRET non définies")
                return None
            
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=api_secret,
                base_url='https://paper-api.alpaca.markets'
            )
            
            # Test de connexion
            api.get_account()
            print("  ✅ Alpaca connecté")
            return api
            
        except ImportError:
            print("  ⚠️ alpaca-trade-api non installé (pip install alpaca-trade-api)")
            return None
        except Exception as e:
            print(f"  ⚠️ Alpaca échec : {str(e)[:50]}")
            return None
    
    def _init_polygon(self):
        """Initialise Polygon.io (optionnel, payant mais excellent)"""
        try:
            from polygon import RESTClient
            
            api_key = os.getenv('POLYGON_API_KEY')
            
            if not api_key:
                return None
            
            client = RESTClient(api_key=api_key)
            print("  ✅ Polygon connecté")
            return client
            
        except ImportError:
            return None
        except Exception as e:
            print(f"  ⚠️ Polygon échec : {str(e)[:50]}")
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
        
        print(f"📥 Fetch {ticker} ({start_str} → {end_str}, {interval})")
        
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
                        print(f"  ✅ {source} : {len(df_normalized)} bougies")
                        return df_normalized
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"  ⚠️ {source} échec (tentative {attempt+1}/{max_retries}) : {str(e)[:50]}")
                    continue
        
        # Si toutes les sources ont échoué
        raise ValueError(f"❌ Impossible de récupérer {ticker} depuis aucune source")
    
    def _fetch_alpaca(self, ticker, start_date, end_date, interval):
        """Récupère depuis Alpaca (ILLIMITÉ et GRATUIT)"""
        
        # Conversion interval
        timeframe_map = {
            '1m': '1Min', '5m': '5Min', '15m': '15Min', '30m': '30Min',
            '1h': '1Hour', '4h': '4Hour', '1d': '1Day'
        }
        timeframe = timeframe_map.get(interval, '1Hour')
        
        # Requête Alpaca v2 API
        bars = self.alpaca_api.get_bars(
            ticker,
            timeframe,
            start=start_date,
            end=end_date,
            adjustment='all',  # Ajusté pour splits/dividendes
            limit=10000
        ).df
        
        return bars
    
    def _fetch_yfinance(self, ticker, start_date, end_date, interval):
        """Récupère depuis yfinance (fallback universel)"""
        import yfinance as yf
        
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
        
        print(f"\n📦 Bulk fetch : {len(tickers)} tickers")
        
        results = {}
        failed = []
        
        def fetch_one(ticker):
            try:
                df = self.fetch(ticker, start_date, end_date, interval)
                
                if save_to_cache:
                    self.save_to_cache(ticker, df)
                
                return (ticker, df)
            except Exception as e:
                print(f"  ❌ {ticker} : {str(e)[:50]}")
                return (ticker, None)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_one, ticker) for ticker in tickers]
            
            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    results[ticker] = df
                else:
                    failed.append(ticker)
        
        print(f"\n✅ Succès: {len(results)}/{len(tickers)}")
        if failed:
            print(f"❌ Échecs: {', '.join(failed)}")
        
        return results
    
    def save_to_cache(self, ticker, df, cache_dir='data_cache'):
        """Sauvegarde dans le cache local"""
        os.makedirs(cache_dir, exist_ok=True)
        path = f"{cache_dir}/{ticker}.csv"
        df.to_csv(path)
        # print(f"  💾 {ticker} → {path}")
    
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
            print(f"  ⏰ Cache {ticker} trop vieux ({file_age.days} jours)")
            return None
        
        try:
            from core.data_loader import load_market_data
            df = load_market_data(path)
            print(f"  💾 {ticker} chargé depuis cache")
            return df
        except:
            return None
