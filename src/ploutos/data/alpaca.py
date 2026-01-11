#!/usr/bin/env python3
"""
📡 ALPACA DATA FETCHER

Récupère données historiques via Alpaca API
Alternative fiable à yfinance

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import signal
from contextlib import contextmanager

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    print("⚠️  alpaca-py non installé: pip install alpaca-py")
    ALPACA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ★ TIMEOUT HANDLER
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager pour timeout
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class AlpacaDataFetcher:
    """
    Fetcher de données historiques via Alpaca
    Utilise le feed IEX gratuit
    """
    
    def __init__(self, api_key=None, secret_key=None):
        """
        Args:
            api_key: Clé API Alpaca (ou depuis env ALPACA_API_KEY)
            secret_key: Clé secrète (ou depuis env ALPACA_SECRET_KEY)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py non installé. Installez avec: pip install alpaca-py")
        
        # Récupérer credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Credentials Alpaca manquantes. Définissez:\n"
                "  export ALPACA_API_KEY='your_key'\n"
                "  export ALPACA_SECRET_KEY='your_secret'"
            )
        
        # Initialiser client avec feed IEX (GRATUIT)
        try:
            self.client = StockHistoricalDataClient(
                self.api_key, 
                self.secret_key,
                raw_data=False  # Retourne DataFrames directement
            )
            logger.info("✅ Alpaca Data Client initialisé (feed: IEX)")
        except Exception as e:
            logger.error(f"❌ Erreur init Alpaca: {e}")
            raise
    
    def fetch_historical(self, ticker, days=30, timeframe='1Day', timeout=10):
        """
        Récupère données historiques pour un ticker
        
        Args:
            ticker: Symbole ticker (ex: 'NVDA')
            days: Nombre de jours à charger
            timeframe: '1Day', '1Hour', '5Min', etc.
            timeout: Timeout en secondes
        
        Returns:
            DataFrame avec colonnes OHLCV
        """
        try:
            logger.debug(f"  🔄 {ticker}: Début fetch...")
            
            # ★ AVEC TIMEOUT
            with time_limit(timeout):
                # Calculer dates
                end = datetime.now()
                start = end - timedelta(days=days + 5)  # +5 jours marge pour week-ends
                
                # Mapper timeframe
                tf_map = {
                    '1Day': TimeFrame.Day,
                    '1Hour': TimeFrame.Hour,
                    '5Min': TimeFrame.Minute,
                    '15Min': TimeFrame(15, 'Min'),
                }
                
                tf = tf_map.get(timeframe, TimeFrame.Day)
                
                # Requête avec feed IEX
                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed='iex'  # 🔑 CLEF: Utiliser IEX gratuit au lieu de SIP
                )
                
                bars = self.client.get_stock_bars(request)
                
                # Convertir en DataFrame
                if ticker not in bars.data or not bars.data[ticker]:
                    logger.warning(f"⚠️  {ticker}: Aucune donnée retournée")
                    return pd.DataFrame()
                
                df = bars.df
                
                # Si MultiIndex (symbol, timestamp), simplifier
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(ticker, level='symbol')
                
                # Renommer colonnes pour compatibilité yfinance
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'trade_count': 'Trade_Count',
                    'vwap': 'VWAP'
                })
                
                # Garder seulement OHLCV
                available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
                df = df[available_cols]
                
                # Limiter au nombre de jours demandés
                if len(df) > days:
                    df = df.tail(days)
                
                logger.info(f"✅ {ticker}: {len(df)} barres chargées (IEX)")
                return df
        
        except TimeoutException:
            logger.error(f"❌ {ticker}: TIMEOUT ({timeout}s) - API trop lente")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Erreur fetch - {e}")
            return pd.DataFrame()
    
    def fetch_multiple(self, tickers, days=30, save_cache=True, cache_dir='data_cache'):
        """
        Récupère données pour plusieurs tickers
        
        Args:
            tickers: Liste de tickers
            days: Nombre de jours
            save_cache: Sauvegarder en cache
            cache_dir: Dossier cache
        
        Returns:
            dict: {ticker: DataFrame}
        """
        if save_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        data = {}
        failed = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.debug(f"  [{i}/{len(tickers)}] {ticker}...")
            
            try:
                df = self.fetch_historical(ticker, days=days, timeout=10)
                
                if not df.empty:
                    data[ticker] = df
                    
                    if save_cache:
                        cache_file = f"{cache_dir}/{ticker}.csv"
                        df.to_csv(cache_file)
                        logger.debug(f"💾 {ticker}: Sauvegardé en {cache_file}")
                else:
                    failed.append(ticker)
                    logger.warning(f"⚠️  {ticker}: Pas de données")
                    
            except Exception as e:
                failed.append(ticker)
                logger.error(f"❌ {ticker}: Échec - {e}")
                continue  # ★ NE PAS BLOQUER
        
        if failed:
            logger.warning(f"⚠️  {len(failed)} tickers échoués: {', '.join(failed)}")
        
        logger.info(f"✅ {len(data)}/{len(tickers)} tickers chargés")
        return data
    
    def get_latest_price(self, ticker):
        """
        Récupère le dernier prix connu
        
        Args:
            ticker: Symbole ticker
        
        Returns:
            float: Dernier prix close
        """
        try:
            df = self.fetch_historical(ticker, days=5)
            
            if df.empty:
                return None
            
            return float(df['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Erreur prix - {e}")
            return None

# Fonction helper pour compatibilité
def fetch_alpaca_data(tickers, days=30):
    """
    Helper pour fetch rapide
    
    Args:
        tickers: Liste de tickers ou ticker unique
        days: Nombre de jours
    
    Returns:
        dict ou DataFrame
    """
    fetcher = AlpacaDataFetcher()
    
    if isinstance(tickers, str):
        # Un seul ticker
        return fetcher.fetch_historical(tickers, days=days)
    else:
        # Plusieurs tickers
        return fetcher.fetch_multiple(tickers, days=days)
