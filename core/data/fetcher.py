"""
Data Fetcher - Téléchargement données Yahoo Finance
Version refactorisée
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from utils.logger import PloutosLogger
from utils.helpers import ensure_dir

logger = PloutosLogger().get_logger(__name__)

class UniversalDataFetcher:
    """
    Télécharge et cache les données de marché
    """
    
    def __init__(self, cache_dir: str = 'data_cache'):
        """
        Args:
            cache_dir: Dossier de cache
        """
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir)
    
    def fetch(
        self,
        ticker: str,
        period: str = '730d',
        interval: str = '1h',
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Télécharge données pour un ticker
        
        Args:
            ticker: Symbole (ex: 'AAPL')
            period: Période ('730d', '1y', etc.)
            interval: Intervalle ('1h', '1d', etc.)
            use_cache: Utiliser cache si disponible
            
        Returns:
            DataFrame ou None si erreur
        """
        
        # Vérifier cache
        cache_file = self.cache_dir / f"{ticker}_{period}.csv"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Vérifier fraîcheur (< 24h)
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                if file_age < timedelta(hours=24):
                    logger.debug(f"Cache hit: {ticker} (age: {file_age.seconds//3600}h)")
                    return df
                else:
                    logger.debug(f"Cache expiré: {ticker}")
            
            except Exception as e:
                logger.warning(f"Erreur lecture cache {ticker}: {e}")
        
        # Télécharger
        logger.info(f"Téléchargement {ticker} ({period}, {interval})...")
        
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"Pas de données pour {ticker}")
                return None
            
            # Nettoyer
            df = df.dropna()
            
            if len(df) < 100:
                logger.warning(f"Données insuffisantes pour {ticker} ({len(df)} rows)")
                return None
            
            # Sauvegarder cache
            df.to_csv(cache_file)
            logger.debug(f"Cache sauvegardé: {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur téléchargement {ticker}: {e}")
            return None
    
    def bulk_fetch(
        self,
        tickers: List[str],
        period: str = '730d',
        interval: str = '1h',
        save_to_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge plusieurs tickers
        
        Args:
            tickers: Liste de symboles
            period: Période
            interval: Intervalle
            save_to_cache: Sauvegarder dans cache
            
        Returns:
            Dict {ticker: DataFrame}
        """
        
        logger.info(f"Téléchargement bulk: {len(tickers)} tickers")
        
        data = {}
        
        for ticker in tickers:
            df = self.fetch(
                ticker=ticker,
                period=period,
                interval=interval,
                use_cache=save_to_cache
            )
            
            if df is not None:
                data[ticker] = df
        
        success_rate = len(data) / len(tickers) * 100
        logger.info(f"✅ {len(data)}/{len(tickers)} tickers chargés ({success_rate:.0f}%)")
        
        return data
    
    def clear_cache(self, ticker: Optional[str] = None):
        """
        Vide le cache
        
        Args:
            ticker: Ticker spécifique (ou None pour tout)
        """
        
        if ticker:
            pattern = f"{ticker}_*.csv"
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob("*.csv"))
        
        for file in files:
            file.unlink()
            logger.debug(f"Cache supprimé: {file.name}")
        
        logger.info(f"✅ {len(files)} fichiers cache supprimés")
