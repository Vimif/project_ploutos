"""Fonctions utilitaires diverses"""

import os
from pathlib import Path
from typing import List, Dict
import pandas as pd

def ensure_dir(path: str) -> Path:
    """
    Crée un dossier s'il n'existe pas
    
    Args:
        path: Chemin du dossier
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_cache_files(cache_dir: str = 'data_cache') -> List[str]:
    """
    Liste tous les fichiers dans le cache
    
    Args:
        cache_dir: Dossier du cache
        
    Returns:
        Liste des tickers disponibles
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return []
    
    tickers = []
    for file in cache_path.glob('*.csv'):
        # Extraire ticker du nom fichier
        ticker = file.stem.replace('_730d', '')
        tickers.append(ticker)
    
    return sorted(set(tickers))

def load_cached_data(tickers: List[str], cache_dir: str = 'data_cache') -> Dict[str, pd.DataFrame]:
    """
    Charge les données depuis le cache
    
    Args:
        tickers: Liste des tickers à charger
        cache_dir: Dossier du cache
        
    Returns:
        Dict {ticker: DataFrame}
    """
    data = {}
    
    for ticker in tickers:
        # Essayer avec suffix _730d d'abord
        for suffix in ['_730d', '']:
            file_path = Path(cache_dir) / f"{ticker}{suffix}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Vérifier colonnes requises
                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required):
                        data[ticker] = df
                        break
                        
                except Exception:
                    continue
    
    return data

def format_number(value: float, precision: int = 2, suffix: str = '') -> str:
    """
    Formate un nombre pour affichage
    
    Args:
        value: Valeur à formater
        precision: Nombre de décimales
        suffix: Suffixe (%, $, etc.)
        
    Returns:
        String formaté
    """
    if abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f}B{suffix}"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K{suffix}"
    else:
        return f"{value:.{precision}f}{suffix}"
