import pandas as pd
import numpy as np
import os

def load_market_data(csv_path):
    """
    Charge et nettoie les données de marché de manière ULTRA-ROBUSTE.
    Gère tous les cas edge: headers multiples, timezones, types str/float.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    
    try:
        # 1. Lecture initiale pour détecter le format
        df = pd.read_csv(csv_path)
        
        # 2. Détection et suppression des headers parasites (ex: "Ticker" de yfinance)
        # Si la première colonne contient "Ticker", "Date", etc., on skip
        if 'Ticker' in df.columns or df.iloc[0].astype(str).str.contains('Ticker|Date', case=False).any():
            # Relire sans header, puis définir les colonnes manuellement
            df = pd.read_csv(csv_path, skiprows=1)
            
        # 3. Identifier la colonne Date (souvent la première ou nommée 'Date'/'Datetime')
        date_col = None
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'Datetime' in df.columns:
            date_col = 'Datetime'
        elif df.columns[0].lower() in ['date', 'datetime', 'timestamp']:
            date_col = df.columns[0]
        else:
            # Première colonne par défaut
            date_col = df.columns[0]
        
        # 4. Set index sur la colonne date
        df = df.set_index(date_col)
        
        # 5. Forcer conversion en Datetime (ignore les erreurs)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        
        # Supprimer les lignes avec index invalide (NaT = Not a Time)
        df = df[df.index.notna()]
        
        # 6. Retirer la timezone pour éviter les conflits
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # 7. Nettoyage des colonnes numériques
        # yfinance peut mettre des colonnes comme "Open SPY", "Close SPY"
        # On renomme pour standardiser
        df.columns = df.columns.str.replace(r'\s+(SPY|NVDA|[A-Z]+)$', '', regex=True)
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 8. Nettoyage final
        original_len = len(df)
        df = df.dropna()
        df = df.sort_index()
        
        # Garder seulement OHLCV (supprimer Adj Close si présent)
        keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[keep_cols]
        
        if len(df) < original_len * 0.9:  # Si on a perdu >10% des données
            print(f"⚠️ {original_len - len(df)} lignes supprimées dans {os.path.basename(csv_path)}")
            
        if df.empty:
            raise ValueError("Le DataFrame est vide après nettoyage.")
        
        if len(df) < 100:
            raise ValueError(f"Pas assez de données: {len(df)} lignes (minimum 100)")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Erreur critique chargement {csv_path}: {e}")
