import pandas as pd
import numpy as np
import os

def load_market_data(csv_path):
    """
    Charge et nettoie les données de marché de manière ROBUSTE
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    
    try:
        # Lecture brute
        df = pd.read_csv(csv_path, index_col=0)
        
        # 1. Nettoyage Index (Dates)
        df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # 2. Nettoyage Colonnes (Conversion numérique forcée)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                # Convertir en numérique, remplacer erreurs par NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Supprimer les lignes vides ou corrompues
        original_len = len(df)
        df = df.dropna()
        
        if len(df) < original_len:
            print(f"⚠️ {original_len - len(df)} lignes corrompues supprimées")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Erreur critique chargement {csv_path}: {e}")
