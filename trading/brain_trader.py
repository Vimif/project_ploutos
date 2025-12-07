#!/usr/bin/env python3
"""
🧠 BRAIN TRADER - Adaptateur Modèle V2

Wrapper compatible avec l'ancien système LiveTrader
Charge et utilise le nouveau modèle UniversalTradingEnvV2

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import logging

try:
    from core.universal_environment_v2 import UniversalTradingEnvV2
except ImportError:
    print("⚠️  UniversalEnvironmentV2 non trouvé, utilise V1")
    from core.universal_environment import UniversalTradingEnv as UniversalTradingEnvV2

logger = logging.getLogger(__name__)

class BrainTrader:
    """
    Wrapper pour compatibilité avec LiveTrader
    Charge le modèle V2 et génère des prédictions
    """
    
    def __init__(self, capital=100000, paper_trading=True, model_path=None):
        """
        Args:
            capital: Capital initial (non utilisé ici, pour compatibilité)
            paper_trading: Mode paper trading (non utilisé ici)
            model_path: Chemin vers le modèle (défaut: models/autonomous/production.zip)
        """
        self.capital = capital
        self.paper_trading = paper_trading
        
        # Charger le modèle
        if model_path is None:
            # Essayer plusieurs emplacements
            possible_paths = [
                'models/autonomous/production.zip',
                'models/autonomous/final_model.zip',
                'models/ploutos_v2_production.zip',
                '/root/ploutos/project_ploutos/models/autonomous/production.zip'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"❌ Aucun modèle trouvé. Vérifié: {possible_paths}"
                )
        
        logger.info(f"🧠 Chargement modèle: {model_path}")
        
        try:
            self.model = PPO.load(model_path)
            logger.info("✅ Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
        
        # Charger config modèle si disponible
        config_path = model_path.replace('.zip', '.json')
        self.model_config = {}
        
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            logger.info(f"✅ Config chargée: {len(self.model_config.get('tickers', []))} tickers")
        
        # Tickers par défaut (si pas de config)
        self.tickers = self.model_config.get('tickers', [
            'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
            'SPY', 'QQQ', 'VOO', 'XLE', 'XLF'
        ])
        
        logger.info(f"🎯 Tickers actifs: {', '.join(self.tickers)}")
    
    def predict_all(self):
        """
        Génère des prédictions pour tous les tickers
        
        Returns:
            dict: Prédictions par secteur (pour compatibilité avec LiveTrader)
                  Format: {'tech': [...], 'indices': [...], 'energy': [...]}
        """
        logger.info("🔮 Génération prédictions...")
        
        # Catégoriser les tickers
        categories = {
            'tech': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
            'indices': ['SPY', 'QQQ', 'VOO', 'VTI'],
            'energy': ['XLE', 'XOM', 'CVX'],
            'finance': ['XLF', 'JPM', 'BAC']
        }
        
        predictions = {cat: [] for cat in categories.keys()}
        
        # Télécharger données récentes
        data = self._load_recent_data(self.tickers)
        
        if not data:
            logger.warning("⚠️  Aucune donnée disponible")
            return predictions
        
        # Créer environnement temporaire pour observations
        try:
            env = UniversalTradingEnvV2(
                data=data,
                initial_balance=self.capital,
                commission=0.0001,
                max_steps=100,
                buy_pct=0.2
            )
            
            obs, _ = env.reset()
            
            # Prédire actions
            actions, _ = self.model.predict(obs, deterministic=True)
            
            # Convertir actions en prédictions par ticker
            for i, ticker in enumerate(env.tickers):
                action = int(actions[i])
                
                # Mapper action à string
                action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
                
                # Catégoriser
                category = 'other'
                for cat, tickers_list in categories.items():
                    if ticker in tickers_list:
                        category = cat
                        break
                
                # Ajouter prédiction
                pred = {
                    'ticker': ticker,
                    'action': action_str,
                    'confidence': 0.8,  # Placeholder
                    'capital': self.capital * 0.1 if action_str == 'BUY' else 0
                }
                
                if category != 'other':
                    predictions[category].append(pred)
            
            logger.info(f"✅ {sum(len(v) for v in predictions.values())} prédictions générées")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération prédictions: {e}")
            import traceback
            traceback.print_exc()
        
        return predictions
    
    def _load_recent_data(self, tickers, days=30):
        """
        Charge les données récentes pour les tickers
        
        Args:
            tickers: Liste de tickers
            days: Nombre de jours à charger
        
        Returns:
            dict: Données par ticker
        """
        import yfinance as yf
        
        data = {}
        end = datetime.now()
        start = end - timedelta(days=days)
        
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval='1d',
                    progress=False
                )
                
                if df.empty or len(df) < 10:
                    logger.warning(f"⚠️  {ticker}: Données insuffisantes")
                    continue
                
                # Flatten MultiIndex si nécessaire
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                data[ticker] = df
                
            except Exception as e:
                logger.warning(f"⚠️  {ticker}: Erreur téléchargement ({e})")
        
        return data

# Alias pour compatibilité
Brain = BrainTrader
