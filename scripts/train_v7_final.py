#!/usr/bin/env python3
"""
🚀 PLOUTOS V7.1 - FINAL TRAINING PIPELINE

Entraîne les modèles finaux V7.1 Enhanced en utilisant :
1. Les meilleures architectures trouvées par Optuna (fichiers JSON)
2. Les données financières réelles (feature engineering complet)
3. La stratégie d'entraînement avancée (Cosine Annealing, Focal Loss)

Usage:
    python scripts/train_v7_final.py

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import sys

# Add scripts directory to path to import classes
sys.path.append(str(Path(__file__).parent))
from v7_hyperparameter_optimizer_fixed import (
    calculate_momentum_features, 
    calculate_reversion_features, 
    calculate_volatility_features,
    EnhancedMomentumClassifier,
    EnhancedReversionModel,
    EnhancedVolatilityModel,
    WeightedFocalLoss
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]
MODELS_DIR = Path("models/v7_enhanced")
LOGS_DIR = Path("logs")

def load_best_params(expert_type):
    """Charge les meilleurs hyperparamètres depuis le JSON"""
    json_path = LOGS_DIR / f"v7_{expert_type}_optimization_FIXED.json"
    
    if not json_path.exists():
        logger.error(f"❌ Config introuvable: {json_path}")
        logger.error(f"👉 Lancez d'abord: python scripts/v7_hyperparameter_optimizer_fixed.py --expert {expert_type}")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"📖 Config chargée pour {expert_type.upper()}")
    logger.info(f"   Loss attendue: {data['best_loss']:.4f}")
    return data['best_params']

def prepare_data(expert_type, tickers):
    """Prépare les données d'entraînement complètes"""
    logger.info(f"🔄 Préparation données pour {expert_type}...")
    
    feature_calculator = {
        'momentum': calculate_momentum_features,
        'reversion': calculate_reversion_features,
        'volatility': calculate_volatility_features
    }
    
    target_map = {
        'momentum': lambda df: (df['Close'].pct_change(5).shift(-5) > 0).astype(int),
        'reversion': lambda df: (df['Close'].shift(-1) < df['sma_20']).astype(int),
        'volatility': lambda df: (df['volatility_20'].shift(-5) > df['volatility_20']).astype(int)
    }
    
    all_X = []
    all_y = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty or len(df) < 100: continue
            
            features = feature_calculator[expert_type](df)
            if len(features) < 50: continue
            
            target = target_map[expert_type](df.loc[features.index])
            
            # Align
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx].values
            y = target.loc[common_idx].values
            
            all_X.append(X)
            all_y.append(y)
            
        except Exception as e:
            pass
            
    X = np.vstack(all_X)
    y = np.concatenate(all_y).flatten()
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logger.info(f"✅ Données prêtes: {X.shape[0]} samples, {X.shape[1]} features")
    return torch.FloatTensor(X), torch.LongTensor(y), scaler

def train_final_expert(expert_type):
    """Entraîne et sauvegarde un expert final"""
    
    # 1. Load Params
    params = load_best_params(expert_type)
    
    # 2. Data
    X, y, scaler = prepare_data(expert_type, TICKERS)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    
    if expert_type == 'momentum':
        hidden_dims = [params['hidden1'], params['hidden2'], params['hidden3']]
        model = EnhancedMomentumClassifier(input_dim, hidden_dims, params['dropout'])
    elif expert_type == 'reversion':
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedReversionModel(input_dim, hidden_dims, params['dropout'])
    else:
        hidden_dims = [params['hidden1'], params['hidden2']]
        model = EnhancedVolatilityModel(input_dim, hidden_dims, params['dropout'])
    
    model = model.to(device)
    
    # 4. Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = WeightedFocalLoss(torch.tensor([0.4, 0.6]).to(device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
    
    # 5. Training Loop
    logger.info(f"🚀 Démarrage entraînement {expert_type.upper()} ({params['epochs']} epochs)...")
    
    model.train()
    for epoch in range(params['epochs']):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            logger.info(f"   Epoch {epoch+1}/{params['epochs']} - Loss: {avg_loss:.4f}")
            
    # 6. Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Model
    model_path = MODELS_DIR / f"v7_1_{expert_type}.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save Scaler (IMPORTANT for inference)
    import pickle
    scaler_path = MODELS_DIR / f"v7_1_{expert_type}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    logger.info(f"💾 Modèle sauvegardé: {model_path}")
    logger.info(f"💾 Scaler sauvegardé: {scaler_path}")
    logger.info("-" * 50)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 PLOUTOS V7.1 - FINAL MODEL TRAINING")
    print("="*60 + "\n")
    
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU")
        
    print("\n")
    
    try:
        train_final_expert('momentum')
        train_final_expert('reversion')
        train_final_expert('volatility')
        
        print("\n" + "="*60)
        print("✅ TOUS LES MODÈLES V7.1 SONT PRÊTS !")
        print(f"📂 Dossier: {MODELS_DIR}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur fatale: {e}")
        sys.exit(1)
