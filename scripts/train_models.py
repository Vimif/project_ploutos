#!/usr/bin/env python3
# scripts/train_models.py
"""Script d'entraînement des modèles"""

import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from training.trainer import Trainer
from config.tickers import ALL_TICKERS, SECTORS

def main():
    parser = argparse.ArgumentParser(description='Entraîner les modèles de trading')
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Tickers à entraîner (ex: MSFT AAPL)'
    )
    
    parser.add_argument(
        '--sector',
        choices=list(SECTORS.keys()),
        help='Entraîner un secteur entier (crypto, tech, etc.)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=5_000_000,
        help='Nombre de timesteps'
    )
    
    args = parser.parse_args()
    
    # Déterminer les tickers
    if args.sector:
        tickers = SECTORS[args.sector]['tickers']
        print(f"📂 Secteur: {args.sector}")
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = ALL_TICKERS
    
    print(f"🎯 Tickers: {', '.join(tickers)}")
    print(f"⏱️  Timesteps: {args.timesteps:,}")
    
    # Confirmation
    response = input("\n▶️  Continuer? (y/n): ")
    if response.lower() != 'y':
        print("❌ Annulé")
        return
    
    # Entraîner
    trainer = Trainer()
    trainer.config['total_timesteps'] = args.timesteps
    trainer.train_all(tickers)

if __name__ == "__main__":
    main()
