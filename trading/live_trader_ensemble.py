import sys
import os
import time
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading.ensemble_trader import EnsembleTrader
from core.environment import TradingEnv

def run_live_ensemble(ticker, check_interval_minutes=60):
    """Trading live avec ensemble de modèles"""
    
    print("="*70)
    print(f"🤖 LIVE TRADING ENSEMBLE : {ticker}")
    print("="*70)
    
    # Charger l'ensemble
    ensemble = EnsembleTrader(ticker)
    
    print(f"\n⏱️  Intervalle de check : {check_interval_minutes} minutes\n")
    
    while True:
        try:
            # Télécharger données récentes
            print(f"📥 [{pd.Timestamp.now()}] Téléchargement données...")
            df = yf.download(ticker, period="5d", interval="1h", progress=False)
            
            if df.empty:
                print("❌ Pas de données reçues, skip")
                time.sleep(check_interval_minutes * 60)
                continue
            
            # Sauvegarder temporairement
            temp_csv = f"temp_{ticker}.csv"
            df.to_csv(temp_csv)
            
            # Créer environnement
            env = TradingEnv(csv_path=temp_csv)
            obs, _ = env.reset()
            
            # PRÉDICTION ENSEMBLE
            action, confidence = ensemble.predict(obs)
            
            # Interpréter l'action
            actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            decision = actions_map[action]
            
            # Affichage
            current_price = df['Close'].iloc[-1]
            print(f"\n{'='*70}")
            print(f"💰 Prix actuel : ${current_price:.2f}")
            print(f"🎯 DÉCISION : {decision} (confiance: {confidence*100:.1f}%)")
            print(f"{'='*70}\n")
            
            # Voir détails votes
            votes = ensemble.predict_all_votes(obs)
            print("📊 Détail des votes :")
            for v in votes:
                print(f"   Modèle {v['model_id']} → {actions_map[v['action']]}")
            
            # Nettoyage
            env.close()
            os.remove(temp_csv)
            
            # Attendre prochain cycle
            print(f"\n⏸️  Pause {check_interval_minutes} minutes...\n")
            time.sleep(check_interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n🛑 Arrêt manuel du trading")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")
            time.sleep(60)  # Pause courte avant retry

if __name__ == "__main__":
    run_live_ensemble("NVDA", check_interval_minutes=60)
