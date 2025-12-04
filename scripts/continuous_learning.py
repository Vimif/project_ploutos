"""
Système d'apprentissage continu
Ré-entraîne le modèle automatiquement avec les nouvelles données
Lance via CRON tous les dimanches
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from core.universal_environment import UniversalTradingEnv
from core.market_regime import MarketRegimeDetector
import json

class ContinuousLearningSystem:
    """Gère l'apprentissage continu du modèle"""
    
    def __init__(self):
        self.model_path = "models/autonomous/production.zip"
        self.config_path = "models/autonomous/config_latest.json"
        self.data_dir = "data_cache"
        self.backup_dir = "models/backups"
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def run_weekly_update(self):
        """Cycle hebdomadaire d'apprentissage continu"""
        
        print("\n" + "="*80)
        print("🔄 CONTINUOUS LEARNING - WEEKLY UPDATE")
        print("="*80)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Charger config actuelle
        if not os.path.exists(self.config_path):
            print("❌ Pas de configuration trouvée, lancer autonomous_system.py d'abord")
            return
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        tickers = config['assets']
        print(f"📊 Assets actuels : {', '.join(tickers)}\n")
        
        # 2. Télécharger nouvelles données
        print("📥 Mise à jour des données...")
        new_data_count = self._update_market_data(tickers)
        
        if new_data_count < 50:
            print(f"⚠️ Seulement {new_data_count} nouvelles bougies, skip fine-tuning")
            return
        
        # 3. Backup du modèle actuel
        self._backup_current_model()
        
        # 4. Fine-tuning
        print("\n🧠 Fine-tuning du modèle...")
        candidate_model = self._fine_tune_model(tickers)
        
        # 5. Validation
        print("\n📊 Validation du nouveau modèle...")
        is_better = self._validate_and_deploy(candidate_model, tickers)
        
        if is_better:
            print("\n✅ Mise à jour déployée avec succès")
        else:
            print("\n⚠️ Ancien modèle conservé (meilleur)")
        
        print("\n" + "="*80)
    
    def _update_market_data(self, tickers):
        """Télécharge les 7 derniers jours de données"""
        
        total_new = 0
        
        for ticker in tickers:
            csv_path = f"{self.data_dir}/{ticker}.csv"
            
            try:
                # Charger anciennes données
                df_old = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                last_date = df_old.index[-1]
                
                # Télécharger depuis last_date
                df_new = yf.download(
                    ticker, 
                    start=last_date, 
                    interval='1h', 
                    progress=False
                )
                
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new = df_new.xs(ticker, axis=1, level=1)
                
                # Fusionner et supprimer doublons
                df_combined = pd.concat([df_old, df_new]).drop_duplicates()
                
                # Garder 730 derniers jours (rolling window)
                cutoff_date = datetime.now() - timedelta(days=730)
                df_combined = df_combined[df_combined.index >= cutoff_date]
                
                # Sauvegarder
                df_combined.to_csv(csv_path)
                
                new_rows = len(df_new)
                total_new += new_rows
                
                print(f"  {ticker}: +{new_rows} bougies (total: {len(df_combined)})")
                
            except Exception as e:
                print(f"  ⚠️ {ticker}: {str(e)[:50]}")
        
        print(f"\n✅ Total nouvelles données : {total_new} bougies")
        return total_new
    
    def _backup_current_model(self):
        """Sauvegarde le modèle actuel avant fine-tuning"""
        
        if not os.path.exists(self.model_path):
            print("⚠️ Pas de modèle production à backuper")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.backup_dir}/production_backup_{timestamp}.zip"
        
        os.system(f"cp {self.model_path} {backup_path}")
        
        print(f"💾 Backup : {backup_path}")
    
    def _fine_tune_model(self, tickers):
        """Fine-tune le modèle existant"""
        
        regime_detector = MarketRegimeDetector()
        
        # Créer env
        env = UniversalTradingEnv(
            tickers=tickers,
            regime_detector=regime_detector
        )
        
        # Charger modèle existant
        if os.path.exists(self.model_path):
            model = PPO.load(self.model_path, env=env)
            print("  ✅ Modèle chargé")
        else:
            print("  ⚠️ Création nouveau modèle")
            model = PPO("MlpPolicy", env, verbose=1, device="cuda")
        
        # Fine-tuning avec learning rate réduit
        model.learning_rate = 1e-5  # 10x plus petit
        
        print("  🏋️ Fine-tuning 500k steps...")
        model.learn(
            total_timesteps=500_000,
            progress_bar=True,
            reset_num_timesteps=False  # Garde les stats
        )
        
        # Sauvegarder candidat
        candidate_path = self.model_path.replace('.zip', '_candidate.zip')
        model.save(candidate_path)
        
        print(f"  ✅ Candidat sauvegardé : {candidate_path}")
        
        return candidate_path
    
    def _validate_and_deploy(self, candidate_path, tickers):
        """Compare ancien vs nouveau modèle"""
        
        regime_detector = MarketRegimeDetector()
        
        # Backtest ancien modèle
        if os.path.exists(self.model_path):
            sharpe_old = self._quick_backtest(self.model_path, tickers, regime_detector)
            print(f"  Sharpe ancien : {sharpe_old:.2f}")
        else:
            sharpe_old = 0
            print("  Pas de modèle ancien")
        
        # Backtest nouveau modèle
        sharpe_new = self._quick_backtest(candidate_path, tickers, regime_detector)
        print(f"  Sharpe nouveau: {sharpe_new:.2f}")
        
        # Décision (tolérance -5%)
        if sharpe_new >= sharpe_old * 0.95:
            print("\n  ✅ Nouveau modèle MEILLEUR → Déploiement")
            os.system(f"cp {candidate_path} {self.model_path}")
            
            # Log dans PostgreSQL (optionnel)
            self._log_deployment(sharpe_new, sharpe_old)
            
            return True
        else:
            print(f"\n  ❌ Nouveau modèle MOINS BON → Conserve ancien")
            return False
    
    def _quick_backtest(self, model_path, tickers, regime_detector, steps=720):
        """Backtest rapide pour Sharpe"""
        
        model = PPO.load(model_path)
        env = UniversalTradingEnv(tickers=tickers, regime_detector=regime_detector)
        obs, _ = env.reset()
        
        values = []
        
        for _ in range(min(steps, env.min_length - env.lookback_window - 1)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
            values.append(info['total_value'])
            if done or trunc:
                break
        
        if len(values) < 10:
            return 0.0
        
        df = pd.DataFrame({'value': values})
        df['ret'] = df['value'].pct_change().fillna(0)
        
        mean = df['ret'].mean()
        std = df['ret'].std()
        
        sharpe = (mean / std) * (252 * 24) ** 0.5 if std > 0 else 0
        
        return sharpe
    
    def _log_deployment(self, sharpe_new, sharpe_old):
        """Log dans PostgreSQL (optionnel)"""
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                dbname="ploutos",
                user="ploutos",
                password="",
                host="localhost"
            )
            
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_versions (timestamp, sharpe_ratio, previous_sharpe, model_path, is_deployed)
                VALUES (NOW(), %s, %s, %s, TRUE)
            """, (sharpe_new, sharpe_old, self.model_path))
            
            conn.commit()
            conn.close()
            
            print("  📊 Logged to PostgreSQL")
            
        except Exception as e:
            print(f"  ⚠️ PostgreSQL log failed: {e}")

def main():
    system = ContinuousLearningSystem()
    system.run_weekly_update()

if __name__ == "__main__":
    main()
