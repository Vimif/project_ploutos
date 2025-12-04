"""
Système de Trading Autonome - Orchestrateur Principal
Coordonne tous les modules pour un système 100% automatique
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from core.market_regime import MarketRegimeDetector
from core.asset_selector import UniversalAssetSelector
from core.auto_optimizer import AutoOptimizer
from core.universal_environment import UniversalTradingEnv

class AutonomousTradingSystem:
    """
    Système de trading 100% autonome
    
    Workflow complet:
    1. Détecte le régime de marché
    2. Sélectionne les meilleurs assets
    3. Optimise les hyper-paramètres
    4. Entraîne le modèle universel
    5. Valide et déploie
    """
    
    def __init__(self, config_path='config/autonomous_config.yaml'):
        """
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        
        # Initialiser les modules
        self.regime_detector = MarketRegimeDetector(
            reference_ticker=self.config['market']['reference_ticker']
        )
        
        self.asset_selector = UniversalAssetSelector(self.regime_detector)
        
        self.selected_assets = None
        self.best_params = None
        self.model = None
        
        # Créer dossiers nécessaires
        os.makedirs('models/autonomous', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('reports/autonomous', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def _load_config(self, config_path):
        """Charge la configuration (ou utilise défauts)"""
        
        default_config = {
            'market': {
                'reference_ticker': 'SPY',
                'lookback_days': 90
            },
            'assets': {
                'n_assets': 10,
                'min_volume': 1_000_000
            },
            'optimization': {
                'n_trials': 50,
                'n_jobs': 1,
                'timeout': None
            },
            'training': {
                'timesteps': 2_000_000,
                'n_envs': 16,
                'device': 'cuda'
            },
            'validation': {
                'min_sharpe': 1.5,
                'test_days': 90
            }
        }
        
        # Si fichier existe, charger et merger avec défauts
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge (user config override defaults)
            default_config.update(user_config)
        
        return default_config
    
    def run_full_cycle(self, skip_optimization=False):
        """
        Exécute le cycle complet d'apprentissage autonome
        
        Args:
            skip_optimization: Si True, utilise params par défaut
            
        Returns:
            dict: Résumé de l'exécution
        """
        
        print("\n" + "="*80)
        print("🤖 AUTONOMOUS TRADING SYSTEM - FULL CYCLE")
        print("="*80)
        print(f"⏰ Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'phases': {}
        }
        
        try:
            # PHASE 1 : Détection du régime
            print("\n" + "─"*80)
            print("📊 PHASE 1 : MARKET REGIME DETECTION")
            print("─"*80)
            
            regime_info = self.regime_detector.detect(
                lookback_days=self.config['market']['lookback_days']
            )
            
            results['phases']['regime_detection'] = regime_info
            
            print(f"\n✅ Régime : {regime_info['regime']} (confiance: {regime_info['confidence']:.1%})")
            
            # PHASE 2 : Sélection d'assets
            print("\n" + "─"*80)
            print("🎯 PHASE 2 : ASSET SELECTION")
            print("─"*80)
            
            self.selected_assets = self.asset_selector.select_assets(
                n_assets=self.config['assets']['n_assets'],
                lookback_days=self.config['market']['lookback_days']
            )
            
            results['phases']['asset_selection'] = {
                'selected': self.selected_assets,
                'count': len(self.selected_assets)
            }
            
            print(f"\n✅ {len(self.selected_assets)} assets sélectionnés")
            
            # Télécharger les données manquantes
            self._download_missing_data(self.selected_assets)
            
            # PHASE 3 : Optimisation des hyper-paramètres
            if not skip_optimization:
                print("\n" + "─"*80)
                print("⚙️ PHASE 3 : HYPERPARAMETER OPTIMIZATION")
                print("─"*80)
                
                optimizer = AutoOptimizer(
                    tickers=self.selected_assets[:3],  # Optimiser sur 3 assets
                    data_dir='data_cache'
                )
                
                self.best_params = optimizer.optimize(
                    n_trials=self.config['optimization']['n_trials'],
                    n_jobs=self.config['optimization']['n_jobs'],
                    timeout=self.config['optimization']['timeout']
                )
                
                results['phases']['optimization'] = {
                    'best_params': self.best_params,
                    'best_sharpe': optimizer.study.best_value
                }
            else:
                print("\n⏩ PHASE 3 : SKIPPED (utilise params par défaut)")
                self.best_params = self._default_params()
                results['phases']['optimization'] = 'skipped'
            
            # PHASE 4 : Entraînement du modèle universel
            print("\n" + "─"*80)
            print("🧠 PHASE 4 : UNIVERSAL MODEL TRAINING")
            print("─"*80)
            
            self.model = self._train_universal_model()
            
            results['phases']['training'] = {
                'timesteps': self.config['training']['timesteps'],
                'n_envs': self.config['training']['n_envs']
            }
            
            print("\n✅ Entraînement terminé")
            
            # PHASE 5 : Validation
            print("\n" + "─"*80)
            print("📊 PHASE 5 : VALIDATION")
            print("─"*80)
            
            validation_results = self._validate_model()
            
            results['phases']['validation'] = validation_results
            
            if validation_results['sharpe'] >= self.config['validation']['min_sharpe']:
                print(f"\n✅ Modèle validé (Sharpe: {validation_results['sharpe']:.2f})")
                
                # PHASE 6 : Déploiement
                print("\n" + "─"*80)
                print("🚀 PHASE 6 : DEPLOYMENT")
                print("─"*80)
                
                self._deploy_model()
                results['deployment'] = 'success'
                
            else:
                print(f"\n⚠️ Modèle rejeté (Sharpe {validation_results['sharpe']:.2f} < {self.config['validation']['min_sharpe']})")
                results['deployment'] = 'rejected'
            
            # Sauvegarder le rapport
            self._save_report(results)
            
            print("\n" + "="*80)
            print("🎉 CYCLE TERMINÉ AVEC SUCCÈS")
            print("="*80 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE : {e}")
            import traceback
            traceback.print_exc()
            
            results['error'] = str(e)
            self._save_report(results)
            
            return results
    
    def _download_missing_data(self, tickers):
        """Télécharge les données manquantes"""
        
        print("\n📥 Vérification des données...")
        
        import yfinance as yf
        
        missing = []
        for ticker in tickers:
            csv_path = f"data_cache/{ticker}.csv"
            if not os.path.exists(csv_path):
                missing.append(ticker)
        
        if len(missing) == 0:
            print("  ✅ Toutes les données disponibles")
            return
        
        print(f"  📥 Téléchargement de {len(missing)} tickers manquants...")
        
        os.makedirs('data_cache', exist_ok=True)
        
        for ticker in missing:
            try:
                print(f"    {ticker}...", end=' ', flush=True)
                
                # Télécharger
                data = yf.download(ticker, period='730d', interval='1h', progress=False)
                
                # Vérifier si données vides
                if data.empty or len(data) < 100:
                    print("❌ (pas de données)")
                    continue
                
                # Si MultiIndex (plusieurs tickers), extraire le bon
                if isinstance(data.columns, pd.MultiIndex):
                    # Chercher le ticker dans les colonnes
                    if ticker in data.columns.get_level_values(1):
                        data = data.xs(ticker, axis=1, level=1)
                    else:
                        # Prendre la première colonne disponible
                        data = data.iloc[:, :5]  # Prendre OHLCV
                        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Vérifier qu'on a bien les colonnes nécessaires
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    print(f"❌ (colonnes manquantes: {data.columns.tolist()})")
                    continue
                
                # Sauvegarder
                data.to_csv(f"data_cache/{ticker}.csv")
                print(f"✅ ({len(data)} bougies)")
                
            except Exception as e:
                print(f"❌ ({str(e)[:30]})")

        print()
        
    def _default_params(self):
        """Paramètres par défaut si skip optimization"""
        return {
            'learning_rate': 1e-4,
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'policy_kwargs': dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
        }
    
    def _train_universal_model(self):
        """Entraîne le modèle sur le portfolio sélectionné"""
        
        print(f"\n🏋️ Entraînement sur {len(self.selected_assets)} assets...")
        print(f"  Device: {self.config['training']['device']}")
        print(f"  Timesteps: {self.config['training']['timesteps']:,}")
        print(f"  Parallel Envs: {self.config['training']['n_envs']}\n")
        
        # Créer environnements parallèles
        def make_env():
            return UniversalTradingEnv(
                tickers=self.selected_assets,
                regime_detector=self.regime_detector
            )
        
        env = SubprocVecEnv([make_env for _ in range(self.config['training']['n_envs'])])
        
        # Créer modèle avec meilleurs params
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=self.config['training']['device'],
            **self.best_params
        )
        
        # Entraîner
        model.learn(
            total_timesteps=self.config['training']['timesteps'],
            progress_bar=True
        )
        
        env.close()
        
        return model
    
    def _validate_model(self):
        """Valide le modèle par backtesting"""
        
        print("\n📊 Backtesting sur 90 derniers jours...\n")
        
        # Créer env de test
        env = UniversalTradingEnv(
            tickers=self.selected_assets,
            regime_detector=self.regime_detector
        )
        
        obs, _ = env.reset()
        
        values = []
        actions_log = []
        
        test_steps = min(2160, env.min_length - env.lookback_window - 1)
        
        for step in range(test_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            
            values.append(info['total_value'])
            actions_log.append({
                'step': step,
                'positions': info['n_active_positions'],
                'value': info['total_value']
            })
            
            if done or trunc:
                break
        
        # Calculer métriques
        df = pd.DataFrame({'value': values})
        df['returns'] = df['value'].pct_change().fillna(0)
        
        initial = 10000
        final = values[-1] if values else initial
        total_return = (final - initial) / initial
        
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24) if std_ret > 0 else 0
        
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'total_return': float(total_return * 100),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown * 100),
            'final_value': float(final),
            'win_rate': float((df['returns'] > 0).mean() * 100),
            'n_trades': len(values)
        }
        
        print(f"  Return      : {results['total_return']:+.2f}%")
        print(f"  Sharpe      : {results['sharpe']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  Win Rate    : {results['win_rate']:.1f}%")
        
        return results
    
    def _deploy_model(self):
        """Sauvegarde le modèle pour déploiement"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder modèle
        model_path = f"models/autonomous/universal_{timestamp}.zip"
        self.model.save(model_path)
        print(f"  💾 Modèle : {model_path}")
        
        # Sauvegarder config associée
        config_path = f"models/autonomous/config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'assets': self.selected_assets,
                'regime': self.regime_detector.current_regime,
                'params': self.best_params
            }, f, indent=2)
        print(f"  📝 Config  : {config_path}")
        
        # Lien symbolique vers modèle "production"
        prod_path = "models/autonomous/production.zip"
        if os.path.exists(prod_path):
            os.remove(prod_path)
        os.symlink(os.path.basename(model_path), prod_path)
        print(f"  🔗 Prod    : {prod_path}")
    
    def _save_report(self, results):
        """Génère un rapport complet"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = f"reports/autonomous/report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Markdown
        md_path = f"reports/autonomous/report_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(f"# 🤖 Autonomous Trading System Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            
            for phase, data in results.get('phases', {}).items():
                f.write(f"## {phase.replace('_', ' ').title()}\n\n")
                f.write(f"``````\n\n")
        
        print(f"\n📊 Rapport sauvegardé :")
        print(f"  - {json_path}")
        print(f"  - {md_path}")

def main():
    """Point d'entrée principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Trading System')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--config', default='config/autonomous_config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Créer et lancer le système
    system = AutonomousTradingSystem(config_path=args.config)
    results = system.run_full_cycle(skip_optimization=args.skip_optimization)
    
    # Exit code selon succès
    if results.get('deployment') == 'success':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
