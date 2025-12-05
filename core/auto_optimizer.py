"""
Auto-optimiseur d'hyper-paramètres via Optuna
Teste des milliers de combinaisons pour trouver les meilleures
"""

import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.universal_environment import TradingEnv
import warnings
warnings.filterwarnings('ignore')

class AutoOptimizer:
    """Optimise automatiquement les hyper-paramètres du modèle"""
    
    def __init__(self, tickers, data_dir='data_cache'):
        """
        Args:
            tickers: Liste des tickers à utiliser pour l'optimisation
            data_dir: Répertoire contenant les CSV
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.data_dir = data_dir
        self.best_params = None
        self.study = None
        
    def optimize(self, n_trials=50, n_jobs=1, timeout=None):
        """
        Lance l'optimisation Bayésienne
        
        Args:
            n_trials: Nombre d'essais
            n_jobs: Processus parallèles (-1 = tous les CPUs)
            timeout: Timeout en secondes (None = illimité)
            
        Returns:
            dict: Meilleurs hyper-paramètres trouvés
        """
        
        print(f"\n⚙️ OPTIMISATION AUTOMATIQUE")
        print(f"  Trials: {n_trials}")
        print(f"  Assets: {', '.join(self.tickers[:5])}{'...' if len(self.tickers) > 5 else ''}")
        print(f"  Device: CUDA\n")
        
        # Créer l'étude Optuna
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Lancer l'optimisation
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Résultats
        self.best_params = self.study.best_params
        
        print(f"\n{'='*70}")
        print(f"🏆 MEILLEURS PARAMÈTRES TROUVÉS")
        print(f"{'='*70}")
        for key, value in self.best_params.items():
            print(f"  {key:20s}: {value}")
        print(f"\n  📊 Sharpe optimal : {self.study.best_value:.3f}")
        print(f"  🎯 Trial gagnant : #{self.study.best_trial.number}")
        print(f"{'='*70}\n")
        
        return self.best_params
    
    def _objective(self, trial):
        """
        Fonction objectif pour Optuna
        Retourne le Sharpe Ratio moyen sur tous les tickers
        """
        print(f"\n🔬 Trial #{trial.number} démarré...")
        
        # Suggérer des hyper-paramètres
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 30),
            'gamma': trial.suggest_float('gamma', 0.90, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.80, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0)
        }
        
        # Architecture réseau
        net_arch_size = trial.suggest_categorical('net_arch', [256, 512, 1024])
        params['policy_kwargs'] = dict(
            net_arch=dict(pi=[net_arch_size]*3, vf=[net_arch_size]*3)
        )
        
        # Évaluer sur plusieurs tickers
        sharpes = []
        
        for ticker in self.tickers[:3]:  # Limiter à 3 pour rapidité
            try:
                sharpe = self._evaluate_params(params, ticker)
                if not np.isnan(sharpe):
                    sharpes.append(sharpe)
            except Exception as e:
                print(f"    ⚠️ Trial {trial.number} échec sur {ticker}: {str(e)[:50]}")
                continue
        
        # Si aucun succès, retourner -inf pour éliminer ce trial
        if len(sharpes) == 0:
            return float('-inf')
        
        # Retourner moyenne des Sharpes
        mean_sharpe = np.mean(sharpes)
        
        # Pruning : Arrêter tôt si mauvais
        trial.report(mean_sharpe, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"✅ Trial #{trial.number} : Sharpe = {mean_sharpe:.3f}")
        
        return mean_sharpe
    
    def _evaluate_params(self, params, ticker):
        """
        Évalue un set d'hyper-paramètres sur un ticker
        
        Returns:
            float: Sharpe ratio
        """
        
        csv_path = f"{self.data_dir}/{ticker}.csv"
        
        # Créer env
        env = DummyVecEnv([lambda: TradingEnv(csv_path=csv_path)])
        
        # Entraîner modèle
        model = PPO("MlpPolicy", env, verbose=0, device="cuda", **params)
        model.learn(total_timesteps=100_000)  # Court pour rapidité
        
        # Backtest
        sharpe = self._quick_backtest(model, csv_path)
        
        env.close()
        
        return sharpe
    
    def _quick_backtest(self, model, csv_path, steps=500):
        """Backtest rapide pour obtenir le Sharpe"""
        
        env = TradingEnv(csv_path=csv_path)
        obs, _ = env.reset()
        values = []
        
        for _ in range(min(steps, len(env.df) - env.lookback_window - 1)):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            obs, _, done, trunc, info = env.step(action_int)
            values.append(info['total_value'])
            if done or trunc:
                break
        
        if len(values) < 10:
            return 0.0
        
        df = pd.DataFrame({'value': values})
        df['ret'] = df['value'].pct_change().fillna(0)
        
        mean_ret = df['ret'].mean()
        std_ret = df['ret'].std()
        
        if std_ret == 0:
            return 0.0
        
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24)
        
        return sharpe
    
    def plot_optimization_history(self, filepath='reports/optimization_history.png'):
        """Visualise l'historique d'optimisation"""
        
        if self.study is None:
            print("⚠️ Aucune étude à visualiser")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150)
            print(f"📊 Graphique sauvegardé : {filepath}")
            
        except ImportError:
            print("⚠️ Matplotlib non installé, skip visualisation")
