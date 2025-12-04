"""
Auto-optimiseur d'hyper-paramètres via Optuna
Teste des milliers de combinaisons pour trouver les meilleures
"""

import optuna
from stable_baselines3 import PPO
from core.environment import TradingEnv

class AutoOptimizer:
    """Optimise automatiquement les hyper-paramètres du modèle"""
    
    def __init__(self, ticker, csv_path):
        self.ticker = ticker
        self.csv_path = csv_path
        
    def optimize(self, n_trials=50):
        """Lance l'optimisation (peut prendre plusieurs heures)"""
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS :")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n📊 Sharpe optimal : {study.best_value:.2f}")
        
        return study.best_params
    
    def _objective(self, trial):
        """Fonction objectif : retourne le Sharpe Ratio"""
        
        # Suggérer des hyper-paramètres
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 30),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
        }
        
        # Entraîner avec ces paramètres
        env = TradingEnv(csv_path=self.csv_path)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cuda",
            **params
        )
        
        # Entraînement court (500k steps pour tester rapidement)
        model.learn(total_timesteps=500_000)
        
        # Évaluer
        sharpe = self._evaluate(model, env)
        
        return sharpe
    
    def _evaluate(self, model, env, steps=720):
        """Évalue le modèle et retourne le Sharpe"""
        obs, _ = env.reset()
        values = []
        
        for _ in range(min(steps, len(env.df) - env.lookback_window - 1)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(int(action.item()))
            values.append(info['total_value'])
            if done or trunc: break
        
        df = pd.DataFrame({'value': values})
        df['ret'] = df['value'].pct_change().fillna(0)
        sharpe = (df['ret'].mean() / df['ret'].std()) * np.sqrt(252 * 24)
        
        return sharpe
