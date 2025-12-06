#!/usr/bin/env python3
"""
Trading Metrics Callback pour W&B
Log métriques trading détaillées pendant l'entraînement

Métriques loggées :
- Sharpe Ratio en temps réel
- Max Drawdown
- Win Rate
- Profit Factor
- Actions distribution (Long/Short/Hold)
- Portfolio value progression
"""

import numpy as np
import pandas as pd
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class TradingMetricsCallback(BaseCallback):
    """
    Callback pour logger métriques trading dans W&B
    
    Usage:
        callback = TradingMetricsCallback(
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=5
        )
        
        model.learn(
            total_timesteps=1_000_000,
            callback=callback
        )
    """
    
    def __init__(
        self,
        eval_env=None,
        eval_freq=10000,
        n_eval_episodes=5,
        log_actions_dist=True,
        verbose=1
    ):
        """
        Args:
            eval_env: Environnement pour évaluation (si None, utilise training env)
            eval_freq: Fréquence d'évaluation (en steps)
            n_eval_episodes: Nombre d'épisodes d'évaluation
            log_actions_dist: Logger distribution des actions
            verbose: Niveau de verbosité (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_actions_dist = log_actions_dist
        
        # Buffers pour métriques en temps réel
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.portfolio_values = deque(maxlen=100)
        self.sharpe_history = deque(maxlen=50)
        self.actions_buffer = deque(maxlen=10000)
        
        # Compteurs
        self.n_evaluations = 0
        self.last_eval_step = 0
        
    def _on_training_start(self):
        """Appelé au début de l'entraînement"""
        if self.verbose > 0:
            print("\n📊 Trading Metrics Callback activé")
            print(f"   Eval freq      : {self.eval_freq:,} steps")
            print(f"   Eval episodes  : {self.n_eval_episodes}")
            print(f"   Actions logged : {self.log_actions_dist}\n")
    
    def _on_step(self) -> bool:
        """
        Appelé à chaque step
        
        Returns:
            bool: True pour continuer l'entraînement
        """
        
        # Logger actions si demandé
        if self.log_actions_dist and hasattr(self.locals, 'actions'):
            actions = self.locals.get('actions')
            if actions is not None:
                self.actions_buffer.extend(actions.flatten().tolist())
        
        # Extraire infos depuis locals
        infos = self.locals.get('infos', [])
        
        for info in infos:
            # Portfolio value
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
            
            # Episode terminé
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_returns.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        
        # Évaluation périodique
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_and_log()
            self.last_eval_step = self.num_timesteps
        
        return True
    
    def _evaluate_and_log(self):
        """
        Évalue le modèle et log métriques dans W&B
        """
        
        self.n_evaluations += 1
        
        # Choisir environnement
        env = self.eval_env if self.eval_env is not None else self.training_env
        
        # Métriques agrégées
        episode_rewards = []
        episode_lengths = []
        portfolio_values_list = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        actions_taken = []
        
        # Lancer épisodes d'évaluation
        for ep in range(self.n_eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Unpack si nouveau format Gym
            
            done = False
            episode_reward = 0
            episode_length = 0
            episode_values = []
            episode_actions = []
            
            while not done:
                # Prédiction
                action, _ = self.model.predict(obs, deterministic=True)
                episode_actions.append(action)
                
                # Step
                step_result = env.step(action)
                
                if len(step_result) == 5:  # Nouveau format (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Ancien format (obs, reward, done, info)
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                episode_length += 1
                
                # Extraire info
                if isinstance(info, dict):
                    if 'portfolio_value' in info:
                        episode_values.append(info['portfolio_value'])
                elif isinstance(info, (list, tuple)) and len(info) > 0:
                    if isinstance(info[0], dict) and 'portfolio_value' in info[0]:
                        episode_values.append(info[0]['portfolio_value'])
            
            # Stocker résultats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            actions_taken.extend(episode_actions)
            
            if len(episode_values) > 10:
                portfolio_values_list.append(episode_values)
                
                # Calculer métriques
                sharpe = self._calculate_sharpe(episode_values)
                max_dd = self._calculate_max_drawdown(episode_values)
                win_rate = self._calculate_win_rate(episode_values)
                
                sharpe_ratios.append(sharpe)
                max_drawdowns.append(max_dd)
                win_rates.append(win_rate)
        
        # Agréger métriques
        metrics = {
            # Récompenses
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/min_reward': np.min(episode_rewards),
            'eval/max_reward': np.max(episode_rewards),
            
            # Longueurs
            'eval/mean_ep_length': np.mean(episode_lengths),
            
            # Trading metrics
            'eval/mean_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'eval/mean_max_dd': np.mean(max_drawdowns) if max_drawdowns else 0,
            'eval/mean_win_rate': np.mean(win_rates) if win_rates else 0,
        }
        
        # Portfolio final moyen
        if portfolio_values_list:
            final_values = [vals[-1] for vals in portfolio_values_list]
            metrics['eval/mean_final_portfolio'] = np.mean(final_values)
            metrics['eval/best_final_portfolio'] = np.max(final_values)
            metrics['eval/worst_final_portfolio'] = np.min(final_values)
        
        # Distribution actions
        if actions_taken and self.log_actions_dist:
            actions_array = np.array(actions_taken).flatten()
            
            # Compter actions (supposant action space discret avec 3 actions)
            if len(actions_array) > 0:
                unique, counts = np.unique(actions_array, return_counts=True)
                total = len(actions_array)
                
                for action_id, count in zip(unique, counts):
                    action_name = self._get_action_name(int(action_id))
                    metrics[f'eval/action_{action_name}_pct'] = (count / total) * 100
        
        # Distribution actions buffer (training)
        if len(self.actions_buffer) > 0 and self.log_actions_dist:
            actions_array = np.array(list(self.actions_buffer))
            unique, counts = np.unique(actions_array, return_counts=True)
            total = len(actions_array)
            
            for action_id, count in zip(unique, counts):
                action_name = self._get_action_name(int(action_id))
                metrics[f'train/action_{action_name}_pct'] = (count / total) * 100
        
        # Profit Factor
        if portfolio_values_list:
            profit_factor = self._calculate_profit_factor(portfolio_values_list)
            if profit_factor is not None:
                metrics['eval/profit_factor'] = profit_factor
        
        # Logger dans W&B
        if wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)
        
        # Print si verbose
        if self.verbose > 0:
            print(f"\n📊 Évaluation #{self.n_evaluations} (step {self.num_timesteps:,})")
            print(f"   Reward    : {metrics['eval/mean_reward']:.3f} ± {metrics['eval/std_reward']:.3f}")
            print(f"   Sharpe    : {metrics['eval/mean_sharpe']:.3f}")
            print(f"   Max DD    : {metrics['eval/mean_max_dd']:.2f}%")
            print(f"   Win Rate  : {metrics['eval/mean_win_rate']:.1f}%")
            
            if 'eval/mean_final_portfolio' in metrics:
                print(f"   Portfolio : ${metrics['eval/mean_final_portfolio']:.0f}")
    
    def _calculate_sharpe(self, portfolio_values):
        """
        Calcule Sharpe Ratio depuis portfolio values
        
        Args:
            portfolio_values: Liste des valeurs du portfolio
            
        Returns:
            float: Sharpe Ratio annualisé
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        # Calculer returns
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Sharpe annualisé (252 jours de trading)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, portfolio_values):
        """
        Calcule Maximum Drawdown
        
        Args:
            portfolio_values: Liste des valeurs du portfolio
            
        Returns:
            float: Max Drawdown en %
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        
        max_dd = np.min(drawdown) * 100  # En pourcentage
        
        return float(max_dd)
    
    def _calculate_win_rate(self, portfolio_values):
        """
        Calcule Win Rate (% de trades gagnants)
        
        Args:
            portfolio_values: Liste des valeurs du portfolio
            
        Returns:
            float: Win rate en %
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        
        win_rate = (winning_trades / total_trades) * 100
        
        return float(win_rate)
    
    def _calculate_profit_factor(self, portfolio_values_list):
        """
        Calcule Profit Factor (gains / pertes)
        
        Args:
            portfolio_values_list: Liste de listes de portfolio values
            
        Returns:
            float: Profit Factor
        """
        all_returns = []
        
        for values in portfolio_values_list:
            if len(values) > 1:
                values_array = np.array(values)
                returns = np.diff(values_array) / values_array[:-1]
                all_returns.extend(returns)
        
        if len(all_returns) == 0:
            return None
        
        all_returns = np.array(all_returns)
        
        gross_profit = np.sum(all_returns[all_returns > 0])
        gross_loss = np.abs(np.sum(all_returns[all_returns < 0]))
        
        if gross_loss == 0:
            return None
        
        profit_factor = gross_profit / gross_loss
        
        return float(profit_factor)
    
    def _get_action_name(self, action_id):
        """
        Convertit action ID en nom lisible
        
        Args:
            action_id: ID de l'action (0, 1, 2)
            
        Returns:
            str: Nom de l'action
        """
        action_names = {
            0: 'HOLD',
            1: 'BUY',
            2: 'SELL'
        }
        return action_names.get(action_id, f'ACTION_{action_id}')
    
    def _on_training_end(self):
        """
        Appelé à la fin de l'entraînement
        """
        if self.verbose > 0:
            print(f"\n✅ Trading Metrics Callback terminé")
            print(f"   Total évaluations : {self.n_evaluations}")

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("📊 TRADING METRICS CALLBACK")
    print("="*80 + "\n")
    
    print("📝 Usage dans train_curriculum.py :\n")
    print("""from core.trading_callback import TradingMetricsCallback

# Créer callback
trading_callback = TradingMetricsCallback(
    eval_env=env,              # Environnement d'évaluation
    eval_freq=10000,           # Évaluer toutes les 10k steps
    n_eval_episodes=5,         # 5 épisodes par évaluation
    log_actions_dist=True,     # Logger distribution actions
    verbose=1
)

# Combiner avec autres callbacks
callback = CallbackList([
    checkpoint_callback,
    wandb_callback,
    trading_callback  # ✅ Ajouté
])

# Entraîner
model.learn(
    total_timesteps=1_000_000,
    callback=callback
)
""")
    
    print("\n📊 Métriques loggées dans W&B :")
    print("   eval/mean_reward")
    print("   eval/mean_sharpe")
    print("   eval/mean_max_dd")
    print("   eval/mean_win_rate")
    print("   eval/mean_final_portfolio")
    print("   eval/profit_factor")
    print("   eval/action_HOLD_pct")
    print("   eval/action_BUY_pct")
    print("   eval/action_SELL_pct")
    print("\n")
