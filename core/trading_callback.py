#!/usr/bin/env python3
"""
Trading Metrics Callback pour W&B
Log métriques trading détaillées pendant l'entraînement
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
    """
    
    def __init__(
        self,
        eval_env=None,
        eval_freq=10000,
        n_eval_episodes=5,
        log_actions_dist=True,
        verbose=1
    ):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_actions_dist = log_actions_dist
        
        # Buffers
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.portfolio_values = deque(maxlen=100)
        self.sharpe_history = deque(maxlen=50)
        self.actions_buffer = deque(maxlen=10000)
        
        # Compteurs
        self.n_evaluations = 0
        self.last_eval_step = 0
        
    def _on_training_start(self):
        if self.verbose > 0:
            print("\n📊 Trading Metrics Callback activé")
            print(f"   Eval freq      : {self.eval_freq:,} steps")
            print(f"   Eval episodes  : {self.n_eval_episodes}")
            print(f"   Actions logged : {self.log_actions_dist}\n")
    
    def _on_step(self) -> bool:
        # Logger actions
        if self.log_actions_dist and hasattr(self.locals, 'actions'):
            actions = self.locals.get('actions')
            if actions is not None:
                self.actions_buffer.extend(actions.flatten().tolist())
        
        # Extraire infos depuis locals
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
            
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
            # ✅ FIX : Reset et extraire obs correctement
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            
            done = False
            episode_reward = 0
            episode_length = 0
            episode_values = [10000]  # ✅ Valeur initiale par défaut
            episode_actions = []
            
            while not done:
                # Prédiction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # ✅ FIX : Assurer que action est array
                if not isinstance(action, np.ndarray):
                    action = np.array([action])
                
                episode_actions.append(action.copy())
                
                # Step
                step_result = env.step(action)
                
                # ✅ FIX : Parser résultat step correctement
                if len(step_result) == 5:  # Nouveau format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Ancien format
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                episode_length += 1
                
                # ✅ FIX : Extraire portfolio_value correctement
                portfolio_value = None
                
                if isinstance(info, dict):
                    portfolio_value = info.get('portfolio_value')
                elif isinstance(info, (list, tuple)):
                    if len(info) > 0 and isinstance(info[0], dict):
                        portfolio_value = info[0].get('portfolio_value')
                
                if portfolio_value is not None:
                    episode_values.append(float(portfolio_value))
            
            # Stocker résultats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            actions_taken.extend(episode_actions)
            
            # ✅ FIX : Vérifier qu'on a des valeurs
            if len(episode_values) > 2:
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
            # ✅ FIX : Flatten actions correctement
            actions_array = np.array([a.flatten() for a in actions_taken if a is not None])
            
            if len(actions_array) > 0:
                # Convertir actions continues en discrètes (HOLD/BUY/SELL)
                # Action > 0.33 = BUY, < -0.33 = SELL, sinon HOLD
                discrete_actions = np.zeros_like(actions_array)
                discrete_actions[actions_array > 0.33] = 1  # BUY
                discrete_actions[actions_array < -0.33] = 2  # SELL
                # 0 = HOLD par défaut
                
                # Compter par asset
                for asset_idx in range(discrete_actions.shape[1]):
                    asset_actions = discrete_actions[:, asset_idx]
                    total = len(asset_actions)
                    
                    if total > 0:
                        hold_pct = (np.sum(asset_actions == 0) / total) * 100
                        buy_pct = (np.sum(asset_actions == 1) / total) * 100
                        sell_pct = (np.sum(asset_actions == 2) / total) * 100
                        
                        metrics[f'eval/asset{asset_idx}_HOLD_pct'] = hold_pct
                        metrics[f'eval/asset{asset_idx}_BUY_pct'] = buy_pct
                        metrics[f'eval/asset{asset_idx}_SELL_pct'] = sell_pct
                
                # Global
                all_discrete = discrete_actions.flatten()
                total = len(all_discrete)
                metrics['eval/action_HOLD_pct'] = (np.sum(all_discrete == 0) / total) * 100
                metrics['eval/action_BUY_pct'] = (np.sum(all_discrete == 1) / total) * 100
                metrics['eval/action_SELL_pct'] = (np.sum(all_discrete == 2) / total) * 100
        
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
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_max_drawdown(self, portfolio_values):
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        max_dd = np.min(drawdown) * 100
        
        return float(max_dd)
    
    def _calculate_win_rate(self, portfolio_values):
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
    
    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\n✅ Trading Metrics Callback terminé")
            print(f"   Total évaluations : {self.n_evaluations}")
