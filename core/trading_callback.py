#!/usr/bin/env python3
"""
Trading Metrics Callback pour W&B
Log métriques trading détaillées pendant l'entraînement

⚠️ VERSION FIXED: Correction bug métriques 10,000%
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
            episode_values = [100000]  # ✅ Valeur initiale par défaut
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
        
        # ✅★★★ FIX DISTRIBUTION ACTIONS (BUG CRITIQUE) ★★★
        if actions_taken and self.log_actions_dist:
            try:
                # Flatten actions correctement
                actions_array = np.concatenate([a.flatten() for a in actions_taken if a is not None])
                
                if len(actions_array) > 0:
                    # Convertir actions continues en discrètes (HOLD/BUY/SELL)
                    # Action > 0.33 = BUY, < -0.33 = SELL, sinon HOLD
                    
                    # Compter pour CHAQUE action (pas par asset)
                    n_buy = np.sum(actions_array > 0.33)
                    n_sell = np.sum(actions_array < -0.33)
                    n_hold = len(actions_array) - n_buy - n_sell
                    total_actions = len(actions_array)
                    
                    # ✅ FIX: Normaliser à 100% (pas 10,000%)
                    if total_actions > 0:
                        metrics['eval/action_HOLD_pct'] = (n_hold / total_actions) * 100
                        metrics['eval/action_BUY_pct'] = (n_buy / total_actions) * 100
                        metrics['eval/action_SELL_pct'] = (n_sell / total_actions) * 100
                    
                    # Par asset (si multi-asset)
                    actions_matrix = np.array([a.flatten() for a in actions_taken if a is not None])
                    
                    if len(actions_matrix.shape) == 2 and actions_matrix.shape[1] > 1:
                        for asset_idx in range(actions_matrix.shape[1]):
                            asset_actions = actions_matrix[:, asset_idx]
                            
                            n_buy_asset = np.sum(asset_actions > 0.33)
                            n_sell_asset = np.sum(asset_actions < -0.33)
                            n_hold_asset = len(asset_actions) - n_buy_asset - n_sell_asset
                            total_asset = len(asset_actions)
                            
                            if total_asset > 0:
                                metrics[f'eval/asset{asset_idx}_HOLD_pct'] = (n_hold_asset / total_asset) * 100
                                metrics[f'eval/asset{asset_idx}_BUY_pct'] = (n_buy_asset / total_asset) * 100
                                metrics[f'eval/asset{asset_idx}_SELL_pct'] = (n_sell_asset / total_asset) * 100
            
            except Exception as e:
                if self.verbose > 0:
                    print(f"\n⚠️  Erreur calcul distribution actions: {e}")
        
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
            
            # ✅ Vérifier cohérence distribution
            if 'eval/action_HOLD_pct' in metrics:
                total_pct = (
                    metrics['eval/action_HOLD_pct'] + 
                    metrics['eval/action_BUY_pct'] + 
                    metrics['eval/action_SELL_pct']
                )
                print(f"   Actions   : HOLD {metrics['eval/action_HOLD_pct']:.1f}% | "
                      f"BUY {metrics['eval/action_BUY_pct']:.1f}% | "
                      f"SELL {metrics['eval/action_SELL_pct']:.1f}% (Total: {total_pct:.1f}%)")
    
    def _calculate_sharpe(self, portfolio_values):
        """✅ FIX: Sharpe ratio correct"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # ✅ Vérifier returns valides
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        
        # ✅ Clip pour éviter valeurs absurdes
        sharpe = np.clip(sharpe, -10, 10)
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, portfolio_values):
        """✅ FIX: Drawdown en pourcentage (0-100%)"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        
        # ✅ Vérifier valeurs positives
        if np.any(values <= 0):
            return 100.0  # Perte totale
        
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        max_dd = np.min(drawdown) * 100  # Convertir en %
        
        # ✅ Drawdown est négatif, on prend valeur absolue
        max_dd = abs(max_dd)
        
        # ✅ Clip à [0, 100]
        max_dd = np.clip(max_dd, 0, 100)
        
        return float(max_dd)
    
    def _calculate_win_rate(self, portfolio_values):
        """✅ FIX: Win rate en pourcentage (0-100%)"""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # ✅ Filtrer NaN/Inf
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = (winning_trades / total_trades) * 100
        
        # ✅ Clip à [0, 100]
        win_rate = np.clip(win_rate, 0, 100)
        
        return float(win_rate)
    
    def _calculate_profit_factor(self, portfolio_values_list):
        """✅ Profit factor correct"""
        all_returns = []
        
        for values in portfolio_values_list:
            if len(values) > 1:
                values_array = np.array(values)
                returns = np.diff(values_array) / values_array[:-1]
                
                # Filtrer NaN/Inf
                returns = returns[np.isfinite(returns)]
                all_returns.extend(returns)
        
        if len(all_returns) == 0:
            return None
        
        all_returns = np.array(all_returns)
        
        gross_profit = np.sum(all_returns[all_returns > 0])
        gross_loss = np.abs(np.sum(all_returns[all_returns < 0]))
        
        if gross_loss == 0:
            return None
        
        profit_factor = gross_profit / gross_loss
        
        # ✅ Clip pour éviter valeurs absurdes
        profit_factor = np.clip(profit_factor, 0, 10)
        
        return float(profit_factor)
    
    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\n✅ Trading Metrics Callback terminé")
            print(f"   Total évaluations : {self.n_evaluations}")
