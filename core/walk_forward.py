#!/usr/bin/env python3
"""
Walk-Forward Analysis pour validation robuste
Prévient l'overfitting en testant sur périodes non-adjacentes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os

class WalkForwardValidator:
    """
    Implémente Walk-Forward Analysis pour validation de stratégie
    
    Principe :
    - Divise historique en windows glissantes
    - Entraîne sur window N, teste sur window N+1
    - Décale window progressivement
    - Évalue stabilité des performances
    
    Example:
        wfv = WalkForwardValidator(train_window=180, test_window=60, step=30)
        folds = wfv.split(data)
        results = wfv.validate(model, folds)
    """
    
    def __init__(self, train_window=180, test_window=60, step=30):
        """
        Args:
            train_window: Jours d'entraînement (180 = 6 mois)
            test_window: Jours de test (60 = 2 mois)
            step: Jours entre chaque fold (30 = 1 mois)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        
    def split(self, data: pd.DataFrame) -> List[Dict]:
        """
        Génère les folds walk-forward
        
        Args:
            data: DataFrame avec index temporel
            
        Returns:
            Liste de dicts {'train': df, 'test': df, 'period': str}
        """
        
        total_days = len(data)
        folds = []
        
        start = 0
        fold_num = 1
        
        while start + self.train_window + self.test_window <= total_days:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            folds.append({
                'fold': fold_num,
                'train': train_data,
                'test': test_data,
                'train_period': f"{train_data.index[0].date()} → {train_data.index[-1].date()}",
                'test_period': f"{test_data.index[0].date()} → {test_data.index[-1].date()}",
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
            
            start += self.step
            fold_num += 1
            
        print(f"✅ {len(folds)} folds générés")
        print(f"  Train window : {self.train_window} jours")
        print(f"  Test window  : {self.test_window} jours")
        print(f"  Step         : {self.step} jours\n")
        
        return folds
    
    def validate(self, model, folds: List[Dict], env_class, initial_balance=100000) -> pd.DataFrame:
        """
        Évalue le modèle sur tous les folds
        
        Args:
            model: Modèle Stable-Baselines3 entraîné
            folds: Liste de folds depuis split()
            env_class: Classe d'environnement (UniversalTradingEnv)
            initial_balance: Capital initial pour chaque fold
            
        Returns:
            DataFrame avec résultats par fold
        """
        
        print("\n" + "="*80)
        print("📊 WALK-FORWARD VALIDATION")
        print("="*80)
        
        results = []
        
        for fold in folds:
            print(f"\n📈 Fold {fold['fold']}/{len(folds)}")
            print(f"  Test : {fold['test_period']}")
            
            # Backtest sur période test
            metrics = self._backtest_fold(model, fold['test'], env_class, initial_balance)
            
            results.append({
                'fold': fold['fold'],
                'test_period': fold['test_period'],
                'sharpe': metrics['sharpe'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'n_trades': metrics['n_trades'],
                'final_value': metrics['final_value']
            })
            
            print(f"  Sharpe       : {metrics['sharpe']:.2f}")
            print(f"  Return       : {metrics['total_return']:.2f}%")
            print(f"  Max DD       : {metrics['max_drawdown']:.2f}%")
            print(f"  Win Rate     : {metrics['win_rate']:.1f}%")
        
        df_results = pd.DataFrame(results)
        
        # Statistiques globales
        print("\n" + "="*80)
        print("📈 RÉSULTATS GLOBAUX")
        print("="*80)
        print(f"  Sharpe Moyen  : {df_results['sharpe'].mean():.2f} ± {df_results['sharpe'].std():.2f}")
        print(f"  Return Moyen  : {df_results['total_return'].mean():.2f}%")
        print(f"  Max DD Moyen  : {df_results['max_drawdown'].mean():.2f}%")
        print(f"  Win Rate Moy  : {df_results['win_rate'].mean():.1f}%")
        
        # Verdict
        mean_sharpe = df_results['sharpe'].mean()
        std_sharpe = df_results['sharpe'].std()
        
        print("\n" + "-"*80)
        if mean_sharpe > 1.0 and std_sharpe < 0.5:
            print("✅ MODÈLE ROBUSTE : Performances stables sur toutes les périodes")
        elif mean_sharpe > 0.5:
            print("⚠️  MODÈLE INSTABLE : Performances variables (risque overfitting)")
        else:
            print("❌ MODÈLE REJETÉ : Performances insuffisantes")
        print("-"*80 + "\n")
        
        return df_results
    
    def _backtest_fold(self, model, test_data: pd.DataFrame, env_class, initial_balance: float) -> Dict:
        """
        Backtest sur un fold spécifique
        
        Args:
            model: Modèle PPO
            test_data: Données de test
            env_class: Classe d'environnement
            initial_balance: Capital initial
            
        Returns:
            Dict avec métriques de performance
        """
        
        # Créer environnement de test
        env = env_class(
            data={'TEST': test_data},  # Adapter selon structure data
            initial_balance=initial_balance,
            commission=0.001,
            max_steps=len(test_data)
        )
        
        obs, _ = env.reset()
        
        values = [initial_balance]
        trades = []
        done = False
        step = 0
        
        while not done and step < len(test_data) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            values.append(info['portfolio_value'])
            
            if info.get('n_trades', 0) > len(trades):
                trades.append(step)
            
            done = terminated or truncated
            step += 1
        
        # Calculer métriques
        df = pd.DataFrame({'value': values})
        df['returns'] = df['value'].pct_change().fillna(0)
        
        # Sharpe Ratio
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 6.5) if std_ret > 0 else 0
        
        # Total Return
        total_return = ((values[-1] - initial_balance) / initial_balance) * 100
        
        # Max Drawdown
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win Rate
        winning_days = (df['returns'] > 0).sum()
        win_rate = (winning_days / len(df)) * 100
        
        # Profit Factor
        gains = df[df['returns'] > 0]['returns'].sum()
        losses = abs(df[df['returns'] < 0]['returns'].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return {
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'n_trades': len(trades),
            'final_value': float(values[-1])
        }
    
    def plot_results(self, results: pd.DataFrame, save_path='reports/walk_forward_results.png'):
        """
        Visualise les résultats walk-forward
        
        Args:
            results: DataFrame depuis validate()
            save_path: Chemin de sauvegarde
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sharpe par fold
        axes[0, 0].bar(results['fold'], results['sharpe'], color='steelblue')
        axes[0, 0].axhline(y=results['sharpe'].mean(), color='red', linestyle='--', label='Moyenne')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio par Fold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Returns par fold
        axes[0, 1].bar(results['fold'], results['total_return'], color='green', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].set_title('Returns par Fold')
        axes[0, 1].grid(alpha=0.3)
        
        # Max Drawdown par fold
        axes[1, 0].bar(results['fold'], results['max_drawdown'], color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Max Drawdown par Fold')
        axes[1, 0].grid(alpha=0.3)
        
        # Win Rate par fold
        axes[1, 1].bar(results['fold'], results['win_rate'], color='purple', alpha=0.7)
        axes[1, 1].axhline(y=50, color='black', linestyle='--', label='50%')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].set_title('Win Rate par Fold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Graphique sauvegardé : {save_path}")
        
        plt.close()

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Exemple d'utilisation du Walk-Forward Validator
    """
    
    print("🧪 Test du Walk-Forward Validator\n")
    
    # Générer données factices
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    fake_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Créer validator
    wfv = WalkForwardValidator(
        train_window=180,  # 6 mois train
        test_window=60,    # 2 mois test
        step=30            # Décalage 1 mois
    )
    
    # Split
    folds = wfv.split(fake_data)
    
    print(f"\n📊 {len(folds)} folds créés :")
    for fold in folds[:3]:  # Afficher 3 premiers
        print(f"  Fold {fold['fold']} : Train {fold['train_period']} | Test {fold['test_period']}")
    
    print("\n✅ Walk-Forward Validator prêt à l'emploi !")
