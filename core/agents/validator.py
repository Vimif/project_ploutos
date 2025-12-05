"""
Validator - Valide les modèles entraînés
Backtesting rigoureux
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from config.config import PloutosConfig
from utils.logger import PloutosLogger
from utils.metrics import calculate_all_metrics

logger = PloutosLogger().get_logger(__name__)

class ModelValidator:
    """
    Valide un modèle par backtesting
    
    Responsabilités:
    - Exécuter backtest sur données test
    - Calculer métriques (Sharpe, Drawdown, etc.)
    - Décider si modèle est déployable
    """
    
    def __init__(self, config: PloutosConfig):
        """
        Args:
            config: Configuration Ploutos
        """
        self.config = config
        self.results = {}
    
    def validate(
        self,
        model,
        env_class,
        data: Dict[str, pd.DataFrame],
        max_steps: int = None
    ) -> Dict[str, float]:
        """
        Valide un modèle
        
        Args:
            model: Modèle PPO à valider
            env_class: Classe d'environnement
            data: Données de test
            max_steps: Steps max (ou config.validation.max_steps)
            
        Returns:
            Dict avec métriques de validation
        """
        
        logger.info("="*80)
        logger.info("VALIDATION MODÈLE")
        logger.info("="*80)
        
        if max_steps is None:
            max_steps = self.config.validation.max_steps
        
        logger.info(f"Données: {len(data)} assets")
        logger.info(f"Steps max: {max_steps}")
        
        # Créer environnement (non vectorisé pour validation)
        env = env_class(
            data=data,
            initial_balance=100000,
            commission=0.001,
            max_steps=max_steps
        )
        
        logger.info("Environnement créé")
        
        # Backtest
        obs, _ = env.reset()
        
        values = []
        actions_log = []
        done = False
        truncated = False
        step = 0
        
        logger.info("Début backtest...")
        
        while not done and not truncated and step < max_steps:
            # Prédiction
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Logger données
            values.append(info['portfolio_value'])
            actions_log.append({
                'step': step,
                'n_positions': sum(1 for p in info['positions'].values() if p > 0),
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance']
            })
            
            step += 1
            
            # Log progression
            if step % 200 == 0:
                logger.info(f"Step {step}/{max_steps} - Value: ${info['portfolio_value']:,.2f}")
        
        logger.info(f"Backtest terminé ({step} steps)")
        
        # Calculer métriques
        metrics = calculate_all_metrics(values)
        
        # Ajouter métriques custom
        metrics['n_steps'] = step
        metrics['avg_positions'] = np.mean([a['n_positions'] for a in actions_log])
        
        # Afficher résultats
        logger.info("="*80)
        logger.info("RÉSULTATS VALIDATION")
        logger.info("="*80)
        logger.info(f"Total Return     : {metrics['total_return']:+.2f}%")
        logger.info(f"Sharpe Ratio     : {metrics['sharpe']:.2f}")
        logger.info(f"Max Drawdown     : {metrics['max_drawdown']:.2f}%")
        logger.info(f"Win Rate         : {metrics['win_rate']:.1f}%")
        logger.info(f"Profit Factor    : {metrics['profit_factor']:.2f}")
        logger.info(f"Final Value      : ${metrics['final_value']:,.2f}")
        logger.info(f"Avg Positions    : {metrics['avg_positions']:.1f}")
        logger.info("="*80)
        
        # Décision
        is_valid = metrics['sharpe'] >= self.config.validation.min_sharpe
        
        if is_valid:
            logger.info(f"✅ VALIDATION RÉUSSIE (Sharpe {metrics['sharpe']:.2f} ≥ {self.config.validation.min_sharpe})")
        else:
            logger.warning(f"⚠️ VALIDATION ÉCHOUÉE (Sharpe {metrics['sharpe']:.2f} < {self.config.validation.min_sharpe})")
        
        metrics['is_valid'] = is_valid
        self.results = metrics
        
        return metrics
