"""
Factory pour créer différentes versions d'environnements
"""

from core.environment import TradingEnv
from core.environment_sharpe import TradingEnvSharpe
from core.environment_continuous import TradingEnvContinuous
from core.environment_multitimeframe import TradingEnvMultiTimeframe

def create_environment(env_type='default', **kwargs):
    """
    Créer un environnement de trading
    
    Args:
        env_type: Type d'environnement
            - 'default': Environnement basique
            - 'sharpe': Reward basé sur Sharpe Ratio
            - 'continuous': Actions continues
            - 'multitimeframe': Données multi-timeframe
        **kwargs: Arguments passés à l'environnement
    
    Returns:
        Instance de l'environnement
    """
    
    environments = {
        'default': TradingEnv,
        'sharpe': TradingEnvSharpe,
        'continuous': TradingEnvContinuous,
        'multitimeframe': TradingEnvMultiTimeframe
    }
    
    if env_type not in environments:
        raise ValueError(f"Type d'environnement invalide: {env_type}. "
                        f"Choix: {list(environments.keys())}")
    
    return environments[env_type](**kwargs)
