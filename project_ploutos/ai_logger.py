# ai_logger.py
# ---------------------------------------------------------
# SYSTÈME DE LOGGING TENSORBOARD
# ---------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # On récupère l'environnement
        # Dans un environnement vectorisé, c'est un peu caché
        env = self.training_env
        
        # On essaie de récupérer la dernière valeur du portefeuille (net_worth)
        # Note: Cela dépend de si l'environnement expose cette info
        # Pour l'instant, on se base sur les rewards standards qui sont déjà loggués.
        
        return True
