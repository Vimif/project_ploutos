import numpy as np
from collections import deque

class DifferentialSharpeReward:
    """
    Calcule le Differential Sharpe Ratio (DSR) pour l'apprentissage par renforcement.
    Le DSR permet d'optimiser directement le Sharpe Ratio étape par étape.
    
    Source: "Reinforcement Learning for Trading" (Moody & Saffell, 2001)
    """
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.A = 0.0  # Moyenne mobile exponentielle des retours (First moment)
        self.B = 0.0  # Moyenne mobile exponentielle des retours au carré (Second moment)
        self.reset()

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def calculate(self, return_val: float) -> float:
        """
        Calcule la récompense basée sur le changement du Sharpe Ratio.
        
        Args:
            return_val (float): Le retour du portefeuille à ce step (ex: 0.01 pour 1%)
        """
        # Sauvegarde des anciennes valeurs
        prev_A = self.A
        prev_B = self.B
        
        # Mise à jour des moyennes mobiles (Online updating)
        self.A = self.decay * self.A + (1 - self.decay) * return_val
        self.B = self.decay * self.B + (1 - self.decay) * (return_val ** 2)
        
        # Calcul de la variance
        # Var = E[x^2] - (E[x])^2
        variance = self.B - self.A ** 2
        
        # Éviter la division par zéro ou variance négative (erreurs numériques)
        if variance < 1e-6:
            return 0.0
            
        std_dev = np.sqrt(variance)
        
        # Formule du Differential Sharpe Ratio
        # D_t = (B_{t-1} * \Delta A_t - 0.5 * A_{t-1} * \Delta B_t) / (Var_{t-1} ^ 1.5)
        # Où \Delta A_t = R_t - A_{t-1} et \Delta B_t = R_t^2 - B_{t-1}
        # Mais une forme simplifiée souvent utilisée est le gradient direct.
        # Ici nous utilisons l'approximation de Moody:
        
        dsr = (self.B * prev_A - 0.5 * self.A * prev_B) / (variance ** 1.5 + 1e-8)
        
        return dsr

class AdvancedRewardCalculator:
    """
    Calculateur de récompense hybride combinant:
    1. Differential Sharpe (Stabilité)
    2. Sortino Ratio (Pénalité downside uniquement)
    3. Bonus Win Rate (Psychologie)
    """
    def __init__(self, window_size=252):
        self.dsr_calculator = DifferentialSharpeReward(decay=0.99)
        self.returns_window = deque(maxlen=window_size)
        self.consistency_window = deque(maxlen=20)
        
    def reset(self):
        self.dsr_calculator.reset()
        self.returns_window.clear()
        self.consistency_window.clear()
        
    def calculate(self, 
                  step_return: float, 
                  current_drawdown: float, 
                  trades_today: int, 
                  is_winning_trade: bool = False) -> float:
        
        self.returns_window.append(step_return)
        
        # 1. Base: Differential Sharpe (Pousse à la régularité)
        dsr_reward = self.dsr_calculator.calculate(step_return)
        
        # 2. Drawdown Penalty (Non-linéaire)
        # Pénalité légère si < 5%, exponentielle si > 10%
        dd_penalty = 0.0
        if current_drawdown > 0.05:
            dd_penalty = -np.exp(current_drawdown * 10) * 0.1
            
        # 3. Sortino Bonus (Récompense les gros gains sans punir la volatilité haussière)
        sortino_bonus = 0.0
        if step_return > 0:
            downside_std = np.std([r for r in self.returns_window if r < 0]) if len(self.returns_window) > 10 else 1.0
            if downside_std < 1e-6: downside_std = 1.0
            sortino_bonus = (step_return / downside_std) * 0.5

        # 4. Trading Activity Bonus (Encourager à prendre des profits)
        trade_bonus = 0.0
        if is_winning_trade:
            trade_bonus = 1.0  # Gros bonus ponctuel pour valider un gain
            
        # 5. Overtrading Penalty
        trade_penalty = 0.0
        if trades_today > 5:
            trade_penalty = -0.1 * (trades_today - 5)

        # Assemblage
        # On clip le DSR car il peut être très volatile au début
        total_reward = (
            np.clip(dsr_reward, -2, 2) * 1.0 +
            dd_penalty + 
            sortino_bonus +
            trade_bonus +
            trade_penalty
        )
        
        return np.clip(total_reward, -10, 10)
