"""Gestion du portfolio (positions, sizing, etc.)"""

import logging
logger = logging.getLogger(__name__)

class PortfolioManager:
    """Gère les positions et le sizing"""
    
    def __init__(self, alpaca_client, initial_capital):
        self.alpaca = alpaca_client
        self.initial_capital = initial_capital
        self.max_position_size_pct = 0.05
        self.max_accumulation_pct = 0.10
    
    def calculate_position_size(self, symbol, current_price, portfolio_value, risk_manager=None):
        """Calcule la taille optimale d'une position"""
        
        if risk_manager:
            qty, _ = risk_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=current_price,
                stop_loss_pct=0.02
            )
            return qty
        
        # Fallback simple
        max_invest = portfolio_value * self.max_position_size_pct
        return int(max_invest / current_price)
    
    def should_add_to_position(self, symbol, position, portfolio_value, add_to_winner=True):
        """Détermine si on doit renforcer une position"""
        
        position_pct = position['market_value'] / portfolio_value
        
        if position_pct >= self.max_accumulation_pct:
            return False, f"Max accumulation ({position_pct*100:.1f}%)"
        
        unrealized_plpc = position['unrealized_plpc']
        
        if unrealized_plpc > 0 and add_to_winner:
            return True, f"Renforcer gagnant (+{unrealized_plpc*100:.1f}%)"
        elif unrealized_plpc < 0:
            return False, "Pas de moyenne à la baisse"
        else:
            return True, "Position neutre OK"
