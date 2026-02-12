# core/risk_manager.py
"""Gestionnaire de risque avancé pour le trading"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from core.utils import setup_logging

logger = setup_logging(__name__)

class RiskManager:
    """Gestionnaire de risque sophistiqué"""
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,
                 max_daily_loss: float = 0.03,
                 max_position_size: float = 0.05,
                 max_correlation: float = 0.7):
        """
        Initialiser le risk manager
        
        Args:
            max_portfolio_risk: Risque max par trade (2% par défaut)
            max_daily_loss: Perte quotidienne max (3% par défaut)
            max_position_size: Taille max d'une position (5% par défaut)
            max_correlation: Corrélation max entre positions (0.7 par défaut)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        
        self.daily_start_value = None
        self.daily_trades = 0
        self.daily_losses = 0
        self.daily_wins = 0
        self.circuit_breaker_triggered = False
        
        logger.info(f"🛡️ Risk Manager initialisé")
        logger.info(f"   Max risk/trade: {max_portfolio_risk*100:.1f}%")
        logger.info(f"   Max daily loss: {max_daily_loss*100:.1f}%")
        logger.info(f"   Max position: {max_position_size*100:.1f}%")
    
    def calculate_position_size(self, 
                               portfolio_value: float,
                               entry_price: float,
                               stop_loss_pct: float,
                               risk_pct: Optional[float] = None) -> Tuple[int, float]:
        """
        Calculer la taille optimale d'une position basée sur le risque
        Position Sizing = Risk Amount / Stop Loss Distance
        
        Args:
            portfolio_value: Valeur totale du portfolio
            entry_price: Prix d'entrée prévu
            stop_loss_pct: Stop loss en % (ex: 0.05 pour 5%)
            risk_pct: Risque par trade (utilise max_portfolio_risk si None)
        
        Returns:
            (quantity, position_value): Quantité et valeur de la position
        """
        if risk_pct is None:
            risk_pct = self.max_portfolio_risk
        
        # Montant à risquer sur ce trade
        risk_amount = portfolio_value * risk_pct
        
        # Distance du stop loss en $
        stop_loss_distance = entry_price * stop_loss_pct
        
        # Nombre d'actions = Risk / Stop Loss Distance
        quantity = int(risk_amount / stop_loss_distance)
        
        # Valeur de la position
        position_value = quantity * entry_price
        position_pct = position_value / portfolio_value
        
        # Vérifier qu'on ne dépasse pas la taille max
        if position_pct > self.max_position_size:
            # Réduire la quantité
            max_position_value = portfolio_value * self.max_position_size
            quantity = int(max_position_value / entry_price)
            position_value = quantity * entry_price
            
            logger.info(f"⚠️  Position réduite à {self.max_position_size*100:.0f}% max")
        
        logger.info(f"📊 Position sizing: {quantity} actions @ ${entry_price:.2f} = ${position_value:,.2f} ({position_value/portfolio_value*100:.1f}%)")
        logger.info(f"   Risk: ${risk_amount:,.2f} ({risk_pct*100:.1f}%) | Stop: {stop_loss_pct*100:.0f}%")
        
        return quantity, position_value
    
    def check_daily_loss_limit(self, current_value: float) -> bool:
        """
        Vérifier si la perte quotidienne dépasse la limite (Circuit Breaker)
        
        Args:
            current_value: Valeur actuelle du portfolio
        
        Returns:
            True si trading autorisé, False si circuit breaker activé
        """
        if self.daily_start_value is None:
            self.daily_start_value = current_value
            return True
        
        # Calculer P&L quotidien
        daily_pl = current_value - self.daily_start_value
        daily_pl_pct = daily_pl / self.daily_start_value
        
        # Vérifier si perte dépasse limite
        if daily_pl_pct <= -self.max_daily_loss:
            if not self.circuit_breaker_triggered:
                self.circuit_breaker_triggered = True
                logger.error(f"🚨 CIRCUIT BREAKER ACTIVÉ!")
                logger.error(f"   Perte quotidienne: ${daily_pl:,.2f} ({daily_pl_pct*100:.2f}%)")
                logger.error(f"   Limite: {self.max_daily_loss*100:.0f}%")
                logger.error(f"   🛑 TRADING SUSPENDU JUSQU'À DEMAIN")
            return False
        
        return True
    
    def reset_daily_stats(self, portfolio_value: float):
        """Réinitialiser les stats quotidiennes (à appeler à minuit)"""
        self.daily_start_value = portfolio_value
        self.daily_trades = 0
        self.daily_losses = 0
        self.daily_wins = 0
        self.circuit_breaker_triggered = False
        logger.info(f"🔄 Stats quotidiennes réinitialisées - Portfolio: ${portfolio_value:,.2f}")
    
    def calculate_portfolio_exposure(self, positions: List[Dict], portfolio_value: float) -> float:
        """
        Calculer l'exposition totale du portfolio
        
        Returns:
            Exposition en % (0.0 à 1.0)
        """
        total_exposure = sum(pos['market_value'] for pos in positions)
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
        return exposure_pct
    
    def should_reduce_exposure(self, positions: List[Dict], portfolio_value: float) -> Tuple[bool, str]:
        """
        Déterminer si on doit réduire l'exposition
        
        Returns:
            (should_reduce, reason)
        """
        exposure_pct = self.calculate_portfolio_exposure(positions, portfolio_value)
        
        # Trop investi (>85%)
        if exposure_pct > 0.85:
            return True, f"Exposition élevée: {exposure_pct*100:.1f}%"
        
        # Trop de positions perdantes
        losing_positions = [p for p in positions if p['unrealized_plpc'] < -0.05]
        if len(losing_positions) >= len(positions) * 0.6:
            return True, f"{len(losing_positions)}/{len(positions)} positions en perte >5%"
        
        return False, "Exposition acceptable"
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculer le Kelly Criterion pour position sizing optimal
        Kelly% = W - [(1-W) / R]
        où W = win rate, R = avg_win / avg_loss
        
        Args:
            win_rate: Taux de réussite (0.0 à 1.0)
            avg_win: Gain moyen en %
            avg_loss: Perte moyenne en % (positif)
        
        Returns:
            Position size optimal en % (0.0 à 1.0)
        """
        if avg_loss == 0 or win_rate == 0:
            return self.max_portfolio_risk
        
        # Ratio gain/perte
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly criterion
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Utiliser half-kelly pour plus de sécurité
        half_kelly = kelly_pct * 0.5
        
        # Limiter entre 0 et max_portfolio_risk
        kelly_capped = max(0, min(half_kelly, self.max_portfolio_risk))
        
        logger.info(f"📈 Kelly Criterion: {kelly_pct*100:.1f}% (Half: {half_kelly*100:.1f}%, Used: {kelly_capped*100:.1f}%)")
        
        return kelly_capped
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.04) -> float:
        """
        Calculer le Sharpe Ratio
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        
        Args:
            returns: Liste des returns quotidiens
            risk_free_rate: Taux sans risque annuel (4% par défaut)
        
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Return moyen annualisé
        mean_return = np.mean(returns_array) * 252  # 252 jours de trading
        
        # Volatilité annualisée
        std_dev = np.std(returns_array) * np.sqrt(252)
        
        if std_dev == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_dev
        
        return sharpe
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> Tuple[float, float]:
        """
        Calculer le maximum drawdown
        
        Args:
            portfolio_values: Liste des valeurs historiques du portfolio
        
        Returns:
            (max_drawdown_pct, max_drawdown_amount)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0.0
        
        values = np.array(portfolio_values)
        
        # Calculer les peaks cumulatifs
        cumulative_max = np.maximum.accumulate(values)
        
        # Drawdown à chaque point
        drawdowns = (values - cumulative_max) / cumulative_max
        
        # Max drawdown
        max_dd_pct = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        peak_value = cumulative_max[max_dd_idx]
        valley_value = values[max_dd_idx]
        max_dd_amount = peak_value - valley_value
        
        return max_dd_pct, max_dd_amount
    
    def assess_position_risk(self, 
                            symbol: str,
                            position_value: float,
                            portfolio_value: float,
                            unrealized_plpc: float,
                            days_held: int) -> Dict:
        """
        Évaluer le risque d'une position existante
        
        Returns:
            Dict avec score de risque et recommandations
        """
        risk_score = 0
        warnings = []
        
        # 1. Taille de la position
        position_pct = position_value / portfolio_value
        if position_pct > self.max_position_size:
            risk_score += 2
            warnings.append(f"Position trop grande: {position_pct*100:.1f}%")
        
        # 2. Perte non réalisée
        if unrealized_plpc < -0.10:  # -10%
            risk_score += 3
            warnings.append(f"Perte importante: {unrealized_plpc*100:.1f}%")
        elif unrealized_plpc < -0.05:  # -5%
            risk_score += 1
            warnings.append(f"Perte significative: {unrealized_plpc*100:.1f}%")
        
        # 3. Durée de détention
        if days_held > 30 and unrealized_plpc < 0:
            risk_score += 1
            warnings.append(f"Perte prolongée: {days_held} jours")
        
        # Évaluation globale
        if risk_score >= 4:
            recommendation = "FERMER IMMÉDIATEMENT"
            risk_level = "CRITIQUE"
        elif risk_score >= 2:
            recommendation = "SURVEILLER DE PRÈS"
            risk_level = "ÉLEVÉ"
        elif risk_score >= 1:
            recommendation = "ATTENTION"
            risk_level = "MOYEN"
        else:
            recommendation = "OK"
            risk_level = "FAIBLE"
        
        return {
            'symbol': symbol,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'warnings': warnings
        }
    
    def get_risk_report(self, 
                       positions: List[Dict],
                       portfolio_value: float,
                       daily_returns: List[float] = None) -> Dict:
        """
        Générer un rapport de risque complet
        
        Returns:
            Dict avec métriques de risque
        """
        # Exposition
        exposure_pct = self.calculate_portfolio_exposure(positions, portfolio_value)
        
        # Positions à risque
        risky_positions = []
        for pos in positions:
            # Calculer les jours de détention (days_held)
            days_held = 0
            purchase_date = pos.get('purchase_date') or pos.get('created_at')

            if purchase_date:
                try:
                    if isinstance(purchase_date, str):
                        # Gérer le format ISO avec ou sans 'Z'
                        dt_str = purchase_date.replace('Z', '+00:00')
                        purchase_dt = datetime.fromisoformat(dt_str)
                    else:
                        purchase_dt = purchase_date

                    if purchase_dt:
                        # Calculer la différence de jours
                        now = datetime.now(purchase_dt.tzinfo) if purchase_dt.tzinfo else datetime.now()
                        delta = now - purchase_dt
                        days_held = max(0, delta.days)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur calcul days_held pour {pos.get('symbol')}: {e}")

            risk = self.assess_position_risk(
                symbol=pos['symbol'],
                position_value=pos['market_value'],
                portfolio_value=portfolio_value,
                unrealized_plpc=pos['unrealized_plpc'],
                days_held=days_held
            )
            if risk['risk_score'] >= 2:
                risky_positions.append(risk)
        
        # Sharpe ratio
        sharpe = 0.0
        if daily_returns and len(daily_returns) > 2:
            sharpe = self.calculate_sharpe_ratio(daily_returns)
        
        # Circuit breaker status
        daily_pl = 0.0
        daily_pl_pct = 0.0
        if self.daily_start_value:
            daily_pl = portfolio_value - self.daily_start_value
            daily_pl_pct = daily_pl / self.daily_start_value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'exposure_pct': exposure_pct,
            'positions_count': len(positions),
            'risky_positions': risky_positions,
            'risky_positions_count': len(risky_positions),
            'sharpe_ratio': sharpe,
            'daily_pl': daily_pl,
            'daily_pl_pct': daily_pl_pct,
            'circuit_breaker': self.circuit_breaker_triggered,
            'max_daily_loss': self.max_daily_loss,
            'daily_trades': self.daily_trades,
            'daily_wins': self.daily_wins,
            'daily_losses': self.daily_losses
        }
        
        return report
    
    def log_trade(self, symbol: str, action: str, pl: float = None):
        """Logger un trade pour les stats quotidiennes"""
        self.daily_trades += 1
        
        if pl is not None:
            if pl > 0:
                self.daily_wins += 1
            elif pl < 0:
                self.daily_losses += 1
    
    def print_risk_summary(self, report: Dict):
        """Afficher un résumé du risque"""
        logger.info("\n" + "="*70)
        logger.info("🛡️ RISK MANAGEMENT REPORT")
        logger.info("="*70)
        logger.info(f"Portfolio: ${report['portfolio_value']:,.2f}")
        logger.info(f"Exposition: {report['exposure_pct']*100:.1f}%")
        logger.info(f"Positions: {report['positions_count']} ({report['risky_positions_count']} à risque)")
        
        if report['sharpe_ratio'] != 0:
            logger.info(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        
        logger.info(f"P&L quotidien: ${report['daily_pl']:+,.2f} ({report['daily_pl_pct']*100:+.2f}%)")
        logger.info(f"Trades aujourd'hui: {report['daily_trades']} (W:{report['daily_wins']} L:{report['daily_losses']})")
        
        if report['circuit_breaker']:
            logger.error(f"🚨 CIRCUIT BREAKER ACTIF - Trading suspendu")
        
        if report['risky_positions']:
            logger.warning(f"\n⚠️  POSITIONS À RISQUE:")
            for pos in report['risky_positions']:
                logger.warning(f"   {pos['symbol']}: {pos['risk_level']} - {pos['recommendation']}")
                for warning in pos['warnings']:
                    logger.warning(f"      • {warning}")
        
        logger.info("="*70)