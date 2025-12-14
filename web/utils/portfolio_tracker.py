#!/usr/bin/env python3
"""
💼 PLOUTOS PORTFOLIO TRACKER

Gestion complète de portfolio avec:
- Suivi positions en temps réel
- Calcul P&L par action et global
- Allocation et rebalancing
- Suggestions d'optimisation

Usage:
    tracker = PortfolioTracker()
    tracker.add_position('AAPL', shares=10, avg_price=150.00)
    summary = tracker.get_summary()
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Gestionnaire de portfolio complet
    """
    
    def __init__(self, portfolio_path: Optional[str] = None):
        self.portfolio_path = portfolio_path or 'web/data/portfolio.json'
        self.positions = self._load_portfolio()
        logger.info(f"💼 Portfolio Tracker initialisé ({len(self.positions)} positions)")
    
    def add_position(self, ticker: str, shares: float, avg_price: float, 
                    date: Optional[str] = None) -> dict:
        """
        Ajoute ou met à jour une position
        """
        ticker = ticker.upper()
        
        if ticker in self.positions:
            # Moyenne pondérée
            old_shares = self.positions[ticker]['shares']
            old_avg = self.positions[ticker]['avg_price']
            
            new_shares = old_shares + shares
            new_avg = ((old_shares * old_avg) + (shares * avg_price)) / new_shares
            
            self.positions[ticker]['shares'] = new_shares
            self.positions[ticker]['avg_price'] = new_avg
            self.positions[ticker]['last_updated'] = datetime.now().isoformat()
        else:
            self.positions[ticker] = {
                'ticker': ticker,
                'shares': shares,
                'avg_price': avg_price,
                'entry_date': date or datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        self._save_portfolio()
        logger.info(f"✅ Position ajoutée/mise à jour: {ticker}")
        
        return self.positions[ticker]
    
    def remove_position(self, ticker: str, shares: Optional[float] = None) -> bool:
        """
        Supprime ou réduit une position
        """
        ticker = ticker.upper()
        
        if ticker not in self.positions:
            return False
        
        if shares is None or shares >= self.positions[ticker]['shares']:
            # Suppression totale
            del self.positions[ticker]
        else:
            # Réduction partielle
            self.positions[ticker]['shares'] -= shares
            self.positions[ticker]['last_updated'] = datetime.now().isoformat()
        
        self._save_portfolio()
        return True
    
    def get_summary(self) -> Dict:
        """
        Résumé complet du portfolio
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'total_cost': 0,
                'total_pl': 0,
                'total_pl_pct': 0
            }
        
        # Récupérer prix actuels
        tickers = list(self.positions.keys())
        current_prices = self._get_current_prices(tickers)
        
        total_value = 0
        total_cost = 0
        positions_detail = []
        
        for ticker, pos in self.positions.items():
            current_price = current_prices.get(ticker, pos['avg_price'])
            
            market_value = pos['shares'] * current_price
            cost_basis = pos['shares'] * pos['avg_price']
            pl = market_value - cost_basis
            pl_pct = (pl / cost_basis * 100) if cost_basis > 0 else 0
            
            total_value += market_value
            total_cost += cost_basis
            
            positions_detail.append({
                'ticker': ticker,
                'shares': round(pos['shares'], 4),
                'avg_price': round(pos['avg_price'], 2),
                'current_price': round(current_price, 2),
                'market_value': round(market_value, 2),
                'cost_basis': round(cost_basis, 2),
                'pl': round(pl, 2),
                'pl_pct': round(pl_pct, 2),
                'entry_date': pos['entry_date']
            })
        
        # Trier par valeur marché décroissante
        positions_detail.sort(key=lambda x: x['market_value'], reverse=True)
        
        # Calculer allocations
        for pos in positions_detail:
            pos['allocation_pct'] = round((pos['market_value'] / total_value * 100) if total_value > 0 else 0, 2)
        
        total_pl = total_value - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_positions': len(self.positions),
            'total_value': round(total_value, 2),
            'total_cost': round(total_cost, 2),
            'total_pl': round(total_pl, 2),
            'total_pl_pct': round(total_pl_pct, 2),
            'positions': positions_detail,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_allocation(self) -> Dict:
        """
        Analyse l'allocation actuelle
        """
        summary = self.get_summary()
        
        if not summary['positions']:
            return {'error': 'Aucune position'}
        
        # Identifier concentration
        top_3_allocation = sum(p['allocation_pct'] for p in summary['positions'][:3])
        
        warnings = []
        recommendations = []
        
        # Concentration excessive
        if top_3_allocation > 70:
            warnings.append(f"⚠️ Forte concentration: Top 3 = {top_3_allocation:.1f}% du portfolio")
            recommendations.append("💡 Réduisez la concentration en diversifiant davantage")
        
        # Position unique trop importante
        for pos in summary['positions']:
            if pos['allocation_pct'] > 30:
                warnings.append(f"⚠️ {pos['ticker']} représente {pos['allocation_pct']:.1f}% (risque élevé)")
                recommendations.append(f"👉 Considérez réduire {pos['ticker']} sous 25% du portfolio")
        
        # Winners & Losers
        winners = [p for p in summary['positions'] if p['pl_pct'] > 10]
        losers = [p for p in summary['positions'] if p['pl_pct'] < -10]
        
        if winners:
            best = max(winners, key=lambda x: x['pl_pct'])
            recommendations.append(f"🟢 Meilleure performance: {best['ticker']} (+{best['pl_pct']:.1f}%)")
        
        if losers:
            worst = min(losers, key=lambda x: x['pl_pct'])
            warnings.append(f"🔴 Pire performance: {worst['ticker']} ({worst['pl_pct']:.1f}%)")
        
        return {
            'top_3_concentration': round(top_3_allocation, 2),
            'winners_count': len(winners),
            'losers_count': len(losers),
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def suggest_rebalancing(self, target_allocation: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Suggère des ajustements de rebalancing
        
        Args:
            target_allocation: Dict optionnel {'AAPL': 20, 'NVDA': 30, ...} en %
                             Si None, vise égalisation
        """
        summary = self.get_summary()
        
        if not summary['positions']:
            return []
        
        suggestions = []
        total_value = summary['total_value']
        
        if target_allocation is None:
            # Égalisation automatique
            n_positions = len(summary['positions'])
            target_pct = 100 / n_positions
            target_allocation = {p['ticker']: target_pct for p in summary['positions']}
        
        for pos in summary['positions']:
            ticker = pos['ticker']
            current_alloc = pos['allocation_pct']
            target_alloc = target_allocation.get(ticker, 0)
            
            diff = target_alloc - current_alloc
            
            if abs(diff) > 5:  # Seuil 5%
                action = 'ACHETER' if diff > 0 else 'VENDRE'
                amount = abs(diff) * total_value / 100
                shares = amount / pos['current_price']
                
                suggestions.append({
                    'ticker': ticker,
                    'action': action,
                    'current_allocation': round(current_alloc, 2),
                    'target_allocation': round(target_alloc, 2),
                    'diff_pct': round(diff, 2),
                    'amount_usd': round(amount, 2),
                    'shares': round(shares, 2)
                })
        
        suggestions.sort(key=lambda x: abs(x['diff_pct']), reverse=True)
        
        return suggestions
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calcule métriques de performance avancées
        """
        summary = self.get_summary()
        
        if not summary['positions']:
            return {}
        
        # Rendement global
        total_return_pct = summary['total_pl_pct']
        
        # Win rate
        winning_positions = len([p for p in summary['positions'] if p['pl'] > 0])
        win_rate = (winning_positions / len(summary['positions']) * 100)
        
        # Meilleure / Pire position
        best = max(summary['positions'], key=lambda x: x['pl_pct'])
        worst = min(summary['positions'], key=lambda x: x['pl_pct'])
        
        # Exposition totale
        total_exposure = summary['total_value']
        
        return {
            'total_return_pct': round(total_return_pct, 2),
            'win_rate': round(win_rate, 2),
            'best_performer': {
                'ticker': best['ticker'],
                'return_pct': round(best['pl_pct'], 2)
            },
            'worst_performer': {
                'ticker': worst['ticker'],
                'return_pct': round(worst['pl_pct'], 2)
            },
            'total_exposure_usd': round(total_exposure, 2)
        }
    
    def _get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Récupère prix actuels
        """
        prices = {}
        
        try:
            data = yf.download(tickers, period='1d', progress=False)
            
            if len(tickers) == 1:
                prices[tickers[0]] = float(data['Close'].iloc[-1])
            else:
                for ticker in tickers:
                    try:
                        prices[ticker] = float(data['Close'][ticker].iloc[-1])
                    except:
                        prices[ticker] = 0
        except Exception as e:
            logger.error(f"Erreur récupération prix: {e}")
        
        return prices
    
    def _load_portfolio(self) -> Dict:
        """Charge portfolio depuis fichier"""
        try:
            path = Path(self.portfolio_path)
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Erreur chargement portfolio: {e}")
        
        return {}
    
    def _save_portfolio(self):
        """Sauvegarde portfolio"""
        try:
            path = Path(self.portfolio_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde portfolio: {e}")


if __name__ == '__main__':
    # Test
    tracker = PortfolioTracker()
    
    # Ajouter positions de test
    tracker.add_position('AAPL', shares=10, avg_price=150.00)
    tracker.add_position('NVDA', shares=5, avg_price=400.00)
    tracker.add_position('TSLA', shares=3, avg_price=250.00)
    
    print("💼 PORTFOLIO SUMMARY")
    print("="*60)
    
    summary = tracker.get_summary()
    print(f"\nTotal Value: ${summary['total_value']:,.2f}")
    print(f"Total P/L: ${summary['total_pl']:,.2f} ({summary['total_pl_pct']:+.2f}%)")
    
    print("\n📊 POSITIONS:")
    for pos in summary['positions']:
        print(f"  {pos['ticker']}: {pos['shares']} shares @ ${pos['current_price']} | "
              f"P/L: ${pos['pl']:+,.2f} ({pos['pl_pct']:+.2f}%) | "
              f"Allocation: {pos['allocation_pct']:.1f}%")
    
    print("\n🔍 ANALYSIS:")
    analysis = tracker.analyze_allocation()
    for warning in analysis['warnings']:
        print(f"  {warning}")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
