#!/usr/bin/env python3
"""
🔥 PLOUTOS CORRELATION ANALYZER

Analyse les corrélations entre tickers pour:
- Identifier les opportunités de diversification
- Détecter les rotations sectorielles
- Optimiser l'allocation de portfolio

Usage:
    analyzer = CorrelationAnalyzer()
    heatmap = analyzer.generate_heatmap(['AAPL', 'NVDA', 'TSLA', ...])
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyse de corrélations multi-tickers
    """
    
    def __init__(self):
        logger.info("🔥 Correlation Analyzer initialisé")
    
    def generate_heatmap(self, tickers: List[str], period: str = '6mo') -> Dict:
        """
        Génère une heatmap de corrélations
        
        Args:
            tickers: Liste de symboles
            period: Période d'analyse
        
        Returns:
            Dict avec matrix, insights, recommendations
        """
        logger.info(f"🔍 Analyse de {len(tickers)} tickers...")
        
        # Télécharger données
        data = yf.download(tickers, period=period, progress=False)['Close']
        
        if data.empty:
            return {'error': 'Aucune donnée disponible'}
        
        # Calculer rendements quotidiens
        returns = data.pct_change().dropna()
        
        # Matrice de corrélation
        corr_matrix = returns.corr()
        
        # Convertir en format exploitable
        matrix_data = []
        for i, ticker1 in enumerate(tickers):
            row = []
            for j, ticker2 in enumerate(tickers):
                try:
                    value = float(corr_matrix.loc[ticker1, ticker2])
                    if np.isnan(value):
                        value = 0
                except:
                    value = 0
                
                row.append({
                    'x': ticker2,
                    'y': ticker1,
                    'value': round(value, 3)
                })
            matrix_data.extend(row)
        
        # Insights
        insights = self._generate_insights(corr_matrix, tickers)
        
        # Recommandations de diversification
        recommendations = self._generate_recommendations(corr_matrix, tickers)
        
        return {
            'tickers': tickers,
            'period': period,
            'matrix': matrix_data,
            'insights': insights,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def find_uncorrelated_pairs(self, tickers: List[str], period: str = '6mo', 
                                threshold: float = 0.3) -> List[Dict]:
        """
        Trouve les paires les MOINS corrélées (bonnes pour diversification)
        """
        data = yf.download(tickers, period=period, progress=False)['Close']
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        pairs = []
        
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                ticker1 = tickers[i]
                ticker2 = tickers[j]
                
                try:
                    corr = float(corr_matrix.loc[ticker1, ticker2])
                    
                    if abs(corr) < threshold:
                        pairs.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': round(corr, 3),
                            'diversification_score': round((1 - abs(corr)) * 100, 1)
                        })
                except:
                    pass
        
        # Trier par score de diversification décroissant
        pairs.sort(key=lambda x: x['diversification_score'], reverse=True)
        
        return pairs
    
    def find_highly_correlated(self, tickers: List[str], period: str = '6mo', 
                              threshold: float = 0.8) -> List[Dict]:
        """
        Trouve les paires TRÈS corrélées (redondance de portfolio)
        """
        data = yf.download(tickers, period=period, progress=False)['Close']
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        pairs = []
        
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                ticker1 = tickers[i]
                ticker2 = tickers[j]
                
                try:
                    corr = float(corr_matrix.loc[ticker1, ticker2])
                    
                    if corr > threshold:
                        pairs.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': round(corr, 3),
                            'redundancy_warning': corr > 0.9
                        })
                except:
                    pass
        
        pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        return pairs
    
    def analyze_sector_rotation(self, sectors: Dict[str, List[str]], 
                               period: str = '3mo') -> Dict:
        """
        Analyse les rotations sectorielles
        
        Args:
            sectors: Dict {'Tech': ['AAPL', 'NVDA'], 'Finance': ['JPM', 'BAC'], ...}
        
        Returns:
            Analyse de performance relative par secteur
        """
        sector_performance = {}
        
        for sector, tickers in sectors.items():
            try:
                data = yf.download(tickers, period=period, progress=False)['Close']
                
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                
                # Performance moyenne du secteur
                returns = data.pct_change().dropna()
                avg_return = returns.mean().mean() * 100
                volatility = returns.std().mean() * 100
                
                # Derniers 5 jours
                recent_return = ((data.iloc[-1] / data.iloc[-5]).mean() - 1) * 100 if len(data) >= 5 else 0
                
                sector_performance[sector] = {
                    'avg_return': round(float(avg_return), 2),
                    'volatility': round(float(volatility), 2),
                    'recent_5d_return': round(float(recent_return), 2),
                    'momentum': 'STRONG' if recent_return > 2 else 'WEAK' if recent_return < -2 else 'NEUTRAL'
                }
            except Exception as e:
                logger.error(f"Erreur secteur {sector}: {e}")
        
        # Identifier secteur leader
        if sector_performance:
            leader = max(sector_performance.items(), key=lambda x: x[1]['recent_5d_return'])
            laggard = min(sector_performance.items(), key=lambda x: x[1]['recent_5d_return'])
            
            return {
                'sectors': sector_performance,
                'leader': {'name': leader[0], **leader[1]},
                'laggard': {'name': laggard[0], **laggard[1]},
                'rotation_signal': 'ROTATE_TO_' + leader[0].upper() if leader[1]['recent_5d_return'] > 3 else 'HOLD',
                'timestamp': datetime.now().isoformat()
            }
        
        return {'error': 'Aucune donnée secteur'}
    
    def _generate_insights(self, corr_matrix: pd.DataFrame, tickers: List[str]) -> List[str]:
        """
        Génère insights sur les corrélations
        """
        insights = []
        
        # Moyenne de corrélation globale
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if avg_corr > 0.7:
            insights.append(f"⚠️ Corrélation moyenne très élevée ({avg_corr:.2f}) - Portfolio peu diversifié")
        elif avg_corr < 0.3:
            insights.append(f"✅ Excellente diversification (corrélation moyenne: {avg_corr:.2f})")
        else:
            insights.append(f"🟡 Diversification modérée (corrélation moyenne: {avg_corr:.2f})")
        
        # Trouver paires extrêmes
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                try:
                    corr = corr_matrix.iloc[i, j]
                    
                    if corr > 0.9:
                        insights.append(f"🔴 {tickers[i]} et {tickers[j]} sont quasi identiques (corr: {corr:.2f})")
                    elif corr < -0.5:
                        insights.append(f"🟢 {tickers[i]} et {tickers[j]} sont inversément corrélés (hedge naturel)")
                except:
                    pass
        
        return insights[:5]  # Max 5 insights
    
    def _generate_recommendations(self, corr_matrix: pd.DataFrame, tickers: List[str]) -> List[str]:
        """
        Génère recommandations d'optimisation
        """
        recommendations = []
        
        # Identifier redondances
        redundant_pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                try:
                    if corr_matrix.iloc[i, j] > 0.85:
                        redundant_pairs.append((tickers[i], tickers[j]))
                except:
                    pass
        
        if redundant_pairs:
            recommendations.append(f"⚠️ Considérez remplacer l'un de ces doublons: {redundant_pairs[0]}")
        
        # Chercher actifs décorrélés manquants
        low_corr_count = 0
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                try:
                    if abs(corr_matrix.iloc[i, j]) < 0.3:
                        low_corr_count += 1
                except:
                    pass
        
        if low_corr_count < len(tickers) * 0.3:
            recommendations.append("💡 Ajoutez des actifs non corrélés (ex: or, obligations, crypto)")
        
        return recommendations


if __name__ == '__main__':
    # Test
    analyzer = CorrelationAnalyzer()
    
    tech_tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META']
    
    print("🔥 Génération heatmap...")
    heatmap = analyzer.generate_heatmap(tech_tickers, period='6mo')
    
    print("\n💡 INSIGHTS:")
    for insight in heatmap['insights']:
        print(f"  {insight}")
    
    print("\n🎯 RECOMMANDATIONS:")
    for rec in heatmap['recommendations']:
        print(f"  {rec}")
