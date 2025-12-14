#!/usr/bin/env python3
"""
⏱️ PLOUTOS MULTI-TIMEFRAME ANALYZER - TRADER PRO

Analyse les signaux sur plusieurs timeframes simultanément
Génère une matrice de confluence pour décisions optimales

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import ta
except ImportError:
    ta = None


class MultiTimeframeAnalyzer:
    """
    Analyse multi-timeframe professionnelle
    """
    
    def __init__(self):
        self.timeframes = {
            '1D': '1d',    # Court terme
            '1W': '1wk',   # Moyen terme
            '1M': '1mo'    # Long terme
        }
    
    def analyze_multi_timeframe(self, ticker: str) -> Dict:
        """
        Analyse complète multi-timeframe
        """
        results = {
            'ticker': ticker,
            'timeframes': {},
            'confluence_matrix': {},
            'overall_signal': None
        }
        
        # Analyser chaque timeframe
        for tf_name, tf_period in self.timeframes.items():
            try:
                # Télécharger les données
                df = yf.download(ticker, period='6mo', interval=tf_period, progress=False)
                
                if df.empty:
                    continue
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Calculer les indicateurs
                signals = self._calculate_timeframe_signals(df, tf_name)
                results['timeframes'][tf_name] = signals
                
            except Exception as e:
                results['timeframes'][tf_name] = {'error': str(e)}
        
        # Générer la matrice de confluence
        results['confluence_matrix'] = self._generate_confluence_matrix(results['timeframes'])
        
        # Signal global
        results['overall_signal'] = self._calculate_overall_signal(results['confluence_matrix'])
        
        return results
    
    def _calculate_timeframe_signals(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Calcule les signaux pour un timeframe donné
        """
        if ta is None or len(df) < 20:
            return {'error': 'Pas assez de données'}
        
        signals = {
            'trend': None,
            'momentum': None,
            'volatility': None,
            'volume': None,
            'overall': 'NEUTRAL'
        }
        
        try:
            current_price = float(df['Close'].iloc[-1])
            
            # ========== TREND ==========
            if len(df) >= 50:
                sma_20 = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
                sma_50 = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1]
                
                if current_price > sma_20 and current_price > sma_50:
                    signals['trend'] = 'BULLISH'
                elif current_price < sma_20 and current_price < sma_50:
                    signals['trend'] = 'BEARISH'
                else:
                    signals['trend'] = 'NEUTRAL'
            
            # ========== MOMENTUM ==========
            if len(df) >= 14:
                rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
                
                if rsi > 60:
                    signals['momentum'] = 'BULLISH'
                elif rsi < 40:
                    signals['momentum'] = 'BEARISH'
                else:
                    signals['momentum'] = 'NEUTRAL'
            
            # ========== VOLATILITY ==========
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(df['Close'])
                bb_upper = bb.bollinger_hband().iloc[-1]
                bb_lower = bb.bollinger_lband().iloc[-1]
                
                if current_price > bb_upper:
                    signals['volatility'] = 'OVERBOUGHT'
                elif current_price < bb_lower:
                    signals['volatility'] = 'OVERSOLD'
                else:
                    signals['volatility'] = 'NORMAL'
            
            # ========== VOLUME ==========
            if len(df) >= 20:
                current_volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                
                if current_volume > avg_volume * 1.5:
                    signals['volume'] = 'HIGH'
                elif current_volume < avg_volume * 0.5:
                    signals['volume'] = 'LOW'
                else:
                    signals['volume'] = 'NORMAL'
            
            # ========== OVERALL ==========
            bullish_count = sum([1 for v in [signals['trend'], signals['momentum']] 
                               if v == 'BULLISH'])
            bearish_count = sum([1 for v in [signals['trend'], signals['momentum']] 
                               if v == 'BEARISH'])
            
            if bullish_count >= 2:
                signals['overall'] = 'BULLISH'
            elif bearish_count >= 2:
                signals['overall'] = 'BEARISH'
            else:
                signals['overall'] = 'NEUTRAL'
            
            # Ajouter le prix pour référence
            signals['price'] = current_price
            
        except Exception as e:
            signals['error'] = str(e)
        
        return signals
    
    def _generate_confluence_matrix(self, timeframes: Dict) -> Dict:
        """
        Génère une matrice de confluence des signaux
        """
        matrix = {
            'trend': {},
            'momentum': {},
            'confluence_score': 0,
            'confluence_direction': 'NEUTRAL'
        }
        
        # Extraire les signaux de tendance et momentum
        for tf_name, signals in timeframes.items():
            if 'error' not in signals:
                matrix['trend'][tf_name] = signals.get('trend', 'NEUTRAL')
                matrix['momentum'][tf_name] = signals.get('momentum', 'NEUTRAL')
        
        # Calculer le score de confluence (0-100)
        total_signals = len(matrix['trend']) + len(matrix['momentum'])
        
        if total_signals > 0:
            bullish_signals = sum([1 for v in list(matrix['trend'].values()) + list(matrix['momentum'].values()) 
                                 if v == 'BULLISH'])
            bearish_signals = sum([1 for v in list(matrix['trend'].values()) + list(matrix['momentum'].values()) 
                                 if v == 'BEARISH'])
            
            matrix['confluence_score'] = int((max(bullish_signals, bearish_signals) / total_signals) * 100)
            
            if bullish_signals > bearish_signals:
                matrix['confluence_direction'] = 'BULLISH'
            elif bearish_signals > bullish_signals:
                matrix['confluence_direction'] = 'BEARISH'
        
        return matrix
    
    def _calculate_overall_signal(self, confluence_matrix: Dict) -> Dict:
        """
        Calcule le signal global avec force
        """
        direction = confluence_matrix.get('confluence_direction', 'NEUTRAL')
        score = confluence_matrix.get('confluence_score', 0)
        
        # Déterminer la force
        if score >= 80:
            strength = 'STRONG'
        elif score >= 60:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        # Recommandation
        if direction == 'BULLISH':
            if strength == 'STRONG':
                recommendation = 'STRONG BUY'
            elif strength == 'MODERATE':
                recommendation = 'BUY'
            else:
                recommendation = 'WEAK BUY'
        elif direction == 'BEARISH':
            if strength == 'STRONG':
                recommendation = 'STRONG SELL'
            elif strength == 'MODERATE':
                recommendation = 'SELL'
            else:
                recommendation = 'WEAK SELL'
        else:
            recommendation = 'HOLD'
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': score,
            'recommendation': recommendation,
            'explanation': self._generate_explanation(direction, strength, score)
        }
    
    def _generate_explanation(self, direction: str, strength: str, score: int) -> str:
        """
        Génère une explication textuelle
        """
        if direction == 'BULLISH':
            if strength == 'STRONG':
                return f"🚀 Confluence haussière FORTE ({score}%) sur tous les timeframes. Signal d'achat très fiable."
            elif strength == 'MODERATE':
                return f"📈 Confluence haussière modérée ({score}%). Signal d'achat avec confirmation recommandée."
            else:
                return f"💡 Confluence haussière faible ({score}%). Attendre une meilleure confirmation."
        
        elif direction == 'BEARISH':
            if strength == 'STRONG':
                return f"🚨 Confluence baissière FORTE ({score}%) sur tous les timeframes. Signal de vente très fiable."
            elif strength == 'MODERATE':
                return f"📉 Confluence baissière modérée ({score}%). Signal de vente avec confirmation recommandée."
            else:
                return f"⚠️ Confluence baissière faible ({score}%). Surveiller l'évolution."
        
        else:
            return f"⏸️ Aucune confluence claire ({score}%). Le marché est indécis, HOLD recommandé."
    
    def get_timeframe_summary(self, analysis_results: Dict) -> str:
        """
        Génère un résumé textuel de l'analyse multi-timeframe
        """
        summary = f"# ⏱️ ANALYSE MULTI-TIMEFRAME - {analysis_results['ticker']}\n\n"
        
        # Détail par timeframe
        for tf_name in ['1D', '1W', '1M']:
            if tf_name in analysis_results['timeframes']:
                signals = analysis_results['timeframes'][tf_name]
                if 'error' not in signals:
                    summary += f"**{tf_name}** : {signals['overall']} "
                    summary += f"(Trend: {signals['trend']}, Momentum: {signals['momentum']})\n"
        
        summary += "\n"
        
        # Signal global
        overall = analysis_results.get('overall_signal', {})
        if overall:
            summary += f"## 🎯 SIGNAL GLOBAL\n\n"
            summary += f"**{overall['recommendation']}** ({overall['confidence']}% confiance)\n\n"
            summary += f"{overall['explanation']}\n"
        
        return summary
