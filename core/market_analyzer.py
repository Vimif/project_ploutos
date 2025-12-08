# core/market_analyzer.py
"""
📊 MARKET ANALYZER

Analyse technique pour expliquer décisions du modèle IA

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """
    Analyseur de marché avec indicateurs techniques
    """
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Calculer RSI (Relative Strength Index)
        
        Args:
            prices: Série de prix
            period: Période (défaut 14)
        
        Returns:
            float: RSI (0-100)
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Dict[str, float]:
        """
        Calculer MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Série de prix
        
        Returns:
            dict: {'macd', 'signal', 'histogram'}
        """
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, float]:
        """
        Calculer Bandes de Bollinger
        
        Args:
            prices: Série de prix
            period: Période
            std: Nombre d'écarts-types
        
        Returns:
            dict: {'upper', 'middle', 'lower', 'position'}
        """
        try:
            middle = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            current_price = prices.iloc[-1]
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(middle.iloc[-1]),
                'lower': float(lower.iloc[-1]),
                'position': float(position)  # 0 = sur lower, 1 = sur upper
            }
        except:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5}
    
    @staticmethod
    def analyze_trend(prices: pd.Series, window: int = 20) -> str:
        """
        Analyser la tendance
        
        Args:
            prices: Série de prix
            window: Fenêtre d'analyse
        
        Returns:
            str: 'hausse', 'baisse', 'neutre'
        """
        try:
            if len(prices) < window:
                return 'neutre'
            
            sma = prices.rolling(window=window).mean()
            current_price = prices.iloc[-1]
            sma_value = sma.iloc[-1]
            
            # Pente
            slope = (sma.iloc[-1] - sma.iloc[-5]) / 5
            
            if current_price > sma_value and slope > 0:
                return 'hausse'
            elif current_price < sma_value and slope < 0:
                return 'baisse'
            else:
                return 'neutre'
        except:
            return 'neutre'
    
    @classmethod
    def explain_decision(cls, symbol: str, action: str, data: pd.DataFrame) -> Dict:
        """
        Générer explication pour une décision
        
        Args:
            symbol: Ticker
            action: 'BUY', 'SELL', 'HOLD'
            data: DataFrame avec colonnes OHLCV
        
        Returns:
            dict: {'reason', 'confidence', 'indicators'}
        """
        try:
            if len(data) < 20:
                return {
                    'reason': 'Données insuffisantes',
                    'confidence': 50,
                    'indicators': {}
                }
            
            prices = data['Close']
            
            # Calculer indicateurs
            rsi = cls.calculate_rsi(prices)
            macd_data = cls.calculate_macd(prices)
            bb = cls.calculate_bollinger_bands(prices)
            trend = cls.analyze_trend(prices)
            
            # Variation récente
            price_change_3d = ((prices.iloc[-1] / prices.iloc[-4]) - 1) * 100
            
            # Volume
            if 'Volume' in data.columns:
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = (current_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
            else:
                volume_ratio = 0
            
            # Construire explication
            reasons = []
            confidence = 50
            
            if action == 'BUY':
                # Signaux d'achat
                if rsi < 30:
                    reasons.append('Survente (RSI < 30)')
                    confidence += 15
                elif rsi < 40:
                    reasons.append('RSI bas')
                    confidence += 10
                
                if macd_data['histogram'] > 0:
                    reasons.append('MACD haussier')
                    confidence += 10
                
                if bb['position'] < 0.2:
                    reasons.append('Prix proche bande inférieure')
                    confidence += 10
                
                if trend == 'hausse':
                    reasons.append('Tendance haussière')
                    confidence += 10
                
                if price_change_3d < -3:
                    reasons.append(f'Correction {price_change_3d:.1f}%')
                    confidence += 5
                
                if volume_ratio > 20:
                    reasons.append('Volume élevé')
                    confidence += 5
            
            elif action == 'SELL':
                # Signaux de vente
                if rsi > 70:
                    reasons.append('Surachat (RSI > 70)')
                    confidence += 15
                elif rsi > 60:
                    reasons.append('RSI élevé')
                    confidence += 10
                
                if macd_data['histogram'] < 0:
                    reasons.append('MACD baissier')
                    confidence += 10
                
                if bb['position'] > 0.8:
                    reasons.append('Prix proche bande supérieure')
                    confidence += 10
                
                if trend == 'baisse':
                    reasons.append('Tendance baissière')
                    confidence += 10
                
                if price_change_3d > 5:
                    reasons.append(f'Prise de profit (+{price_change_3d:.1f}%)')
                    confidence += 5
            
            else:  # HOLD
                reasons.append('Aucun signal fort')
                confidence = 60
            
            # Limiter confiance
            confidence = min(95, max(30, confidence))
            
            # Formatter raison
            if reasons:
                reason = ', '.join(reasons[:3])  # Max 3 raisons
            else:
                reason = f"Décision modèle IA ({action})"
            
            # Indicateurs pour Discord
            indicators = {
                'RSI': f"{rsi:.0f} {'(survente)' if rsi < 30 else '(surachat)' if rsi > 70 else ''}",
                'MACD': 'Haussier' if macd_data['histogram'] > 0 else 'Baissier',
                'Tendance': trend.capitalize(),
                'Prix 3j': f"{price_change_3d:+.1f}%"
            }
            
            if volume_ratio != 0:
                indicators['Volume'] = f"{volume_ratio:+.0f}%"
            
            return {
                'reason': reason,
                'confidence': int(confidence),
                'indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {symbol}: {e}")
            return {
                'reason': f'Décision modèle IA ({action})',
                'confidence': 50,
                'indicators': {}
            }

# Helper pour tests
if __name__ == "__main__":
    import yfinance as yf
    
    print("📊 Test Market Analyzer...\n")
    
    # Télécharger données
    ticker = yf.Ticker('NVDA')
    data = ticker.history(period='1mo', interval='1d')
    
    # Test BUY
    explanation_buy = MarketAnalyzer.explain_decision('NVDA', 'BUY', data)
    print("🟢 BUY NVDA:")
    print(f"  Raison: {explanation_buy['reason']}")
    print(f"  Confiance: {explanation_buy['confidence']}%")
    print(f"  Indicateurs: {explanation_buy['indicators']}\n")
    
    # Test SELL
    explanation_sell = MarketAnalyzer.explain_decision('NVDA', 'SELL', data)
    print("🔴 SELL NVDA:")
    print(f"  Raison: {explanation_sell['reason']}")
    print(f"  Confiance: {explanation_sell['confidence']}%")
    print(f"  Indicateurs: {explanation_sell['indicators']}")
