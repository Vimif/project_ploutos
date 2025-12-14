#!/usr/bin/env python3
"""
🔍 PLOUTOS STOCK SCREENER

Scanne automatiquement des listes de tickers pour trouver les meilleures opportunités
Basé sur: RSI, MACD, Patterns, Volume, Tendance

Usage:
    screener = StockScreener()
    opportunities = screener.scan(tickers=['AAPL', 'NVDA', 'TSLA', ...])
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging

try:
    from web.utils.all_indicators import calculate_complete_indicators, get_indicator_signals
    ADVANCED_MODE = True
except:
    import ta
    ADVANCED_MODE = False

try:
    from web.utils.pattern_detector import PatternDetector
    PATTERN_MODE = True
except:
    PATTERN_MODE = False

logger = logging.getLogger(__name__)


class StockScreener:
    """
    Screener automatique multi-critères
    """
    
    def __init__(self):
        self.pattern_detector = PatternDetector() if PATTERN_MODE else None
        logger.info(f"🔍 Screener initialisé (Advanced: {ADVANCED_MODE}, Patterns: {PATTERN_MODE})")
    
    def scan(self, tickers: List[str], period: str = '3mo', 
             filters: Optional[Dict] = None) -> Dict:
        """
        Scanne une liste de tickers
        
        Args:
            tickers: Liste de symboles à analyser
            period: Période de données (1mo, 3mo, 6mo, 1y)
            filters: Critères de filtrage optionnels
                - min_rsi: RSI minimum (défaut: 0)
                - max_rsi: RSI maximum (défaut: 100)
                - min_volume_ratio: Ratio volume minimum (défaut: 1.0)
                - trend: 'bullish', 'bearish', ou None
        
        Returns:
            Dict avec buy_opportunities, sell_signals, watch_list
        """
        filters = filters or {}
        
        results = {
            'buy_opportunities': [],
            'sell_signals': [],
            'watch_list': [],
            'errors': [],
            'scan_time': datetime.now().isoformat(),
            'total_scanned': len(tickers)
        }
        
        logger.info(f"🔍 Démarrage scan de {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                analysis = self._analyze_single_ticker(ticker, period)
                
                if not analysis:
                    continue
                
                # Appliquer filtres
                if not self._passes_filters(analysis, filters):
                    continue
                
                # Classifier
                score = analysis['score']
                recommendation = analysis['recommendation']
                
                if score >= 70 and 'BUY' in recommendation:
                    results['buy_opportunities'].append(analysis)
                elif score <= 30 and 'SELL' in recommendation:
                    results['sell_signals'].append(analysis)
                elif 40 <= score <= 60:
                    results['watch_list'].append(analysis)
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur scan {ticker}: {e}")
                results['errors'].append({'ticker': ticker, 'error': str(e)})
        
        # Trier par score
        results['buy_opportunities'].sort(key=lambda x: x['score'], reverse=True)
        results['sell_signals'].sort(key=lambda x: x['score'])
        results['watch_list'].sort(key=lambda x: abs(x['score'] - 50))
        
        logger.info(f"✅ Scan terminé: {len(results['buy_opportunities'])} BUY, "
                   f"{len(results['sell_signals'])} SELL, {len(results['watch_list'])} WATCH")
        
        return results
    
    def _analyze_single_ticker(self, ticker: str, period: str) -> Optional[Dict]:
        """
        Analyse un ticker unique
        """
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty or len(df) < 20:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Calculer indicateurs
            if ADVANCED_MODE:
                indicators = calculate_complete_indicators(df)
                signals = get_indicator_signals(df, indicators)
            else:
                indicators = self._calculate_basic_indicators(df)
                signals = self._generate_basic_signals(df, indicators)
            
            # Détecter patterns
            bullish_patterns = 0
            bearish_patterns = 0
            latest_pattern = None
            
            if self.pattern_detector:
                try:
                    patterns = self.pattern_detector.detect_all_patterns(df)
                    candle_patterns = patterns.get('candlestick_patterns', [])
                    
                    for p in candle_patterns[-5:]:  # 5 derniers
                        if p['signal'] == 'BULLISH':
                            bullish_patterns += 1
                        elif p['signal'] == 'BEARISH':
                            bearish_patterns += 1
                        latest_pattern = p['type']
                except:
                    pass
            
            # Extraire métriques
            current_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            rsi = self._safe_get(indicators.get('rsi', []), 50)
            adx = self._safe_get(indicators.get('adx', []), 20)
            
            current_volume = float(df['Volume'].iloc[-1])
            avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else current_volume
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
            
            # Score global (0-100)
            score = self._calculate_score(signals, rsi, adx, volume_ratio, 
                                         bullish_patterns, bearish_patterns)
            
            recommendation = signals.get('overall', {}).get('recommendation', 'HOLD')
            confidence = signals.get('overall', {}).get('confidence', 50)
            
            return {
                'ticker': ticker,
                'price': round(current_price, 2),
                'change_pct': round(change_pct, 2),
                'rsi': round(rsi, 1),
                'adx': round(adx, 1),
                'volume_ratio': round(volume_ratio, 2),
                'score': int(score),
                'recommendation': recommendation,
                'confidence': int(confidence),
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'latest_pattern': latest_pattern,
                'trend': self._detect_trend(df, indicators),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse {ticker}: {e}")
            return None
    
    def _calculate_score(self, signals: dict, rsi: float, adx: float, 
                        volume_ratio: float, bullish_pat: int, bearish_pat: int) -> float:
        """
        Calcule un score global 0-100
        """
        score = 50  # Neutre
        
        # RSI (poids 25%)
        if rsi < 30:
            score += 12.5  # Survendu = opportunité
        elif rsi > 70:
            score -= 12.5  # Suracheté = risque
        else:
            score += (50 - rsi) * 0.25  # Linéaire
        
        # Tendance (poids 30%)
        trend_signals = signals.get('trend', {})
        if trend_signals.get('sma', {}).get('signal', '') == 'STRONG_BUY':
            score += 15
        elif 'BUY' in trend_signals.get('sma', {}).get('signal', ''):
            score += 10
        elif 'SELL' in trend_signals.get('sma', {}).get('signal', ''):
            score -= 10
        
        # MACD
        if trend_signals.get('macd', {}).get('signal', '') == 'STRONG_BUY':
            score += 15
        elif 'BUY' in trend_signals.get('macd', {}).get('signal', ''):
            score += 10
        elif 'SELL' in trend_signals.get('macd', {}).get('signal', ''):
            score -= 10
        
        # ADX (force) - poids 15%
        if adx > 25:
            # Tendance forte, amplifier le signal
            if score > 50:
                score += 7.5
            else:
                score -= 7.5
        
        # Volume (poids 15%)
        if volume_ratio > 1.5:
            score += 7.5
        elif volume_ratio < 0.7:
            score -= 7.5
        
        # Patterns (poids 15%)
        pattern_score = (bullish_pat - bearish_pat) * 3
        score += min(max(pattern_score, -7.5), 7.5)
        
        return max(0, min(100, score))
    
    def _detect_trend(self, df: pd.DataFrame, indicators: dict) -> str:
        """
        Détecte la tendance globale
        """
        sma_20 = self._safe_get(indicators.get('sma_20', []), None)
        sma_50 = self._safe_get(indicators.get('sma_50', []), None)
        price = float(df['Close'].iloc[-1])
        
        if sma_20 and sma_50:
            if price > sma_20 > sma_50:
                return 'STRONG_UPTREND'
            elif price > sma_20:
                return 'UPTREND'
            elif price < sma_20 < sma_50:
                return 'STRONG_DOWNTREND'
            elif price < sma_20:
                return 'DOWNTREND'
        
        return 'SIDEWAYS'
    
    def _passes_filters(self, analysis: dict, filters: dict) -> bool:
        """
        Vérifie si l'analyse passe les filtres
        """
        if 'min_rsi' in filters and analysis['rsi'] < filters['min_rsi']:
            return False
        
        if 'max_rsi' in filters and analysis['rsi'] > filters['max_rsi']:
            return False
        
        if 'min_volume_ratio' in filters and analysis['volume_ratio'] < filters['min_volume_ratio']:
            return False
        
        if 'trend' in filters:
            if filters['trend'] == 'bullish' and 'UPTREND' not in analysis['trend']:
                return False
            if filters['trend'] == 'bearish' and 'DOWNTREND' not in analysis['trend']:
                return False
        
        return True
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> dict:
        """Fallback indicateurs basiques"""
        import ta
        indicators = {}
        
        if len(df) >= 20:
            indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20).tolist()
        if len(df) >= 50:
            indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50).tolist()
        if len(df) >= 14:
            indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).tolist()
            indicators['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close']).tolist()
        
        return indicators
    
    def _generate_basic_signals(self, df: pd.DataFrame, indicators: dict) -> dict:
        """Génère signaux basiques"""
        signals = {'trend': {}, 'overall': {'recommendation': 'HOLD', 'confidence': 50}}
        
        price = float(df['Close'].iloc[-1])
        sma_20 = self._safe_get(indicators.get('sma_20', []), None)
        sma_50 = self._safe_get(indicators.get('sma_50', []), None)
        
        if sma_20 and sma_50:
            if price > sma_20 > sma_50:
                signals['trend']['sma'] = {'signal': 'STRONG_BUY'}
                signals['overall']['recommendation'] = 'BUY'
                signals['overall']['confidence'] = 75
            elif price > sma_20:
                signals['trend']['sma'] = {'signal': 'BUY'}
                signals['overall']['recommendation'] = 'BUY'
                signals['overall']['confidence'] = 60
            elif price < sma_20 < sma_50:
                signals['trend']['sma'] = {'signal': 'STRONG_SELL'}
                signals['overall']['recommendation'] = 'SELL'
                signals['overall']['confidence'] = 75
        
        return signals
    
    def _safe_get(self, arr, default=0):
        """Récupère dernière valeur valide"""
        if not arr or len(arr) == 0:
            return default
        val = arr[-1]
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)


if __name__ == '__main__':
    # Test
    screener = StockScreener()
    
    test_tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    results = screener.scan(test_tickers, period='3mo')
    
    print("\n🟢 BUY OPPORTUNITIES:")
    for stock in results['buy_opportunities'][:5]:
        print(f"  {stock['ticker']}: Score {stock['score']} | RSI {stock['rsi']} | {stock['recommendation']}")
    
    print("\n🔴 SELL SIGNALS:")
    for stock in results['sell_signals'][:5]:
        print(f"  {stock['ticker']}: Score {stock['score']} | RSI {stock['rsi']} | {stock['recommendation']}")
