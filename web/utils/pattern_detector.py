#!/usr/bin/env python3
"""
📊 PLOUTOS PATTERN DETECTOR - TRADER PRO

Détecte automatiquement :
- Patterns de chandeliers japonais (doji, hammer, engulfing...)
- Patterns graphiques (triangles, head & shoulders...)
- Niveaux de Fibonacci
- Support/Résistance dynamiques

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class PatternDetector:
    """
    Détecteur professionnel de patterns de trading
    """
    
    def __init__(self):
        self.min_pattern_bars = 3
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Détecte TOUS les patterns sur un DataFrame
        """
        if len(df) < 10:
            return {'error': 'Pas assez de données'}
        
        results = {
            'candlestick_patterns': self.detect_candlestick_patterns(df),
            'chart_patterns': self.detect_chart_patterns(df),
            'fibonacci': self.calculate_fibonacci_levels(df),
            'support_resistance': self.find_support_resistance(df)
        }
        
        return results
    
    # ========== CANDLESTICK PATTERNS ==========
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Détecte les patterns de chandeliers japonais
        """
        patterns = []
        
        for i in range(len(df) - 3):
            # Doji
            if self._is_doji(df.iloc[i]):
                patterns.append({
                    'type': 'DOJI',
                    'index': i,
                    'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'signal': 'NEUTRAL',
                    'strength': 0.5,
                    'description': 'Indécision du marché'
                })
            
            # Hammer
            if self._is_hammer(df.iloc[i]):
                patterns.append({
                    'type': 'HAMMER',
                    'index': i,
                    'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'signal': 'BULLISH',
                    'strength': 0.7,
                    'description': 'Retournement haussier possible'
                })
            
            # Shooting Star
            if self._is_shooting_star(df.iloc[i]):
                patterns.append({
                    'type': 'SHOOTING_STAR',
                    'index': i,
                    'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'signal': 'BEARISH',
                    'strength': 0.7,
                    'description': 'Retournement baissier possible'
                })
            
            # Engulfing (nécessite 2 bougies)
            if i < len(df) - 1:
                engulfing = self._is_engulfing(df.iloc[i], df.iloc[i+1])
                if engulfing:
                    patterns.append({
                        'type': f'{engulfing}_ENGULFING',
                        'index': i+1,
                        'date': df.index[i+1].isoformat() if hasattr(df.index[i+1], 'isoformat') else str(df.index[i+1]),
                        'signal': engulfing,
                        'strength': 0.8,
                        'description': f'Englobante {engulfing.lower()} - Signal fort'
                    })
            
            # Morning/Evening Star (nécessite 3 bougies)
            if i < len(df) - 2:
                star = self._is_star_pattern(df.iloc[i], df.iloc[i+1], df.iloc[i+2])
                if star:
                    patterns.append({
                        'type': f'{star}_STAR',
                        'index': i+2,
                        'date': df.index[i+2].isoformat() if hasattr(df.index[i+2], 'isoformat') else str(df.index[i+2]),
                        'signal': star,
                        'strength': 0.9,
                        'description': f'Étoile {star.lower()} - Signal très fort'
                    })
        
        # Garder seulement les 20 derniers patterns
        return patterns[-20:] if patterns else []
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Détecte un Doji"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            range_hl = candle['High'] - candle['Low']
            return body <= range_hl * 0.1 and range_hl > 0
        except:
            return False
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Détecte un Hammer"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
            upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
            
            return (lower_shadow > body * 2 and 
                    upper_shadow < body * 0.3 and
                    body > 0)
        except:
            return False
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Détecte une Shooting Star"""
        try:
            body = abs(candle['Close'] - candle['Open'])
            lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
            upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
            
            return (upper_shadow > body * 2 and 
                    lower_shadow < body * 0.3 and
                    body > 0)
        except:
            return False
    
    def _is_engulfing(self, prev: pd.Series, curr: pd.Series) -> Optional[str]:
        """Détecte un pattern Engulfing"""
        try:
            prev_body = abs(prev['Close'] - prev['Open'])
            curr_body = abs(curr['Close'] - curr['Open'])
            
            # Bullish Engulfing
            if (prev['Close'] < prev['Open'] and  # Prev baissière
                curr['Close'] > curr['Open'] and  # Curr haussière
                curr['Open'] < prev['Close'] and  # Ouvre en dessous
                curr['Close'] > prev['Open'] and  # Ferme au-dessus
                curr_body > prev_body * 1.2):     # Corps plus grand
                return 'BULLISH'
            
            # Bearish Engulfing
            if (prev['Close'] > prev['Open'] and  # Prev haussière
                curr['Close'] < curr['Open'] and  # Curr baissière
                curr['Open'] > prev['Close'] and  # Ouvre au-dessus
                curr['Close'] < prev['Open'] and  # Ferme en-dessous
                curr_body > prev_body * 1.2):     # Corps plus grand
                return 'BEARISH'
            
            return None
        except:
            return None
    
    def _is_star_pattern(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> Optional[str]:
        """Détecte Morning/Evening Star"""
        try:
            # Morning Star (retournement haussier)
            if (c1['Close'] < c1['Open'] and  # Bougie 1 baissière
                abs(c2['Close'] - c2['Open']) < (c1['High'] - c1['Low']) * 0.3 and  # Bougie 2 petit corps
                c3['Close'] > c3['Open'] and  # Bougie 3 haussière
                c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Ferme au-dessus du milieu de c1
                return 'MORNING'
            
            # Evening Star (retournement baissier)
            if (c1['Close'] > c1['Open'] and  # Bougie 1 haussière
                abs(c2['Close'] - c2['Open']) < (c1['High'] - c1['Low']) * 0.3 and  # Bougie 2 petit corps
                c3['Close'] < c3['Open'] and  # Bougie 3 baissière
                c3['Close'] < (c1['Open'] + c1['Close']) / 2):  # Ferme en-dessous du milieu de c1
                return 'EVENING'
            
            return None
        except:
            return None
    
    # ========== CHART PATTERNS ==========
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Détecte les patterns graphiques (triangles, channels...)
        """
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Double Top/Bottom
        double_pattern = self._detect_double_pattern(df)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Triangle
        triangle = self._detect_triangle(df)
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def _detect_double_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Détecte Double Top/Bottom"""
        try:
            window = min(30, len(df))
            recent = df.tail(window)
            
            # Trouver les pics
            highs = recent['High'].nlargest(2)
            lows = recent['Low'].nsmallest(2)
            
            # Double Top
            if len(highs) == 2:
                diff = abs(highs.iloc[0] - highs.iloc[1]) / highs.iloc[0]
                if diff < 0.02:  # <2% de différence
                    return {
                        'type': 'DOUBLE_TOP',
                        'signal': 'BEARISH',
                        'strength': 0.7,
                        'levels': [float(highs.iloc[0]), float(highs.iloc[1])],
                        'description': 'Double sommet - Retournement baissier'
                    }
            
            # Double Bottom
            if len(lows) == 2:
                diff = abs(lows.iloc[0] - lows.iloc[1]) / lows.iloc[0]
                if diff < 0.02:
                    return {
                        'type': 'DOUBLE_BOTTOM',
                        'signal': 'BULLISH',
                        'strength': 0.7,
                        'levels': [float(lows.iloc[0]), float(lows.iloc[1])],
                        'description': 'Double creux - Retournement haussier'
                    }
            
            return None
        except:
            return None
    
    def _detect_triangle(self, df: pd.DataFrame) -> Optional[Dict]:
        """Détecte les triangles"""
        try:
            window = min(40, len(df))
            recent = df.tail(window)
            
            # Calculer les trendlines
            highs = recent['High'].values
            lows = recent['Low'].values
            
            # Pente des plus hauts
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            # Pente des plus bas
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Triangle ascendant
            if abs(high_slope) < 0.01 and low_slope > 0.01:
                return {
                    'type': 'ASCENDING_TRIANGLE',
                    'signal': 'BULLISH',
                    'strength': 0.6,
                    'description': 'Triangle ascendant - Continuation haussière probable'
                }
            
            # Triangle descendant
            if abs(low_slope) < 0.01 and high_slope < -0.01:
                return {
                    'type': 'DESCENDING_TRIANGLE',
                    'signal': 'BEARISH',
                    'strength': 0.6,
                    'description': 'Triangle descendant - Continuation baissière probable'
                }
            
            # Triangle symétrique
            if high_slope < -0.01 and low_slope > 0.01:
                return {
                    'type': 'SYMMETRICAL_TRIANGLE',
                    'signal': 'NEUTRAL',
                    'strength': 0.5,
                    'description': 'Triangle symétrique - Breakout imminent'
                }
            
            return None
        except:
            return None
    
    # ========== FIBONACCI ==========
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calcule les niveaux de Fibonacci
        """
        try:
            window = min(100, len(df))
            recent = df.tail(window)
            
            high = float(recent['High'].max())
            low = float(recent['Low'].min())
            diff = high - low
            
            # Retracements
            retracements = {
                '0.0': high,
                '23.6': high - diff * 0.236,
                '38.2': high - diff * 0.382,
                '50.0': high - diff * 0.500,
                '61.8': high - diff * 0.618,
                '78.6': high - diff * 0.786,
                '100.0': low
            }
            
            # Extensions
            extensions = {
                '127.2': high + diff * 0.272,
                '161.8': high + diff * 0.618,
                '200.0': high + diff * 1.000,
                '261.8': high + diff * 1.618
            }
            
            # Prix actuel
            current_price = float(df['Close'].iloc[-1])
            
            # Trouver le niveau le plus proche
            all_levels = {**retracements, **extensions}
            closest_level = min(all_levels.items(), key=lambda x: abs(x[1] - current_price))
            
            return {
                'high': high,
                'low': low,
                'range': diff,
                'retracements': retracements,
                'extensions': extensions,
                'current_price': current_price,
                'closest_level': {
                    'name': closest_level[0],
                    'price': closest_level[1],
                    'distance_pct': ((closest_level[1] - current_price) / current_price) * 100
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ========== SUPPORT/RESISTANCE ==========
    
    def find_support_resistance(self, df: pd.DataFrame, num_levels: int = 5) -> Dict:
        """
        Trouve les niveaux de support et résistance dynamiques
        """
        try:
            window = min(100, len(df))
            recent = df.tail(window)
            
            # Méthode : Clustering des prix autour des zones de forte activité
            all_prices = pd.concat([recent['High'], recent['Low'], recent['Close']])
            
            # Histogramme des prix
            hist, bins = np.histogram(all_prices, bins=50)
            
            # Trouver les pics (zones de forte activité)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                    price = (bins[i] + bins[i+1]) / 2
                    peaks.append((price, hist[i]))
            
            # Trier par importance
            peaks.sort(key=lambda x: x[1], reverse=True)
            peaks = peaks[:num_levels]
            
            # Classifier en support/résistance
            current_price = float(df['Close'].iloc[-1])
            
            supports = [p[0] for p in peaks if p[0] < current_price]
            resistances = [p[0] for p in peaks if p[0] > current_price]
            
            return {
                'supports': sorted(supports, reverse=True)[:3],  # 3 plus proches
                'resistances': sorted(resistances)[:3],  # 3 plus proches
                'current_price': current_price
            }
        except Exception as e:
            return {'error': str(e), 'supports': [], 'resistances': []}
