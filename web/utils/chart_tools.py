#!/usr/bin/env python3
"""
📊 CHART TOOLS - Fibonacci, Volume Profile, Smart Annotations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ChartTools:
    """
    Outils avancés pour analyse chartiste
    """
    
    @staticmethod
    def calculate_fibonacci(df: pd.DataFrame, lookback: int = 90) -> Dict:
        """
        Calcule niveaux Fibonacci automatiquement
        
        Args:
            df: DataFrame avec OHLC
            lookback: Nombre de jours pour trouver swing high/low
            
        Returns:
            Dict avec niveaux Fibonacci
        """
        # Trouver swing high et swing low
        data = df.tail(lookback)
        
        swing_high = data['High'].max()
        swing_low = data['Low'].min()
        
        # Trouver les dates
        high_date = data['High'].idxmax()
        low_date = data['Low'].idxmin()
        
        # Déterminer la tendance (high avant low = baisse, sinon hausse)
        trend = 'down' if high_date < low_date else 'up'
        
        # Calculer niveaux
        diff = swing_high - swing_low
        
        if trend == 'up':
            levels = {
                '0.0': {'price': swing_low, 'label': '0% (Low)'},
                '23.6': {'price': swing_low + diff * 0.236, 'label': '23.6%'},
                '38.2': {'price': swing_low + diff * 0.382, 'label': '38.2%'},
                '50.0': {'price': swing_low + diff * 0.5, 'label': '50%'},
                '61.8': {'price': swing_low + diff * 0.618, 'label': '61.8%'},
                '78.6': {'price': swing_low + diff * 0.786, 'label': '78.6%'},
                '100.0': {'price': swing_high, 'label': '100% (High)'},
                '161.8': {'price': swing_low + diff * 1.618, 'label': '161.8% (Extension)'},
                '261.8': {'price': swing_low + diff * 2.618, 'label': '261.8% (Extension)'}
            }
        else:
            levels = {
                '0.0': {'price': swing_high, 'label': '0% (High)'},
                '23.6': {'price': swing_high - diff * 0.236, 'label': '23.6%'},
                '38.2': {'price': swing_high - diff * 0.382, 'label': '38.2%'},
                '50.0': {'price': swing_high - diff * 0.5, 'label': '50%'},
                '61.8': {'price': swing_high - diff * 0.618, 'label': '61.8%'},
                '78.6': {'price': swing_high - diff * 0.786, 'label': '78.6%'},
                '100.0': {'price': swing_low, 'label': '100% (Low)'},
                '161.8': {'price': swing_high - diff * 1.618, 'label': '161.8% (Extension)'},
                '261.8': {'price': swing_high - diff * 2.618, 'label': '261.8% (Extension)'}
            }
        
        return {
            'trend': trend,
            'swing_high': float(swing_high),
            'swing_low': float(swing_low),
            'high_date': high_date.isoformat(),
            'low_date': low_date.isoformat(),
            'levels': {k: {'price': float(v['price']), 'label': v['label']} for k, v in levels.items()}
        }
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
        """
        Calcule Volume Profile (histogram de volume par niveau de prix)
        
        Args:
            df: DataFrame avec OHLC + Volume
            bins: Nombre de niveaux de prix
            
        Returns:
            Dict avec profil de volume
        """
        # Calculer prix min/max
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        # Créer bins de prix
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Pour chaque chandelier, distribuer le volume sur les bins touchés
        volume_profile = np.zeros(bins)
        
        for idx, row in df.iterrows():
            # Trouver les bins touchés par ce chandelier
            low_bin = np.digitize(row['Low'], price_bins) - 1
            high_bin = np.digitize(row['High'], price_bins) - 1
            
            # Limiter aux indices valides
            low_bin = max(0, min(low_bin, bins - 1))
            high_bin = max(0, min(high_bin, bins - 1))
            
            # Distribuer le volume uniformément sur les bins touchés
            bins_touched = high_bin - low_bin + 1
            volume_per_bin = row['Volume'] / bins_touched if bins_touched > 0 else row['Volume']
            
            for b in range(low_bin, high_bin + 1):
                volume_profile[b] += volume_per_bin
        
        # Trouver le Point of Control (POC) = prix avec le plus de volume
        poc_bin = np.argmax(volume_profile)
        poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
        
        # Calculer Value Area (70% du volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        # Commencer au POC et étendre de chaque côté
        va_bins = [poc_bin]
        va_volume = volume_profile[poc_bin]
        
        left = poc_bin - 1
        right = poc_bin + 1
        
        while va_volume < target_volume and (left >= 0 or right < bins):
            left_vol = volume_profile[left] if left >= 0 else 0
            right_vol = volume_profile[right] if right < bins else 0
            
            if left_vol > right_vol and left >= 0:
                va_bins.append(left)
                va_volume += left_vol
                left -= 1
            elif right < bins:
                va_bins.append(right)
                va_volume += right_vol
                right += 1
            else:
                break
        
        va_low = price_bins[min(va_bins)]
        va_high = price_bins[max(va_bins) + 1]
        
        # Formater résultats
        profile_data = []
        for i in range(bins):
            profile_data.append({
                'price': float((price_bins[i] + price_bins[i + 1]) / 2),
                'volume': float(volume_profile[i]),
                'is_poc': i == poc_bin,
                'in_value_area': i in va_bins
            })
        
        return {
            'profile': profile_data,
            'poc_price': float(poc_price),
            'value_area_high': float(va_high),
            'value_area_low': float(va_low),
            'total_volume': float(total_volume)
        }
    
    @staticmethod
    def generate_smart_annotations(df: pd.DataFrame, signals: Dict, patterns: Dict = None) -> List[Dict]:
        """
        Génère annotations intelligentes sur le chart
        
        Args:
            df: DataFrame OHLC
            signals: Dict de signaux techniques
            patterns: Dict de patterns détectés
            
        Returns:
            Liste d'annotations
        """
        annotations = []
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]
        
        # RSI oversold/overbought
        if 'overall' in signals:
            rec = signals['overall'].get('recommendation', 'HOLD')
            conf = signals['overall'].get('confidence', 50)
            
            if rec == 'STRONG BUY' and conf > 70:
                annotations.append({
                    'date': current_date.isoformat(),
                    'price': float(current_price),
                    'type': 'buy',
                    'text': f'🟢 STRONG BUY ({conf}%)',
                    'color': '#00f260'
                })
            elif rec == 'STRONG SELL' and conf > 70:
                annotations.append({
                    'date': current_date.isoformat(),
                    'price': float(current_price),
                    'type': 'sell',
                    'text': f'🔴 STRONG SELL ({conf}%)',
                    'color': '#ff4757'
                })
        
        # Patterns chartistes
        if patterns:
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    annotations.append({
                        'date': current_date.isoformat(),
                        'price': float(current_price),
                        'type': 'pattern',
                        'text': f'⭐ {pattern_name}',
                        'color': '#ffd700'
                    })
        
        # Volume spike
        if len(df) >= 20:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 3:
                annotations.append({
                    'date': current_date.isoformat(),
                    'price': float(current_price),
                    'type': 'volume',
                    'text': f'📈 Volume Spike {int(current_volume / avg_volume)}x',
                    'color': '#ff6348'
                })
        
        # Support/Resistance touché
        # Calculer supports/résistances sur 90 jours
        if len(df) >= 90:
            data_90d = df.tail(90)
            resistance = data_90d['High'].quantile(0.95)
            support = data_90d['Low'].quantile(0.05)
            
            # Si prix proche support (+/- 1%)
            if abs(current_price - support) / support < 0.01:
                annotations.append({
                    'date': current_date.isoformat(),
                    'price': float(support),
                    'type': 'support',
                    'text': f'🟢 Support touché (${support:.2f})',
                    'color': '#00f260'
                })
            
            # Si prix proche résistance (+/- 1%)
            if abs(current_price - resistance) / resistance < 0.01:
                annotations.append({
                    'date': current_date.isoformat(),
                    'price': float(resistance),
                    'type': 'resistance',
                    'text': f'🔴 Résistance touchée (${resistance:.2f})',
                    'color': '#ff4757'
                })
        
        return annotations
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Détecte supports et résistances clés
        
        Args:
            df: DataFrame OHLC
            window: Fenêtre pour détection
            
        Returns:
            Dict avec supports/résistances
        """
        supports = []
        resistances = []
        
        # Méthode : trouver les pivots locaux
        for i in range(window, len(df) - window):
            # Support = low local
            if df['Low'].iloc[i] == df['Low'].iloc[i - window:i + window + 1].min():
                supports.append({
                    'price': float(df['Low'].iloc[i]),
                    'date': df.index[i].isoformat(),
                    'strength': 1  # Peut être amélioré avec nombre de touches
                })
            
            # Résistance = high local
            if df['High'].iloc[i] == df['High'].iloc[i - window:i + window + 1].max():
                resistances.append({
                    'price': float(df['High'].iloc[i]),
                    'date': df.index[i].isoformat(),
                    'strength': 1
                })
        
        # Regrouper les niveaux proches (clustering)
        def cluster_levels(levels, threshold=0.02):
            if not levels:
                return []
            
            # Trier par prix
            levels = sorted(levels, key=lambda x: x['price'])
            
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # Si prix proche du cluster actuel (< threshold%)
                if abs(level['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] < threshold:
                    current_cluster.append(level)
                else:
                    # Moyenne du cluster
                    avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                    clustered.append({
                        'price': avg_price,
                        'strength': len(current_cluster),
                        'touches': len(current_cluster)
                    })
                    current_cluster = [level]
            
            # Dernier cluster
            if current_cluster:
                avg_price = sum(l['price'] for l in current_cluster) / len(current_cluster)
                clustered.append({
                    'price': avg_price,
                    'strength': len(current_cluster),
                    'touches': len(current_cluster)
                })
            
            return clustered
        
        supports_clustered = cluster_levels(supports)
        resistances_clustered = cluster_levels(resistances)
        
        # Garder seulement les 5 plus forts de chaque
        supports_clustered = sorted(supports_clustered, key=lambda x: x['strength'], reverse=True)[:5]
        resistances_clustered = sorted(resistances_clustered, key=lambda x: x['strength'], reverse=True)[:5]
        
        return {
            'supports': supports_clustered,
            'resistances': resistances_clustered
        }


if __name__ == '__main__':
    # Test
    import yfinance as yf
    
    df = yf.download('AAPL', period='6mo', progress=False)
    tools = ChartTools()
    
    print("📊 FIBONACCI")
    fib = tools.calculate_fibonacci(df)
    print(f"Trend: {fib['trend']}")
    print(f"Swing High: ${fib['swing_high']:.2f}")
    print(f"Swing Low: ${fib['swing_low']:.2f}")
    for level, data in fib['levels'].items():
        print(f"  {data['label']}: ${data['price']:.2f}")
    
    print("\n📉 VOLUME PROFILE")
    vp = tools.calculate_volume_profile(df)
    print(f"POC: ${vp['poc_price']:.2f}")
    print(f"Value Area: ${vp['value_area_low']:.2f} - ${vp['value_area_high']:.2f}")
    
    print("\n🎯 SUPPORT/RESISTANCE")
    sr = tools.detect_support_resistance(df)
    print("Supports:")
    for s in sr['supports']:
        print(f"  ${s['price']:.2f} (strength: {s['strength']})")
    print("Resistances:")
    for r in sr['resistances']:
        print(f"  ${r['price']:.2f} (strength: {r['strength']})")
