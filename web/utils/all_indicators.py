#!/usr/bin/env python3
"""
📊 PLOUTOS - COMPLETE TECHNICAL INDICATORS LIBRARY

Tous les indicateurs techniques professionnels
Gestion ULTRA-ROBUSTE des données insuffisantes
"""

import pandas as pd
import numpy as np
import ta
import warnings
import traceback
warnings.filterwarnings('ignore')


def safe_calculate(func, *args, default=None, **kwargs):
    """
    Wrapper ultra-sécurisé pour tous les calculs
    """
    try:
        result = func(*args, **kwargs)
        if result is None:
            return default
        
        # Convertir en liste et nettoyer
        if hasattr(result, 'tolist'):
            arr = result.tolist()
        elif isinstance(result, pd.Series):
            arr = result.tolist()
        else:
            arr = result
        
        # Remplacer NaN/Inf par None
        cleaned = []
        for val in arr:
            if val is None:
                cleaned.append(None)
            elif isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    cleaned.append(None)
                else:
                    cleaned.append(float(val))
            else:
                cleaned.append(val)
        
        return cleaned
    except Exception as e:
        # Silencieux mais logué
        # print(f"[WARN] {func.__name__}: {e}")
        return default


def calculate_complete_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule TOUS les indicateurs techniques avec gestion ULTRA-ROBUSTE des erreurs
    """
    indicators = {}
    n = len(df)
    
    if n < 5:
        return {'error': 'Pas assez de données (min 5 périodes)'}
    
    # ========== TREND INDICATORS ==========
    
    # SMA
    for period in [10, 20, 50, 100, 200]:
        if n >= period:
            indicators[f'sma_{period}'] = safe_calculate(
                ta.trend.sma_indicator, df['Close'], window=period, default=[None]*n
            )
    
    # EMA
    for period in [9, 12, 20, 26, 50]:
        if n >= period:
            indicators[f'ema_{period}'] = safe_calculate(
                ta.trend.ema_indicator, df['Close'], window=period, default=[None]*n
            )
    
    # WMA
    if n >= 20:
        indicators['wma_20'] = safe_calculate(
            ta.trend.wma_indicator, df['Close'], window=20, default=[None]*n
        )
    
    # MACD
    if n >= 26:
        try:
            macd = ta.trend.MACD(df['Close'])
            indicators['macd'] = safe_calculate(macd.macd, default=[None]*n)
            indicators['macd_signal'] = safe_calculate(macd.macd_signal, default=[None]*n)
            indicators['macd_hist'] = safe_calculate(macd.macd_diff, default=[None]*n)
        except:
            pass
    
    # ADX
    if n >= 14:
        indicators['adx'] = safe_calculate(
            ta.trend.adx, df['High'], df['Low'], df['Close'], window=14, default=[None]*n
        )
        indicators['adx_pos'] = safe_calculate(
            ta.trend.adx_pos, df['High'], df['Low'], df['Close'], window=14, default=[None]*n
        )
        indicators['adx_neg'] = safe_calculate(
            ta.trend.adx_neg, df['High'], df['Low'], df['Close'], window=14, default=[None]*n
        )
    
    # Parabolic SAR
    if n >= 5:
        try:
            sar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
            indicators['sar'] = safe_calculate(sar.psar, default=[None]*n)
            indicators['sar_up'] = safe_calculate(sar.psar_up, default=[None]*n)
            indicators['sar_down'] = safe_calculate(sar.psar_down, default=[None]*n)
        except:
            pass
    
    # Ichimoku
    if n >= 52:
        try:
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            indicators['ichimoku_a'] = safe_calculate(ichimoku.ichimoku_a, default=[None]*n)
            indicators['ichimoku_b'] = safe_calculate(ichimoku.ichimoku_b, default=[None]*n)
            indicators['ichimoku_base'] = safe_calculate(ichimoku.ichimoku_base_line, default=[None]*n)
            indicators['ichimoku_conv'] = safe_calculate(ichimoku.ichimoku_conversion_line, default=[None]*n)
        except:
            pass
    
    # Aroon
    if n >= 25:
        try:
            aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
            indicators['aroon_up'] = safe_calculate(aroon.aroon_up, default=[None]*n)
            indicators['aroon_down'] = safe_calculate(aroon.aroon_down, default=[None]*n)
            indicators['aroon_indicator'] = safe_calculate(aroon.aroon_indicator, default=[None]*n)
        except:
            pass
    
    # CCI
    if n >= 20:
        indicators['cci'] = safe_calculate(
            ta.trend.cci, df['High'], df['Low'], df['Close'], window=20, default=[None]*n
        )
    
    # DPO
    if n >= 20:
        indicators['dpo'] = safe_calculate(
            ta.trend.dpo, df['Close'], window=20, default=[None]*n
        )
    
    # ========== MOMENTUM INDICATORS ==========
    
    # RSI
    if n >= 14:
        indicators['rsi'] = safe_calculate(
            ta.momentum.rsi, df['Close'], window=14, default=[50.0]*n
        )
    
    if n >= 6:
        indicators['rsi_6'] = safe_calculate(
            ta.momentum.rsi, df['Close'], window=6, default=[50.0]*n
        )
    
    if n >= 21:
        indicators['rsi_21'] = safe_calculate(
            ta.momentum.rsi, df['Close'], window=21, default=[50.0]*n
        )
    
    # Stochastic
    if n >= 14:
        try:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            indicators['stoch_k'] = safe_calculate(stoch.stoch, default=[None]*n)
            indicators['stoch_d'] = safe_calculate(stoch.stoch_signal, default=[None]*n)
        except:
            pass
    
    # Stochastic RSI
    if n >= 14:
        try:
            stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'])
            indicators['stoch_rsi'] = safe_calculate(stoch_rsi.stochrsi, default=[None]*n)
            indicators['stoch_rsi_k'] = safe_calculate(stoch_rsi.stochrsi_k, default=[None]*n)
            indicators['stoch_rsi_d'] = safe_calculate(stoch_rsi.stochrsi_d, default=[None]*n)
        except:
            pass
    
    # Williams %R
    if n >= 14:
        indicators['williams_r'] = safe_calculate(
            ta.momentum.williams_r, df['High'], df['Low'], df['Close'], default=[None]*n
        )
    
    # ROC
    if n >= 12:
        indicators['roc'] = safe_calculate(
            ta.momentum.roc, df['Close'], window=12, default=[None]*n
        )
    
    # TSI
    if n >= 25:
        indicators['tsi'] = safe_calculate(
            ta.momentum.tsi, df['Close'], default=[None]*n
        )
    
    # Ultimate Oscillator
    if n >= 28:
        indicators['uo'] = safe_calculate(
            ta.momentum.ultimate_oscillator, df['High'], df['Low'], df['Close'], default=[None]*n
        )
    
    # Awesome Oscillator
    if n >= 34:
        indicators['ao'] = safe_calculate(
            ta.momentum.awesome_oscillator, df['High'], df['Low'], default=[None]*n
        )
    
    # KAMA
    if n >= 10:
        indicators['kama'] = safe_calculate(
            ta.momentum.kama, df['Close'], default=[None]*n
        )
    
    # ========== VOLATILITY INDICATORS ==========
    
    # Bollinger Bands
    if n >= 20:
        try:
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            indicators['bb_upper'] = safe_calculate(bb.bollinger_hband, default=[None]*n)
            indicators['bb_middle'] = safe_calculate(bb.bollinger_mavg, default=[None]*n)
            indicators['bb_lower'] = safe_calculate(bb.bollinger_lband, default=[None]*n)
            indicators['bb_width'] = safe_calculate(bb.bollinger_wband, default=[None]*n)
            indicators['bb_percent'] = safe_calculate(bb.bollinger_pband, default=[None]*n)
        except:
            pass
    
    # ATR
    if n >= 14:
        indicators['atr'] = safe_calculate(
            ta.volatility.average_true_range, df['High'], df['Low'], df['Close'], window=14, default=[None]*n
        )
    
    # Keltner Channels
    if n >= 20:
        try:
            kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            indicators['kc_upper'] = safe_calculate(kc.keltner_channel_hband, default=[None]*n)
            indicators['kc_middle'] = safe_calculate(kc.keltner_channel_mband, default=[None]*n)
            indicators['kc_lower'] = safe_calculate(kc.keltner_channel_lband, default=[None]*n)
        except:
            pass
    
    # Donchian Channels
    if n >= 20:
        try:
            dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
            indicators['dc_upper'] = safe_calculate(dc.donchian_channel_hband, default=[None]*n)
            indicators['dc_middle'] = safe_calculate(dc.donchian_channel_mband, default=[None]*n)
            indicators['dc_lower'] = safe_calculate(dc.donchian_channel_lband, default=[None]*n)
        except:
            pass
    
    # Ulcer Index
    if n >= 14:
        indicators['ui'] = safe_calculate(
            ta.volatility.ulcer_index, df['Close'], default=[None]*n
        )
    
    # ========== VOLUME INDICATORS ==========
    
    # OBV
    indicators['obv'] = safe_calculate(
        ta.volume.on_balance_volume, df['Close'], df['Volume'], default=[None]*n
    )
    
    # CMF
    if n >= 20:
        indicators['cmf'] = safe_calculate(
            ta.volume.chaikin_money_flow, df['High'], df['Low'], df['Close'], df['Volume'], default=[None]*n
        )
    
    # MFI
    if n >= 14:
        indicators['mfi'] = safe_calculate(
            ta.volume.money_flow_index, df['High'], df['Low'], df['Close'], df['Volume'], default=[None]*n
        )
    
    # ADI
    indicators['adi'] = safe_calculate(
        ta.volume.acc_dist_index, df['High'], df['Low'], df['Close'], df['Volume'], default=[None]*n
    )
    
    # Force Index
    if n >= 13:
        indicators['fi'] = safe_calculate(
            ta.volume.force_index, df['Close'], df['Volume'], default=[None]*n
        )
    
    # EMV
    if n >= 14:
        indicators['emv'] = safe_calculate(
            ta.volume.ease_of_movement, df['High'], df['Low'], df['Volume'], default=[None]*n
        )
    
    # VPT
    indicators['vpt'] = safe_calculate(
        ta.volume.volume_price_trend, df['Close'], df['Volume'], default=[None]*n
    )
    
    # NVI
    indicators['nvi'] = safe_calculate(
        ta.volume.negative_volume_index, df['Close'], df['Volume'], default=[None]*n
    )
    
    # VWAP
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        indicators['vwap'] = safe_calculate(lambda: vwap, default=[None]*n)
    except:
        pass
    
    # Volume SMA
    if n >= 20:
        try:
            vol_sma = df['Volume'].rolling(window=20).mean()
            indicators['volume_sma_20'] = safe_calculate(lambda: vol_sma, default=[None]*n)
        except:
            pass
    
    # ========== CUSTOM ==========
    
    # Price distance from SMA
    for period in [20, 50, 200]:
        if n >= period and f'sma_{period}' in indicators:
            try:
                sma = pd.Series(indicators[f'sma_{period}'])
                distance = ((df['Close'] - sma) / sma * 100).fillna(0)
                indicators[f'price_distance_sma_{period}'] = safe_calculate(lambda: distance, default=[None]*n)
            except:
                pass
    
    # Returns
    try:
        ret_1d = df['Close'].pct_change(1)
        indicators['return_1d'] = safe_calculate(lambda: ret_1d, default=[None]*n)
    except:
        pass
    
    if n >= 5:
        try:
            ret_5d = df['Close'].pct_change(5)
            indicators['return_5d'] = safe_calculate(lambda: ret_5d, default=[None]*n)
        except:
            pass
    
    if n >= 20:
        try:
            ret_20d = df['Close'].pct_change(20)
            indicators['return_20d'] = safe_calculate(lambda: ret_20d, default=[None]*n)
            
            vol_20d = df['Close'].pct_change().rolling(window=20).std()
            indicators['volatility_20d'] = safe_calculate(lambda: vol_20d, default=[None]*n)
        except:
            pass
    
    # High-Low Range
    try:
        hl_range = ((df['High'] - df['Low']) / df['Close'] * 100).fillna(0)
        indicators['hl_range'] = safe_calculate(lambda: hl_range, default=[None]*n)
        
        close_pos = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).fillna(0.5)
        indicators['close_position'] = safe_calculate(lambda: close_pos, default=[None]*n)
    except:
        pass
    
    return indicators


def get_indicator_signals(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Analyse ULTRA-ROBUSTE de tous les indicateurs
    """
    def safe_get(arr, default=50):
        """Récupère la dernière valeur valide ou default"""
        if not arr or len(arr) == 0:
            return default
        
        # Cherche la dernière valeur non-None en partant de la fin
        for i in range(len(arr)-1, -1, -1):
            val = arr[i]
            if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return float(val)
        
        return default
    
    signals = {
        'trend': {},
        'momentum': {},
        'volatility': {},
        'volume': {},
        'overall': {}
    }
    
    try:
        close = float(df['Close'].iloc[-1])
    except:
        close = 100
    
    # ========== TREND ==========
    
    sma_20 = safe_get(indicators.get('sma_20', []), close)
    sma_50 = safe_get(indicators.get('sma_50', []), close)
    sma_200 = safe_get(indicators.get('sma_200', []), close)
    
    if close > sma_20 and close > sma_50:
        signals['trend']['sma'] = {'signal': 'BUY', 'strength': 0.7}
    elif close < sma_20 and close < sma_50:
        signals['trend']['sma'] = {'signal': 'SELL', 'strength': 0.7}
    else:
        signals['trend']['sma'] = {'signal': 'NEUTRAL', 'strength': 0.3}
    
    # MACD
    if 'macd' in indicators:
        macd = safe_get(indicators['macd'], 0)
        macd_signal = safe_get(indicators['macd_signal'], 0)
        
        if macd > macd_signal:
            signals['trend']['macd'] = {'signal': 'BUY', 'strength': 0.6}
        else:
            signals['trend']['macd'] = {'signal': 'SELL', 'strength': 0.6}
    
    # ADX
    if 'adx' in indicators:
        adx = safe_get(indicators['adx'], 20)
        adx_pos = safe_get(indicators.get('adx_pos', []), 20)
        adx_neg = safe_get(indicators.get('adx_neg', []), 20)
        
        if adx > 25:
            if adx_pos > adx_neg:
                signals['trend']['adx'] = {'signal': 'STRONG_BUY', 'strength': min(adx/50, 1.0)}
            else:
                signals['trend']['adx'] = {'signal': 'STRONG_SELL', 'strength': min(adx/50, 1.0)}
        else:
            signals['trend']['adx'] = {'signal': 'WEAK', 'strength': 0.3}
    
    # ========== MOMENTUM ==========
    
    if 'rsi' in indicators:
        rsi = safe_get(indicators['rsi'], 50)
        
        if rsi > 70:
            signals['momentum']['rsi'] = {'signal': 'OVERBOUGHT', 'strength': min((rsi - 70) / 30, 1.0)}
        elif rsi < 30:
            signals['momentum']['rsi'] = {'signal': 'OVERSOLD', 'strength': min((30 - rsi) / 30, 1.0)}
        else:
            signals['momentum']['rsi'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # Stochastic
    if 'stoch_k' in indicators:
        stoch_k = safe_get(indicators['stoch_k'], 50)
        stoch_d = safe_get(indicators['stoch_d'], 50)
        
        if stoch_k > 80:
            signals['momentum']['stochastic'] = {'signal': 'OVERBOUGHT', 'strength': 0.7}
        elif stoch_k < 20:
            signals['momentum']['stochastic'] = {'signal': 'OVERSOLD', 'strength': 0.7}
        else:
            signals['momentum']['stochastic'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # ========== VOLATILITY ==========
    
    if 'bb_upper' in indicators:
        bb_upper = safe_get(indicators['bb_upper'], close * 1.02)
        bb_lower = safe_get(indicators['bb_lower'], close * 0.98)
        
        if close > bb_upper:
            signals['volatility']['bollinger'] = {'signal': 'OVERBOUGHT', 'strength': 0.6}
        elif close < bb_lower:
            signals['volatility']['bollinger'] = {'signal': 'OVERSOLD', 'strength': 0.6}
        else:
            signals['volatility']['bollinger'] = {'signal': 'NORMAL', 'strength': 0.5}
    
    # ========== VOLUME ==========
    
    if 'mfi' in indicators:
        mfi = safe_get(indicators['mfi'], 50)
        
        if mfi > 80:
            signals['volume']['mfi'] = {'signal': 'OVERBOUGHT', 'strength': 0.6}
        elif mfi < 20:
            signals['volume']['mfi'] = {'signal': 'OVERSOLD', 'strength': 0.6}
        else:
            signals['volume']['mfi'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # ========== OVERALL ==========
    
    buy_score = 0
    sell_score = 0
    
    for category in ['trend', 'momentum', 'volatility', 'volume']:
        for data in signals[category].values():
            signal = data['signal']
            strength = data.get('strength', 0.5)
            
            if 'BUY' in signal or 'OVERSOLD' in signal:
                buy_score += strength
            elif 'SELL' in signal or 'OVERBOUGHT' in signal:
                sell_score += strength
    
    total = buy_score + sell_score
    if total == 0:
        total = 1  # Éviter division par zéro
    
    if buy_score > sell_score * 1.5:
        signals['overall']['recommendation'] = 'STRONG_BUY'
        signals['overall']['confidence'] = min((buy_score / total) * 100, 100)
    elif buy_score > sell_score:
        signals['overall']['recommendation'] = 'BUY'
        signals['overall']['confidence'] = min((buy_score / total) * 100, 100)
    elif sell_score > buy_score * 1.5:
        signals['overall']['recommendation'] = 'STRONG_SELL'
        signals['overall']['confidence'] = min((sell_score / total) * 100, 100)
    elif sell_score > buy_score:
        signals['overall']['recommendation'] = 'SELL'
        signals['overall']['confidence'] = min((sell_score / total) * 100, 100)
    else:
        signals['overall']['recommendation'] = 'HOLD'
        signals['overall']['confidence'] = 50
    
    signals['overall']['buy_score'] = round(buy_score, 2)
    signals['overall']['sell_score'] = round(sell_score, 2)
    
    return signals
