#!/usr/bin/env python3
"""
📊 PLOUTOS - COMPLETE TECHNICAL INDICATORS LIBRARY

Tous les indicateurs techniques professionnels
Gestion intelligente des données insuffisantes
"""

import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')


def calculate_complete_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule TOUS les indicateurs techniques avec gestion des erreurs
    """
    indicators = {}
    n = len(df)
    
    # ========== TREND INDICATORS ==========
    
    # SMA (seulement si assez de données)
    for period in [10, 20, 50, 100, 200]:
        if n >= period:
            try:
                indicators[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=period).tolist()
            except:
                indicators[f'sma_{period}'] = [None] * n
    
    # EMA
    for period in [9, 12, 20, 26, 50]:
        if n >= period:
            try:
                indicators[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=period).tolist()
            except:
                indicators[f'ema_{period}'] = [None] * n
    
    # WMA
    if n >= 20:
        try:
            indicators['wma_20'] = ta.trend.wma_indicator(df['Close'], window=20).tolist()
        except:
            indicators['wma_20'] = [None] * n
    
    # MACD
    if n >= 26:
        try:
            macd = ta.trend.MACD(df['Close'])
            indicators['macd'] = macd.macd().tolist()
            indicators['macd_signal'] = macd.macd_signal().tolist()
            indicators['macd_hist'] = macd.macd_diff().tolist()
        except:
            pass
    
    # ADX
    if n >= 14:
        try:
            indicators['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14).tolist()
            indicators['adx_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14).tolist()
            indicators['adx_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14).tolist()
        except:
            pass
    
    # Parabolic SAR
    if n >= 5:
        try:
            sar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
            indicators['sar'] = sar.psar().tolist()
            indicators['sar_up'] = sar.psar_up().tolist()
            indicators['sar_down'] = sar.psar_down().tolist()
        except:
            pass
    
    # Ichimoku
    if n >= 52:
        try:
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            indicators['ichimoku_a'] = ichimoku.ichimoku_a().tolist()
            indicators['ichimoku_b'] = ichimoku.ichimoku_b().tolist()
            indicators['ichimoku_base'] = ichimoku.ichimoku_base_line().tolist()
            indicators['ichimoku_conv'] = ichimoku.ichimoku_conversion_line().tolist()
        except:
            pass
    
    # Aroon
    if n >= 25:
        try:
            aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
            indicators['aroon_up'] = aroon.aroon_up().tolist()
            indicators['aroon_down'] = aroon.aroon_down().tolist()
            indicators['aroon_indicator'] = aroon.aroon_indicator().tolist()
        except:
            pass
    
    # CCI
    if n >= 20:
        try:
            indicators['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20).tolist()
        except:
            pass
    
    # DPO
    if n >= 20:
        try:
            indicators['dpo'] = ta.trend.dpo(df['Close'], window=20).tolist()
        except:
            pass
    
    # ========== MOMENTUM INDICATORS ==========
    
    # RSI
    if n >= 14:
        try:
            indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).tolist()
        except:
            indicators['rsi'] = [50.0] * n
    
    if n >= 6:
        try:
            indicators['rsi_6'] = ta.momentum.rsi(df['Close'], window=6).tolist()
        except:
            pass
    
    if n >= 21:
        try:
            indicators['rsi_21'] = ta.momentum.rsi(df['Close'], window=21).tolist()
        except:
            pass
    
    # Stochastic
    if n >= 14:
        try:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            indicators['stoch_k'] = stoch.stoch().tolist()
            indicators['stoch_d'] = stoch.stoch_signal().tolist()
        except:
            pass
    
    # Stochastic RSI
    if n >= 14:
        try:
            stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'])
            indicators['stoch_rsi'] = stoch_rsi.stochrsi().tolist()
            indicators['stoch_rsi_k'] = stoch_rsi.stochrsi_k().tolist()
            indicators['stoch_rsi_d'] = stoch_rsi.stochrsi_d().tolist()
        except:
            pass
    
    # Williams %R
    if n >= 14:
        try:
            indicators['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close']).tolist()
        except:
            pass
    
    # ROC
    if n >= 12:
        try:
            indicators['roc'] = ta.momentum.roc(df['Close'], window=12).tolist()
        except:
            pass
    
    # TSI
    if n >= 25:
        try:
            indicators['tsi'] = ta.momentum.tsi(df['Close']).tolist()
        except:
            pass
    
    # Ultimate Oscillator
    if n >= 28:
        try:
            indicators['uo'] = ta.momentum.ultimate_oscillator(
                df['High'], df['Low'], df['Close']
            ).tolist()
        except:
            pass
    
    # Awesome Oscillator
    if n >= 34:
        try:
            indicators['ao'] = ta.momentum.awesome_oscillator(
                df['High'], df['Low']
            ).tolist()
        except:
            pass
    
    # KAMA
    if n >= 10:
        try:
            indicators['kama'] = ta.momentum.kama(df['Close']).tolist()
        except:
            pass
    
    # ========== VOLATILITY INDICATORS ==========
    
    # Bollinger Bands
    if n >= 20:
        try:
            bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            indicators['bb_upper'] = bb.bollinger_hband().tolist()
            indicators['bb_middle'] = bb.bollinger_mavg().tolist()
            indicators['bb_lower'] = bb.bollinger_lband().tolist()
            indicators['bb_width'] = bb.bollinger_wband().tolist()
            indicators['bb_percent'] = bb.bollinger_pband().tolist()
        except:
            pass
    
    # ATR
    if n >= 14:
        try:
            indicators['atr'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close'], window=14
            ).tolist()
        except:
            pass
    
    # Keltner Channels
    if n >= 20:
        try:
            kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            indicators['kc_upper'] = kc.keltner_channel_hband().tolist()
            indicators['kc_middle'] = kc.keltner_channel_mband().tolist()
            indicators['kc_lower'] = kc.keltner_channel_lband().tolist()
        except:
            pass
    
    # Donchian Channels
    if n >= 20:
        try:
            dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
            indicators['dc_upper'] = dc.donchian_channel_hband().tolist()
            indicators['dc_middle'] = dc.donchian_channel_mband().tolist()
            indicators['dc_lower'] = dc.donchian_channel_lband().tolist()
        except:
            pass
    
    # Ulcer Index
    if n >= 14:
        try:
            indicators['ui'] = ta.volatility.ulcer_index(df['Close']).tolist()
        except:
            pass
    
    # ========== VOLUME INDICATORS ==========
    
    # OBV
    try:
        indicators['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume']).tolist()
    except:
        pass
    
    # CMF
    if n >= 20:
        try:
            indicators['cmf'] = ta.volume.chaikin_money_flow(
                df['High'], df['Low'], df['Close'], df['Volume']
            ).tolist()
        except:
            pass
    
    # MFI
    if n >= 14:
        try:
            indicators['mfi'] = ta.volume.money_flow_index(
                df['High'], df['Low'], df['Close'], df['Volume']
            ).tolist()
        except:
            pass
    
    # ADI
    try:
        indicators['adi'] = ta.volume.acc_dist_index(
            df['High'], df['Low'], df['Close'], df['Volume']
        ).tolist()
    except:
        pass
    
    # Force Index
    if n >= 13:
        try:
            indicators['fi'] = ta.volume.force_index(df['Close'], df['Volume']).tolist()
        except:
            pass
    
    # EMV
    if n >= 14:
        try:
            indicators['emv'] = ta.volume.ease_of_movement(
                df['High'], df['Low'], df['Volume']
            ).tolist()
        except:
            pass
    
    # VPT
    try:
        indicators['vpt'] = ta.volume.volume_price_trend(df['Close'], df['Volume']).tolist()
    except:
        pass
    
    # NVI
    try:
        indicators['nvi'] = ta.volume.negative_volume_index(df['Close'], df['Volume']).tolist()
    except:
        pass
    
    # VWAP
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        indicators['vwap'] = vwap.tolist()
    except:
        pass
    
    # Volume SMA
    if n >= 20:
        try:
            indicators['volume_sma_20'] = df['Volume'].rolling(window=20).mean().tolist()
        except:
            pass
    
    # ========== CUSTOM ==========
    
    # Price distance from SMA
    for period in [20, 50, 200]:
        if n >= period and f'sma_{period}' in indicators:
            try:
                sma = pd.Series(indicators[f'sma_{period}'])
                distance = ((df['Close'] - sma) / sma * 100)
                indicators[f'price_distance_sma_{period}'] = distance.tolist()
            except:
                pass
    
    # Returns
    try:
        indicators['return_1d'] = df['Close'].pct_change(1).tolist()
    except:
        pass
    
    if n >= 5:
        try:
            indicators['return_5d'] = df['Close'].pct_change(5).tolist()
        except:
            pass
    
    if n >= 20:
        try:
            indicators['return_20d'] = df['Close'].pct_change(20).tolist()
            indicators['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std().tolist()
        except:
            pass
    
    # High-Low Range
    try:
        indicators['hl_range'] = ((df['High'] - df['Low']) / df['Close'] * 100).tolist()
        indicators['close_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).fillna(0.5).tolist()
    except:
        pass
    
    return indicators


def get_indicator_signals(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Analyse tous les indicateurs et génère des signaux
    """
    def safe_get(arr, default=50):
        if not arr or len(arr) == 0:
            return default
        val = arr[-1]
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    
    signals = {
        'trend': {},
        'momentum': {},
        'volatility': {},
        'volume': {},
        'overall': {}
    }
    
    close = df['Close'].iloc[-1]
    
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
            signals['momentum']['rsi'] = {'signal': 'OVERBOUGHT', 'strength': (rsi - 70) / 30}
        elif rsi < 30:
            signals['momentum']['rsi'] = {'signal': 'OVERSOLD', 'strength': (30 - rsi) / 30}
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
        bb_width = safe_get(indicators.get('bb_width', []), 0.04)
        
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
            strength = data['strength']
            
            if 'BUY' in signal or 'OVERSOLD' in signal:
                buy_score += strength
            elif 'SELL' in signal or 'OVERBOUGHT' in signal:
                sell_score += strength
    
    if buy_score > sell_score * 1.5:
        signals['overall']['recommendation'] = 'STRONG_BUY'
        signals['overall']['confidence'] = min((buy_score / (buy_score + sell_score)) * 100, 100)
    elif buy_score > sell_score:
        signals['overall']['recommendation'] = 'BUY'
        signals['overall']['confidence'] = min((buy_score / (buy_score + sell_score)) * 100, 100)
    elif sell_score > buy_score * 1.5:
        signals['overall']['recommendation'] = 'STRONG_SELL'
        signals['overall']['confidence'] = min((sell_score / (buy_score + sell_score)) * 100, 100)
    elif sell_score > buy_score:
        signals['overall']['recommendation'] = 'SELL'
        signals['overall']['confidence'] = min((sell_score / (buy_score + sell_score)) * 100, 100)
    else:
        signals['overall']['recommendation'] = 'HOLD'
        signals['overall']['confidence'] = 50
    
    signals['overall']['buy_score'] = round(buy_score, 2)
    signals['overall']['sell_score'] = round(sell_score, 2)
    
    return signals
