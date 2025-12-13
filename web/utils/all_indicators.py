#!/usr/bin/env python3
"""
📊 PLOUTOS - COMPLETE TECHNICAL INDICATORS LIBRARY

Tous les indicateurs techniques professionnels utilisés par les traders

Categories:
- Tendance (Trend)
- Momentum
- Volatilité (Volatility)
- Volume
- Support/Résistance

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
import ta


def calculate_complete_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule TOUS les indicateurs techniques professionnels
    
    Args:
        df: DataFrame avec colonnes OHLCV
    
    Returns:
        dict avec tous les indicateurs
    """
    indicators = {}
    
    # ========== TREND INDICATORS ==========
    
    # SMA (Simple Moving Averages)
    for period in [10, 20, 50, 100, 200]:
        indicators[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=period).tolist()
    
    # EMA (Exponential Moving Averages)
    for period in [9, 12, 20, 26, 50]:
        indicators[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=period).tolist()
    
    # WMA (Weighted Moving Average)
    indicators['wma_20'] = ta.trend.wma_indicator(df['Close'], window=20).tolist()
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    indicators['macd'] = macd.macd().tolist()
    indicators['macd_signal'] = macd.macd_signal().tolist()
    indicators['macd_hist'] = macd.macd_diff().tolist()
    
    # ADX (Average Directional Index) - Force de la tendance
    indicators['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14).tolist()
    indicators['adx_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14).tolist()
    indicators['adx_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14).tolist()
    
    # Parabolic SAR (Stop and Reverse)
    sar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
    indicators['sar'] = sar.psar().tolist()
    indicators['sar_up'] = sar.psar_up().tolist()
    indicators['sar_down'] = sar.psar_down().tolist()
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    indicators['ichimoku_a'] = ichimoku.ichimoku_a().tolist()
    indicators['ichimoku_b'] = ichimoku.ichimoku_b().tolist()
    indicators['ichimoku_base'] = ichimoku.ichimoku_base_line().tolist()
    indicators['ichimoku_conv'] = ichimoku.ichimoku_conversion_line().tolist()
    
    # Aroon (Trend strength)
    aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
    indicators['aroon_up'] = aroon.aroon_up().tolist()
    indicators['aroon_down'] = aroon.aroon_down().tolist()
    indicators['aroon_indicator'] = aroon.aroon_indicator().tolist()
    
    # CCI (Commodity Channel Index)
    indicators['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20).tolist()
    
    # DPO (Detrended Price Oscillator)
    indicators['dpo'] = ta.trend.dpo(df['Close'], window=20).tolist()
    
    # ========== MOMENTUM INDICATORS ==========
    
    # RSI (Relative Strength Index)
    indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).tolist()
    indicators['rsi_6'] = ta.momentum.rsi(df['Close'], window=6).tolist()  # Court terme
    indicators['rsi_21'] = ta.momentum.rsi(df['Close'], window=21).tolist()  # Long terme
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    indicators['stoch_k'] = stoch.stoch().tolist()
    indicators['stoch_d'] = stoch.stoch_signal().tolist()
    
    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'])
    indicators['stoch_rsi'] = stoch_rsi.stochrsi().tolist()
    indicators['stoch_rsi_k'] = stoch_rsi.stochrsi_k().tolist()
    indicators['stoch_rsi_d'] = stoch_rsi.stochrsi_d().tolist()
    
    # Williams %R
    indicators['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close']).tolist()
    
    # ROC (Rate of Change)
    indicators['roc'] = ta.momentum.roc(df['Close'], window=12).tolist()
    
    # TSI (True Strength Index)
    indicators['tsi'] = ta.momentum.tsi(df['Close']).tolist()
    
    # Ultimate Oscillator
    indicators['uo'] = ta.momentum.ultimate_oscillator(
        df['High'], df['Low'], df['Close']
    ).tolist()
    
    # Awesome Oscillator
    indicators['ao'] = ta.momentum.awesome_oscillator(
        df['High'], df['Low']
    ).tolist()
    
    # KAMA (Kaufman's Adaptive Moving Average)
    indicators['kama'] = ta.momentum.kama(df['Close']).tolist()
    
    # ========== VOLATILITY INDICATORS ==========
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    indicators['bb_upper'] = bb.bollinger_hband().tolist()
    indicators['bb_middle'] = bb.bollinger_mavg().tolist()
    indicators['bb_lower'] = bb.bollinger_lband().tolist()
    indicators['bb_width'] = bb.bollinger_wband().tolist()
    indicators['bb_percent'] = bb.bollinger_pband().tolist()
    
    # ATR (Average True Range)
    indicators['atr'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=14
    ).tolist()
    
    # Keltner Channels
    kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
    indicators['kc_upper'] = kc.keltner_channel_hband().tolist()
    indicators['kc_middle'] = kc.keltner_channel_mband().tolist()
    indicators['kc_lower'] = kc.keltner_channel_lband().tolist()
    
    # Donchian Channels
    dc = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
    indicators['dc_upper'] = dc.donchian_channel_hband().tolist()
    indicators['dc_middle'] = dc.donchian_channel_mband().tolist()
    indicators['dc_lower'] = dc.donchian_channel_lband().tolist()
    
    # Ulcer Index
    indicators['ui'] = ta.volatility.ulcer_index(df['Close']).tolist()
    
    # ========== VOLUME INDICATORS ==========
    
    # OBV (On-Balance Volume)
    indicators['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume']).tolist()
    
    # CMF (Chaikin Money Flow)
    indicators['cmf'] = ta.volume.chaikin_money_flow(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).tolist()
    
    # MFI (Money Flow Index)
    indicators['mfi'] = ta.volume.money_flow_index(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).tolist()
    
    # ADI (Accumulation/Distribution Index)
    indicators['adi'] = ta.volume.acc_dist_index(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).tolist()
    
    # Force Index
    indicators['fi'] = ta.volume.force_index(df['Close'], df['Volume']).tolist()
    
    # EMV (Ease of Movement)
    indicators['emv'] = ta.volume.ease_of_movement(
        df['High'], df['Low'], df['Volume']
    ).tolist()
    
    # VPT (Volume Price Trend)
    indicators['vpt'] = ta.volume.volume_price_trend(df['Close'], df['Volume']).tolist()
    
    # NVI (Negative Volume Index)
    indicators['nvi'] = ta.volume.negative_volume_index(df['Close'], df['Volume']).tolist()
    
    # Volume Weighted Average Price (VWAP approximation)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    indicators['vwap'] = vwap.tolist()
    
    # Volume SMA
    indicators['volume_sma_20'] = df['Volume'].rolling(window=20).mean().tolist()
    
    # ========== CUSTOM INDICATORS ==========
    
    # Price Distance from SMA
    for period in [20, 50, 200]:
        sma = ta.trend.sma_indicator(df['Close'], window=period)
        distance = ((df['Close'] - sma) / sma * 100)
        indicators[f'price_distance_sma_{period}'] = distance.tolist()
    
    # Returns
    indicators['return_1d'] = df['Close'].pct_change(1).tolist()
    indicators['return_5d'] = df['Close'].pct_change(5).tolist()
    indicators['return_20d'] = df['Close'].pct_change(20).tolist()
    
    # Volatility (rolling std)
    indicators['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std().tolist()
    
    # High-Low Range
    indicators['hl_range'] = ((df['High'] - df['Low']) / df['Close'] * 100).tolist()
    
    # Close position in daily range
    indicators['close_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).tolist()
    
    return indicators


def get_indicator_signals(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Analyse tous les indicateurs et génère des signaux
    
    Returns:
        dict avec signaux par catégorie
    """
    def safe_get(arr, default=50):
        """Get last value safely"""
        if not arr:
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
    
    # ========== TREND SIGNALS ==========
    
    close = df['Close'].iloc[-1]
    sma_20 = safe_get(indicators['sma_20'], close)
    sma_50 = safe_get(indicators['sma_50'], close)
    sma_200 = safe_get(indicators['sma_200'], close)
    
    # SMA Trend
    if close > sma_20 and close > sma_50 and close > sma_200:
        signals['trend']['sma'] = {'signal': 'STRONG_BUY', 'strength': 1.0}
    elif close > sma_20 and close > sma_50:
        signals['trend']['sma'] = {'signal': 'BUY', 'strength': 0.7}
    elif close < sma_20 and close < sma_50 and close < sma_200:
        signals['trend']['sma'] = {'signal': 'STRONG_SELL', 'strength': 1.0}
    elif close < sma_20 and close < sma_50:
        signals['trend']['sma'] = {'signal': 'SELL', 'strength': 0.7}
    else:
        signals['trend']['sma'] = {'signal': 'NEUTRAL', 'strength': 0.0}
    
    # MACD
    macd = safe_get(indicators['macd'], 0)
    macd_signal = safe_get(indicators['macd_signal'], 0)
    
    if macd > macd_signal and macd > 0:
        signals['trend']['macd'] = {'signal': 'STRONG_BUY', 'strength': 1.0}
    elif macd > macd_signal:
        signals['trend']['macd'] = {'signal': 'BUY', 'strength': 0.6}
    elif macd < macd_signal and macd < 0:
        signals['trend']['macd'] = {'signal': 'STRONG_SELL', 'strength': 1.0}
    elif macd < macd_signal:
        signals['trend']['macd'] = {'signal': 'SELL', 'strength': 0.6}
    else:
        signals['trend']['macd'] = {'signal': 'NEUTRAL', 'strength': 0.0}
    
    # ADX (Trend Strength)
    adx = safe_get(indicators['adx'], 20)
    adx_pos = safe_get(indicators['adx_pos'], 20)
    adx_neg = safe_get(indicators['adx_neg'], 20)
    
    if adx > 25:
        if adx_pos > adx_neg:
            signals['trend']['adx'] = {'signal': 'STRONG_BUY', 'strength': min(adx/50, 1.0)}
        else:
            signals['trend']['adx'] = {'signal': 'STRONG_SELL', 'strength': min(adx/50, 1.0)}
    else:
        signals['trend']['adx'] = {'signal': 'WEAK_TREND', 'strength': 0.3}
    
    # ========== MOMENTUM SIGNALS ==========
    
    # RSI
    rsi = safe_get(indicators['rsi'], 50)
    
    if rsi > 70:
        signals['momentum']['rsi'] = {'signal': 'OVERBOUGHT', 'strength': (rsi - 70) / 30}
    elif rsi < 30:
        signals['momentum']['rsi'] = {'signal': 'OVERSOLD', 'strength': (30 - rsi) / 30}
    elif rsi > 50:
        signals['momentum']['rsi'] = {'signal': 'BULLISH', 'strength': (rsi - 50) / 20}
    elif rsi < 50:
        signals['momentum']['rsi'] = {'signal': 'BEARISH', 'strength': (50 - rsi) / 20}
    else:
        signals['momentum']['rsi'] = {'signal': 'NEUTRAL', 'strength': 0.0}
    
    # Stochastic
    stoch_k = safe_get(indicators['stoch_k'], 50)
    stoch_d = safe_get(indicators['stoch_d'], 50)
    
    if stoch_k > 80 and stoch_d > 80:
        signals['momentum']['stochastic'] = {'signal': 'OVERBOUGHT', 'strength': 0.8}
    elif stoch_k < 20 and stoch_d < 20:
        signals['momentum']['stochastic'] = {'signal': 'OVERSOLD', 'strength': 0.8}
    elif stoch_k > stoch_d:
        signals['momentum']['stochastic'] = {'signal': 'BUY', 'strength': 0.5}
    else:
        signals['momentum']['stochastic'] = {'signal': 'SELL', 'strength': 0.5}
    
    # Williams %R
    williams = safe_get(indicators['williams_r'], -50)
    
    if williams > -20:
        signals['momentum']['williams'] = {'signal': 'OVERBOUGHT', 'strength': 0.7}
    elif williams < -80:
        signals['momentum']['williams'] = {'signal': 'OVERSOLD', 'strength': 0.7}
    else:
        signals['momentum']['williams'] = {'signal': 'NEUTRAL', 'strength': 0.3}
    
    # ========== VOLATILITY SIGNALS ==========
    
    bb_upper = safe_get(indicators['bb_upper'], close * 1.02)
    bb_lower = safe_get(indicators['bb_lower'], close * 0.98)
    bb_width = safe_get(indicators['bb_width'], 0)
    
    if close > bb_upper:
        signals['volatility']['bollinger'] = {'signal': 'OVERBOUGHT', 'strength': 0.7}
    elif close < bb_lower:
        signals['volatility']['bollinger'] = {'signal': 'OVERSOLD', 'strength': 0.7}
    else:
        signals['volatility']['bollinger'] = {'signal': 'NORMAL', 'strength': 0.5}
    
    if bb_width > 0.05:
        signals['volatility']['width'] = {'signal': 'HIGH', 'strength': min(bb_width * 10, 1.0)}
    elif bb_width < 0.02:
        signals['volatility']['width'] = {'signal': 'LOW', 'strength': 0.3}
    else:
        signals['volatility']['width'] = {'signal': 'MEDIUM', 'strength': 0.5}
    
    # ========== VOLUME SIGNALS ==========
    
    mfi = safe_get(indicators['mfi'], 50)
    
    if mfi > 80:
        signals['volume']['mfi'] = {'signal': 'OVERBOUGHT', 'strength': (mfi - 80) / 20}
    elif mfi < 20:
        signals['volume']['mfi'] = {'signal': 'OVERSOLD', 'strength': (20 - mfi) / 20}
    else:
        signals['volume']['mfi'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    obv_ma = safe_get(indicators['obv'][-20:] if len(indicators['obv']) >= 20 else indicators['obv'], 0)
    obv = safe_get(indicators['obv'], 0)
    
    if obv > obv_ma:
        signals['volume']['obv'] = {'signal': 'ACCUMULATION', 'strength': 0.6}
    else:
        signals['volume']['obv'] = {'signal': 'DISTRIBUTION', 'strength': 0.6}
    
    # ========== OVERALL SIGNAL ==========
    
    buy_score = 0
    sell_score = 0
    total_weight = 0
    
    for category in ['trend', 'momentum', 'volatility', 'volume']:
        for indicator, data in signals[category].items():
            signal = data['signal']
            strength = data['strength']
            
            if 'BUY' in signal or 'OVERSOLD' in signal or 'ACCUMULATION' in signal:
                buy_score += strength
            elif 'SELL' in signal or 'OVERBOUGHT' in signal or 'DISTRIBUTION' in signal:
                sell_score += strength
            
            total_weight += 1
    
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
    signals['overall']['total_indicators'] = total_weight
    
    return signals
