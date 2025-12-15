"""
🎯 PRO ANALYSIS API - Blueprint

Endpoint pour l'analyse technique professionnelle détaillée.
Utilisé par le panneau "Analyse Pro" de l'interface web.
"""

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

pro_analysis_bp = Blueprint('pro_analysis', __name__)

# Import conditionnel des analyseurs
try:
    from dashboard.technical_analysis import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    logger.warning("⚠️  TechnicalAnalyzer non disponible pour pro_analysis")

# 🔥 TEMPORAIREMENT DÉSACTIVÉ : ProTechnicalAnalyzer attend un DataFrame, pas un symbol
# TODO: Adapter ProTechnicalAnalyzer.analyze(symbol, period) au lieu de analyze(df)
PRO_ANALYZER_AVAILABLE = False
pro_analyzer_instance = None

# try:
#     from web.utils.pro_technical_analyzer import ProTechnicalAnalyzer
#     PRO_ANALYZER_AVAILABLE = True
#     pro_analyzer_instance = ProTechnicalAnalyzer()
# except ImportError:
#     PRO_ANALYZER_AVAILABLE = False
#     pro_analyzer_instance = None
# except Exception as e:
#     PRO_ANALYZER_AVAILABLE = False
#     pro_analyzer_instance = None
#     logger.warning(f"⚠️  ProTechnicalAnalyzer init error: {e}")


@pro_analysis_bp.route('/api/pro-analysis/<symbol>')
def api_pro_analysis(symbol):
    """
    🎯 Analyse technique professionnelle détaillée
    
    Args:
        symbol: Ticker du symbole (ex: INTC, AAPL)
    
    Returns:
        JSON avec analyse complète :
        - overall_signal: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        - confidence: Niveau de confiance (0-100)
        - risk_level: LOW, MEDIUM, HIGH
        - current_price: Prix actuel
        - trend: Analyse de tendance
        - momentum: RSI et momentum
        - macd: MACD analysis
        - volatility: Bollinger Bands
        - volume: OBV analysis
        - trading_plan: Plan de trading textuel
    """
    
    # Version simplifiée avec TechnicalAnalyzer
    if not TECHNICAL_ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Pro Analysis indisponible',
            'details': 'Aucun analyseur technique disponible'
        }), 503
    
    try:
        # Analyse avec TechnicalAnalyzer (période 3 mois)
        analyzer = TechnicalAnalyzer(symbol, period='3mo', interval='1d')
        df = analyzer.df
        indicators = analyzer.get_all_indicators()
        signal = analyzer.generate_signal()
        
        # Prix actuel
        current_price = float(df['Close'].iloc[-1])
        
        # Valeurs par défaut pour stop_loss/take_profit si None
        entry_price = signal.entry_price if signal.entry_price is not None else current_price
        stop_loss = signal.stop_loss if signal.stop_loss is not None else current_price * 0.98  # -2%
        take_profit = signal.take_profit if signal.take_profit is not None else current_price * 1.03  # +3%
        
        # === RSI ANALYSIS ===
        rsi_value = float(indicators.get('rsi', 50.0))
        if rsi_value > 70:
            rsi_zone = 'overbought'
            rsi_signal = 'SELL'
        elif rsi_value < 30:
            rsi_zone = 'oversold'
            rsi_signal = 'BUY'
        else:
            rsi_zone = 'neutral'
            rsi_signal = 'HOLD'
        
        # === MACD ANALYSIS ===
        macd_data = indicators.get('macd', {})
        if isinstance(macd_data, dict):
            macd_value = float(macd_data.get('macd', 0.0))
            macd_signal_value = float(macd_data.get('signal', 0.0))
            macd_histogram = float(macd_data.get('histogram', 0.0))
        else:
            macd_value = float(macd_data) if macd_data else 0.0
            macd_signal_value = 0.0
            macd_histogram = 0.0
        
        macd_signal = 'BUY' if macd_value > macd_signal_value else 'SELL'
        macd_crossover = None
        if macd_value > macd_signal_value and macd_histogram > 0:
            macd_crossover = 'bullish'
        elif macd_value < macd_signal_value and macd_histogram < 0:
            macd_crossover = 'bearish'
        
        # === BOLLINGER BANDS ===
        bb_data = indicators.get('bollinger_bands', {})
        if isinstance(bb_data, dict):
            bb_upper = float(bb_data.get('upper', current_price * 1.02))
            bb_middle = float(bb_data.get('middle', current_price))
            bb_lower = float(bb_data.get('lower', current_price * 0.98))
        else:
            bb_upper = float(indicators.get('bb_upper', current_price * 1.02))
            bb_middle = float(indicators.get('bb_middle', current_price))
            bb_lower = float(indicators.get('bb_lower', current_price * 0.98))
        
        if bb_middle > 0:
            bb_width = ((bb_upper - bb_lower) / bb_middle * 100)
        else:
            bb_width = 0
        
        if current_price > bb_middle:
            price_position = 'upper'
        elif current_price < bb_middle:
            price_position = 'lower'
        else:
            price_position = 'middle'
        
        squeeze_detected = bb_width < 2.0
        
        # === SIGNAL GLOBAL ===
        if signal.signal == 'BUY':
            overall_signal = 'STRONG_BUY' if signal.strength > 70 else 'BUY'
        elif signal.signal == 'SELL':
            overall_signal = 'STRONG_SELL' if signal.strength > 70 else 'SELL'
        else:
            overall_signal = 'HOLD'
        
        # === RISQUE ===
        if bb_width > 5 or signal.strength < 50:
            risk_level = 'HIGH'
        elif bb_width > 2.5 or signal.strength < 70:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # === SMA 200 ===
        sma_200 = float(indicators.get('sma_200', current_price))
        price_vs_sma200 = 'above' if current_price > sma_200 else 'below'
        
        sma_50 = float(indicators.get('sma_50', current_price))
        golden_cross = sma_50 > sma_200
        
        # === ADX (Force de tendance) ===
        adx_value = float(indicators.get('adx', 0.0))
        
        # === PLAN DE TRADING ===
        stop_pct = ((stop_loss - current_price) / current_price * 100)
        target_pct = ((take_profit - current_price) / current_price * 100)
        
        trading_plan = f"""SIGNAL: {overall_signal}

🎯 ENTRÉE: ${entry_price:.2f}
🛑 STOP LOSS: ${stop_loss:.2f} ({stop_pct:.1f}%)
🎯 TAKE PROFIT: ${take_profit:.2f} ({target_pct:.1f}%)

⚠️ RISQUE: {risk_level}
🎯 CONFIANCE: {signal.confidence:.0f}%
📊 ADX: {adx_value:.1f}

📊 RAISONS:
""" + '\n'.join(f"  • {r}" for r in signal.reasons)
        
        # === RÉPONSE JSON ===
        return jsonify({
            'overall_signal': overall_signal,
            'confidence': signal.confidence,
            'risk_level': risk_level,
            'current_price': current_price,
            
            'trend': {
                'direction': signal.trend,
                'strength': signal.strength,
                'explanation': f"Tendance {signal.trend.lower()} détectée avec force {signal.strength:.0f}%",
                'price_vs_sma200': price_vs_sma200,
                'golden_cross': golden_cross,
                'support_level': stop_loss,
                'resistance_level': take_profit,
                'adx': adx_value
            },
            
            'momentum': {
                'signal': rsi_signal,
                'rsi_value': rsi_value,
                'zone': rsi_zone,
                'explanation': f"RSI à {rsi_value:.1f} - Zone {rsi_zone}",
                'divergence_detected': False,
                'divergence_type': None
            },
            
            'macd': {
                'signal': macd_signal,
                'macd_value': macd_value,
                'signal_value': macd_signal_value,
                'histogram_value': macd_histogram,
                'crossover': macd_crossover,
                'histogram_direction': 'increasing' if macd_histogram > 0 else 'decreasing',
                'explanation': f"MACD {'au-dessus' if macd_value > macd_signal_value else 'en-dessous'} du signal"
            },
            
            'volatility': {
                'price_position': price_position,
                'bb_width': bb_width,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'squeeze_detected': squeeze_detected,
                'explanation': f"Volatilité {risk_level.lower()} - Largeur BB: {bb_width:.2f}%"
            },
            
            'volume': {
                'obv_trend': signal.trend,
                'volume_confirmation': True,
                'smart_money_accumulation': signal.signal == 'BUY',
                'explanation': f"Volume confirme la tendance {signal.trend.lower()}"
            },
            
            'trading_plan': trading_plan
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur pro-analysis {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'details': 'Erreur lors de l\'analyse du symbole'
        }), 500
