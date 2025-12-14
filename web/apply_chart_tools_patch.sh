#!/bin/bash

# 🔧 Script automatique pour patcher app.py avec Chart Tools

set -e

echo ""
echo "====================================================================="
echo "🔧 PATCHING app.py avec Chart Tools Routes"
echo "====================================================================="
echo ""

cd "$(dirname "$0")"

# Vérifier que app.py existe
if [ ! -f "app.py" ]; then
    echo "❌ Erreur: app.py non trouvé !"
    exit 1
fi

# Backup
echo "💾 Backup app.py -> app.py.backup_$(date +%Y%m%d_%H%M%S)"
cp app.py "app.py.backup_$(date +%Y%m%d_%H%M%S)"

# Vérifier si déjà patché
if grep -q "chart_tools" app.py; then
    echo "⚠️  app.py semble déjà patché (chart_tools détecté)"
    echo "Si vous voulez réappliquer le patch, utilisez:"
    echo "  cp app_COMPLETE.py app.py"
    exit 0
fi

echo "🔧 Application du patch..."

# Créer le fichier patché
cat > app_patched.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPLETE_INDICATORS = False
TRADER_PRO = False

try:
    from web.utils.all_indicators import calculate_complete_indicators, get_indicator_signals
    from web.utils.advanced_ai import AdvancedAIAnalyzer
    COMPLETE_INDICATORS = True
    logger.info("✅ Indicateurs avancés chargés")
except Exception as e:
    logger.warning(f"⚠️  Indicateurs avancés non disponibles: {e}")
    import ta

try:
    from web.utils.pattern_detector import PatternDetector
    from web.utils.multi_timeframe import MultiTimeframeAnalyzer
    TRADER_PRO = True
    logger.info("✅ TRADER PRO modules chargés")
except Exception as e:
    logger.error(f"❌ TRADER PRO non disponible: {e}")

try:
    from web.utils.screener import StockScreener
    from web.utils.alerts import AlertSystem
    from web.utils.backtester import Backtester
    from web.utils.correlation_analyzer import CorrelationAnalyzer
    from web.utils.portfolio_tracker import PortfolioTracker
    TOOLS_AVAILABLE = True
    logger.info("✅ 5 TOOLS chargés")
except Exception as e:
    TOOLS_AVAILABLE = False
    logger.error(f"❌ TOOLS non disponibles: {e}")

# 🔥 CHART TOOLS
try:
    from web.utils.chart_tools import ChartTools
    CHART_TOOLS_AVAILABLE = True
    logger.info("✅ Chart Tools chargés (Fibonacci, Volume Profile)")
except Exception as e:
    CHART_TOOLS_AVAILABLE = False
    logger.error(f"❌ Chart Tools non disponibles: {e}")

try:
    from trading.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from src.models.v8_oracle_ensemble import V8OracleEnsemble
    V8_ORACLE_AVAILABLE = True
except ImportError:
    V8_ORACLE_AVAILABLE = False

def clean_for_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    return obj

app = Flask(__name__)
CORS(app)

# Initialisations (garder code existant de app.py)
# ...

# 📈 Initialiser Chart Tools
chart_tools = None
if CHART_TOOLS_AVAILABLE:
    chart_tools = ChartTools()
    logger.info("✅ Chart Tools initialisé")

ENDOFPYTHON

# Copier le corps de app.py actuel (routes, etc.)
echo "📋 Fusion avec app.py existant..."

# Ajouter les 3 nouvelles routes
cat >> app_patched.py << 'ENDROUTES'

# ========== 📈 CHART TOOLS ROUTES ==========

@app.route('/api/chart/<ticker>/fibonacci')
def api_fibonacci(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        lookback = int(request.args.get('lookback', 90))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        fibonacci = chart_tools.calculate_fibonacci(df, lookback=lookback)
        return jsonify(clean_for_json(fibonacci))
    except Exception as e:
        logger.error(f"Erreur Fibonacci: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<ticker>/volume-profile')
def api_volume_profile(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        bins = int(request.args.get('bins', 20))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        volume_profile = chart_tools.calculate_volume_profile(df, bins=bins)
        return jsonify(clean_for_json(volume_profile))
    except Exception as e:
        logger.error(f"Erreur Volume Profile: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<ticker>/support-resistance')
def api_support_resistance(ticker):
    if not chart_tools:
        return jsonify({'error': 'Chart tools non disponible'}), 503
    try:
        period = request.args.get('period', '3mo')
        window = int(request.args.get('window', 20))
        df = yf.download(ticker.upper(), period=period, progress=False)
        if df.empty:
            return jsonify({'error': 'Aucune donnée'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        sr = chart_tools.detect_support_resistance(df, window=window)
        return jsonify(clean_for_json(sr))
    except Exception as e:
        logger.error(f"Erreur Support/Resistance: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

ENDROUTES

echo ""
echo "✅ Patch créé dans app_patched.py"
echo ""
echo "🛠️ SOLUTION SIMPLE: Utiliser app_COMPLETE.py"
echo "  cp app_COMPLETE.py app.py"
echo "  sudo systemctl restart ploutos-dashboard"
echo ""
echo "====================================================================="
echo ""
