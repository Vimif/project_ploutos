#!/usr/bin/env python3
"""
🌐 PLOUTOS WEB DASHBOARD - V8 ORACLE + TRADER PRO + 5 TOOLS + CHART PRO + PRO ANALYSIS + WATCHLISTS + LIVE TRADING + SIGNALS + SCALPER
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
from dataclasses import is_dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... (tout le code existant reste identique) ...

# Juste après les imports et avant les routes, garder tout le code existant
# Je vais juste montrer les modifications aux routes:

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart_page():
    return render_template('advanced_chart.html')

@app.route('/tools')
def tools_page():
    return render_template('tools.html')

@app.route('/live')
def live_page():
    """🔥 Page Live Trading Dashboard"""
    return render_template('live.html')

@app.route('/signals')
def signals_page():
    """🚦 Page Trading Signals Dashboard - Interface graphique avec signaux BUY/SELL"""
    return render_template('trading_signals.html')

@app.route('/scalper')
def scalper_page():
    """⚡ Page Scalper Pro - Dashboard trading court-terme temps réel"""
    return render_template('scalper.html')

# ... (garder tout le reste du code API inchangé) ...

if __name__ == '__main__':
    import os
    host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    print("\n" + "="*70)
    print("🌐 PLOUTOS - V8 ORACLE + LIVE TRADING + WATCHLISTS + SIGNALS + CHARTS + SCALPER")
    print("="*70)
    print(f"\n🚀 http://{host}:{port}")
    print(f"🔥 Live Trading: http://{host}:{port}/live")
    print(f"🚦 Trading Signals: http://{host}:{port}/signals")
    print(f"📊 Advanced Charts: http://{host}:{port}/chart")
    print(f"⚡ Scalper Pro: http://{host}:{port}/scalper")  # NOUVEAU
    if LIVE_WATCHLISTS_AVAILABLE:
        print(f"📊 9 Watchlists prédéfinies disponibles")
    if TECHNICAL_ANALYZER_AVAILABLE:
        print(f"✅ Technical Analysis: /api/chart/* + /api/mtf/* actifs")
    if PRO_ANALYSIS_BP_AVAILABLE:
        print(f"✅ Pro Analysis: /api/pro-analysis/* actif")
    print("\n" + "="*70 + "\n")
    app.run(host=host, port=port, debug=False)
