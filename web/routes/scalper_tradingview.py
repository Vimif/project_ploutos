#!/usr/bin/env python3
"""
⚡ SCALPER TRADINGVIEW ROUTE
Route Flask pour le Scalper Pro TradingView
"""

from flask import Blueprint, render_template

scalper_tv_bp = Blueprint('scalper_tv', __name__)

@scalper_tv_bp.route('/scalper-tv')
def scalper_tradingview():
    """Page Scalper Pro TradingView"""
    return render_template('scalper_tradingview.html')
