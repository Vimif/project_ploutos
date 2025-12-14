#!/usr/bin/env python3
"""
🔧 PATCH pour ajouter les routes Chart Tools dans app.py

Ce fichier contient les routes à ajouter dans app.py
"""

# ========== 1️⃣ IMPORTS À AJOUTER (après les autres imports de web.utils) ==========
"""
Ajouter ces lignes vers la ligne 50 (après TOOLS_AVAILABLE):

# 📈 CHART TOOLS
try:
    from web.utils.chart_tools import ChartTools
    CHART_TOOLS_AVAILABLE = True
    logger.info("✅ Chart Tools chargés (Fibonacci, Volume Profile)")
except Exception as e:
    CHART_TOOLS_AVAILABLE = False
    logger.error(f"❌ Chart Tools non disponibles: {e}")
"""

# ========== 2️⃣ INITIALISATION À AJOUTER (après portfolio = None) ==========
"""
Ajouter après la section portfolio (ligne ~150):

# 📈 Initialiser Chart Tools
chart_tools = None
if CHART_TOOLS_AVAILABLE:
    chart_tools = ChartTools()
    logger.info("✅ Chart Tools initialisé")
"""

# ========== 3️⃣ ROUTES À AJOUTER (après @app.route('/tools')) ==========
"""
Ajouter ces 3 routes après la route /tools:
"""

@app.route('/api/chart/<ticker>/fibonacci')
def api_fibonacci(ticker):
    """🔥 Fibonacci Retracement"""
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
    """🔥 Volume Profile"""
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
    """🔥 Support & Resistance"""
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


# ========== 4️⃣ MODIFIER /api/health (ajouter chart_tools) ==========
"""
Dans la fonction api_health(), ajouter dans le dict modules:

'chart_tools': CHART_TOOLS_AVAILABLE,
'chart_tools_instance': chart_tools is not None,
"""

# ========== 5️⃣ MODIFIER le print de démarrage ==========
"""
Ajouter après "if TOOLS_AVAILABLE:":

if CHART_TOOLS_AVAILABLE:
    print("📈 CHART PRO activé")
    print("  ✅ Fibonacci Retracement")
    print("  ✅ Volume Profile")
    print("  ✅ Support/Resistance")
"""

print("""
🛠️ INSTRUCTIONS D'APPLICATION DU PATCH:

1. Ouvrir web/app.py
2. Ajouter les imports (section 1)
3. Ajouter l'initialisation (section 2)
4. Ajouter les 3 routes (section 3)
5. Modifier /api/health (section 4)
6. Modifier le print de démarrage (section 5)

Ou simplement copier web/app_COMPLETE.py vers web/app.py !
""")
