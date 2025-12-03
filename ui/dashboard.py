# ui/dashboard.py
"""Dashboard Streamlit pour Ploutos"""

# FIX: Ajouter le projet au path AVANT les autres imports
import sys
from pathlib import Path

# Remonter d'un niveau depuis ui/ pour arriver à la racine du projet
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Maintenant les imports fonctionnent
import streamlit as st
import pandas as pd
import time
from datetime import datetime

from trading.brain_trader import BrainTrader
from config.settings import TRADING_CONFIG
from config.tickers import SECTORS

# Configuration
st.set_page_config(
    page_title="Ploutos Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles personnalisés
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.title("🧠 PLOUTOS - Dashboard Multi-Cerveaux")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    capital = st.number_input(
        "💰 Capital ($)",
        min_value=1000,
        max_value=10_000_000,
        value=TRADING_CONFIG['initial_capital'],
        step=10000
    )
    
    st.info("📊 Mode: Paper Trading")
    
    st.markdown("---")
    
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    if st.button("🚀 ANALYSER", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True
    
    st.markdown("---")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")

# Initialisation
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False

# Analyse
if st.session_state['run_analysis'] or auto_refresh:
    
    with st.spinner("🧠 Consultation des cerveaux..."):
        trader = BrainTrader(capital=capital, paper_trading=True)
        predictions = trader.predict_all()
    
    # Calcul des métriques
    total_buy = 0
    total_sell = 0
    total_hold = 0
    capital_deployed = 0
    
    for sector_preds in predictions.values():
        for pred in sector_preds:
            if pred['action'] == 'BUY':
                total_buy += 1
                capital_deployed += pred['capital']
            elif pred['action'] == 'SELL':
                total_sell += 1
            else:
                total_hold += 1
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🟢 SIGNAUX D'ACHAT",
            total_buy,
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "🔴 SIGNAUX DE VENTE",
            total_sell,
            delta=None
        )
    
    with col3:
        st.metric(
            "⚪ EN ATTENTE",
            total_hold,
            delta=None
        )
    
    with col4:
        st.metric(
            "💰 CAPITAL DÉPLOYÉ",
            f"${capital_deployed:,.0f}",
            delta=f"{(capital_deployed/capital)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Analyse par secteur
    st.subheader("📊 Analyse Détaillée par Secteur")
    
    tabs = st.tabs([
        "₿ CRYPTO",
        "🛡️ DEFENSIVE",
        "⚡ ENERGY",
        "💻 TECH"
    ])
    
    sector_keys = ['crypto', 'defensive', 'energy', 'tech']
    
    for tab, sector_key in zip(tabs, sector_keys):
        with tab:
            sector_preds = predictions.get(sector_key, [])
            
            if sector_preds:
                # Infos secteur
                sector_config = SECTORS[sector_key]
                
                col_a, col_b = st.columns([1, 3])
                
                with col_a:
                    st.metric(
                        "Allocation",
                        f"{sector_config['allocation']*100:.0f}%"
                    )
                    st.metric(
                        "Tickers",
                        len(sector_config['tickers'])
                    )
                
                with col_b:
                    # Tableau
                    df = pd.DataFrame(sector_preds)
                    
                    # Sélection colonnes
                    df_display = df[['ticker', 'action', 'capital']].copy()
                    df_display.columns = ['Ticker', 'Action', 'Capital ($)']
                    
                    # Style
                    def style_action(val):
                        colors = {
                            'BUY': 'background-color: #90EE90; color: black;',
                            'SELL': 'background-color: #FFB6C1; color: black;',
                            'HOLD': 'background-color: #E0E0E0; color: black;'
                        }
                        return colors.get(val, '')
                    
                    styled_df = df_display.style.applymap(
                        style_action,
                        subset=['Action']
                    ).format({
                        'Capital ($)': '${:,.2f}'
                    })
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=200
                    )
                
                # Capital du secteur
                sector_capital = sum(p['capital'] for p in sector_preds if p['action'] == 'BUY')
                if sector_capital > 0:
                    st.info(f"💵 Capital actif: ${sector_capital:,.2f}")
            
            else:
                st.warning("Aucune donnée pour ce secteur")
    
    st.markdown("---")
    
    # Recommandations
    st.subheader("💡 Recommandations")
    
    if total_buy > 0:
        st.success(f"✅ {total_buy} opportunité(s) d'achat détectée(s)")
    
    if total_sell > 0:
        st.warning(f"⚠️ {total_sell} signal(aux) de vente")
    
    if total_hold == len([p for sp in predictions.values() for p in sp]):
        st.info("📊 Tous les signaux sont en HOLD - Pas d'action recommandée")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

else:
    # Écran d'accueil
    st.info("👈 Cliquez sur 'ANALYSER' dans la barre latérale pour commencer")
    
    st.markdown("""
    ## 🎯 Fonctionnalités
    
    - 🧠 **4 Cerveaux IA** spécialisés par secteur
    - 📊 **Analyse en temps réel** des marchés
    - 💰 **Gestion automatique** des allocations
    - 📈 **Signaux BUY/SELL/HOLD** précis
    
    ## 🚀 Utilisation
    
    1. Configurez votre capital dans la barre latérale
    2. Cliquez sur "ANALYSER"
    3. Consultez les recommandations par secteur
    4. Activez l'auto-refresh pour un suivi continu
    """)
