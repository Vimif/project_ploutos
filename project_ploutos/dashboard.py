# dashboard.py (V25 - ULTIMATE FULL OPTION + LOGS TAB)
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import time
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from stable_baselines3 import PPO 
import sys
import subprocess

# IMPORTS INTERNES
from gestion_portefeuille import PortefeuilleManager
from trading_bot import TradingBrain
from alpaca_training import AlpacaTrainer
from gestion_watchlist import WatchlistManager
import ai_trainer 
from trading_env import StockTradingEnv
import logger_bot

# --- CONFIGURATION & CSS ---
st.set_page_config(page_title="Ploutos Ultimate", layout="wide", page_icon="🦅")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: #161B22; border-radius: 12px; padding: 15px; border: 1px solid #30363D;
    }
    div[data-testid="stMetricValue"] { color: #E6EDF3 !important; }
    .stButton>button { border-radius: 8px; background-color: #238636; color: white; border: none; font-weight: 600; width: 100%; }
    .stTextInput>div>div>input { background-color: #0D1117; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦅 Ploutos : Station de Commandement")

# --- INIT ---
portfolio = PortefeuilleManager()
brain = TradingBrain()
alpaca = AlpacaTrainer()
wl_manager = WatchlistManager()

# --- SIDEBAR INTELLIGENCE ---
st.sidebar.header("🧠 Cerveau Neuronal")
model_path = "ppo_trading_brain.zip"

if os.path.exists(model_path):
    st.sidebar.success("Modèle PPO Actif ✅")
    if 'active_graph_ticker' in st.session_state:
        tk = st.session_state['active_graph_ticker']
        if st.sidebar.button(f"🔮 Avis IA sur {tk}"):
            with st.spinner("Analyse neurale..."):
                try:
                    df = ai_trainer.prepare_data(tk)
                    if not df.empty:
                        temp_env = StockTradingEnv(df)
                        temp_env.current_step = len(df) - 1
                        obs = temp_env._next_observation()
                        model = PPO.load(model_path)
                        action, _ = model.predict(obs)
                        verdicts = ["✋ ATTENDRE", "🚀 ACHETER", "📉 VENDRE"]
                        st.sidebar.markdown(f"### Verdict : **{verdicts[action]}**")
                except Exception as e: st.sidebar.error(f"Erreur IA: {e}")
else:
    st.sidebar.warning("Pas de cerveau entraîné.")

if st.sidebar.button("🏋️‍♂️ Entraîner (Deep Learning)"):
    with st.sidebar.status("Entraînement PPO en cours...", expanded=True):
        ai_trainer.train_model()
        st.success("Terminé !")
    st.rerun()

# --- AJOUT BOUTON AUTO-TRADER ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 Auto-Trader")
if st.sidebar.button("▶️ DÉMARRER AUTO-TRADER"):
    subprocess.Popen([sys.executable, "auto_trader.py"])
    st.toast("🚀 Moteur lancé en fond !")

if st.sidebar.button("🔴 STOP MOTEUR"):
    os.system("pkill -f auto_trader.py")
    st.sidebar.error("Moteur arrêté.")

# --- SIDEBAR CONTRÔLES MANUELS (ACHAT & VENTE) ---
st.sidebar.markdown("---")
st.sidebar.header("🎮 Gestion Manuelle")

# Bloc ACHAT
with st.sidebar.expander("➕ Ajouter Position (Achat)", expanded=False):
    with st.form("f_add"):
        tk_a = st.text_input("Ticker").upper()
        p_a = st.number_input("Prix Achat", 0.01, 10000.0)
        q_a = st.number_input("Qté", 1, 10000)
        sect = st.text_input("Secteur", "Tech")
        if st.form_submit_button("Enregistrer Achat"):
            if tk_a: 
                portfolio.ajouter_position(tk_a, p_a, int(q_a), sect)
                st.success(f"Achat {tk_a} noté ✅")
                time.sleep(0.5); st.rerun()

# Bloc VENTE (Le retour !)
with st.sidebar.expander("❌ Retirer Position (Vente)", expanded=False):
    with st.form("f_del"):
        actions_possedees = list(portfolio.positions.keys())
        if actions_possedees:
            tk_v = st.selectbox("Action à vendre", actions_possedees)
            p_v = st.number_input("Prix Vente", 0.01, 10000.0)
            if st.form_submit_button("Confirmer Vente"):
                gain, perf = portfolio.retirer_position(tk_v, p_v)
                st.success(f"Vente notée ! P/L: {gain:+.2f}$ ({perf:+.2f}%) 💸")
                time.sleep(1); st.rerun()
        else:
            st.info("Portefeuille vide, rien à vendre.")
            st.form_submit_button("Inactif", disabled=True)

# --- SIDEBAR LOGS CLASSIQUES ---
st.sidebar.markdown("---")
st.sidebar.header("📟 Logs Système")
try:
    logs = logger_bot.lire_logs(10)
    if logs: st.sidebar.code(''.join(logs), language='text')
    else: st.sidebar.caption("Aucun log récent.")
except: pass
if st.sidebar.button("🔄 Rafraîchir Logs"): st.rerun()

# --- FONCTIONS GRAPHIQUES ---
def afficher_graphique_pro(ticker):
    df = brain.telecharger_donnees(ticker) 
    if df is None or len(df) < 30: 
        st.error("Données insuffisantes (Yahoo bloque ? Réessayez plus tard).")
        return

    df = brain.ajouter_indicateurs(df)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3], subplot_titles=(ticker, "RSI"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='Prix',
                                 increasing_line_color='#00CC96', decreasing_line_color='#EF553B'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#FFA15A', width=1), name='SMA 20'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB63FA', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#EF553B", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00CC96", row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=550, margin=dict(t=40,b=0,l=0,r=0),
                      template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    desc, earn = brain.recuperer_profil_entreprise(ticker)
    c1, c2 = st.columns([3, 1])
    c1.info(desc)
    c2.metric("📅 Résultats", earn)

# --- ANALYSEUR ---
st.subheader("🔍 Analyseur de Marché")
c_s, c_b = st.columns([4, 1])
with c_s: qt = st.text_input("Rechercher :", placeholder="Ex: TSLA, NVDA...", label_visibility="collapsed").upper()
with c_b: 
    if st.button("Voir Graphique"): st.session_state['active_graph_ticker'] = qt

if 'active_graph_ticker' in st.session_state:
    afficher_graphique_pro(st.session_state['active_graph_ticker'])

st.divider()

# --- ONGLETS PRINCIPAUX ---
tabs = st.tabs(["💼 Portefeuille", "📡 Scanner IA", "🌍 Météo & Rente", "📊 Historique", "🎓 Alpaca", "🧠 Cerveau IA (Logs)"])

# T1 : PORTFOLIO
with tabs[0]:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Positions Suivies")
        if portfolio.positions:
            rows = []
            tot = 0
            for t, i in portfolio.positions.items():
                df = brain.telecharger_donnees(t)
                cur = df['Close'].iloc[-1] if df is not None else i['prix_achat']
                val = cur * i['quantite']
                gain = val - (i['prix_achat'] * i['quantite'])
                pct = (gain / (i['prix_achat'] * i['quantite'])) * 100
                tot += val
                rows.append({"Action": t, "Prix Achat": i['prix_achat'], "Prix Actuel": cur, "Qté": i['quantite'], "P/L ($)": gain, "P/L (%)": pct})
            
            st.metric("Valeur Totale", f"{tot:.2f} $")
            
            def color_pl(val):
                color = '#00CC96' if val > 0 else '#EF553B'
                return f'color: {color}; font-weight: bold;'
                
            st.dataframe(pd.DataFrame(rows).style.map(color_pl, subset=['P/L ($)', 'P/L (%)']).format({'P/L ($)': "{:+.2f}", 'P/L (%)': "{:+.2f} %", "Prix Achat": "{:.2f}", "Prix Actuel": "{:.2f}"}), use_container_width=True)
        else: st.info("Portefeuille vide.")
        
    with c2:
        st.subheader("Diversification")
        if portfolio.positions:
            ds = {}
            for t, i in portfolio.positions.items():
                s = i.get('secteur', 'Autre')
                ds[s] = ds.get(s, 0) + (i['prix_achat'] * i['quantite'])
            fig = px.pie(names=list(ds.keys()), values=list(ds.values()), hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=250, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# T2 : SCANNER
with tabs[1]:
    c_scan, c_res = st.columns([1, 3])
    with c_scan:
        st.markdown("### Sources")
        if st.button("💎 Charger Scan de Nuit"):
            try:
                with open("shortlist_ia.json", "r") as f: wl = json.load(f)
                st.session_state['scan_wl'] = wl
                st.session_state['run_scan'] = True
            except: st.error("Fichier 'shortlist_ia.json' manquant.")
        
        if st.button("🚀 Watchlist Perso"):
            st.session_state['scan_wl'] = wl_manager.get_list()
            st.session_state['run_scan'] = True
            
    with c_res:
        if st.session_state.get('run_scan') and 'scan_wl' in st.session_state:
            res = []
            wl = st.session_state['scan_wl']
            pb = st.progress(0)
            for i, t in enumerate(wl):
                df = brain.telecharger_donnees(t)
                if df is not None:
                    rsi = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
                    last_rsi = rsi.iloc[-1]
                    trend = brain.analyser_tendance_weekly(t)
                    sig = "ACHAT" if last_rsi < 35 and trend == "HAUSSIER" else ("VENTE" if last_rsi > 70 else "NEUTRE")
                    res.append({"Action": t, "Prix": df['Close'].iloc[-1], "RSI": last_rsi, "Tendance": trend, "Signal": sig})
                pb.progress((i+1)/len(wl))
            pb.empty()
            
            df_res = pd.DataFrame(res)
            if not df_res.empty:
                st.dataframe(df_res.style.applymap(lambda v: "color: #00CC96; font-weight: bold" if v=="ACHAT" else ("color: #EF553B" if v=="VENTE" else ""), subset=['Signal']), use_container_width=True)

# T3 : MÉTÉO & RENTE
with tabs[2]:
    c_meteo, c_rente = st.columns(2)
    with c_meteo:
        st.subheader("🌍 Météo du Marché")
        if st.button("Scanner S&P 500"):
            trend = brain.analyser_tendance_marche()
            color = "#00CC96" if trend == "BULL" else "#EF553B"
            st.markdown(f"<h2 style='text-align: center; color: {color}; border: 2px solid {color}; padding: 10px; border-radius: 10px;'>MARCHÉ {trend}</h2>", unsafe_allow_html=True)
            
            # Heatmap Mini
            S = {"Tech": "XLK", "Energie": "XLE", "Finance": "XLF", "Santé": "XLV"}
            cols = st.columns(4)
            for i, (n, tk) in enumerate(S.items()):
                d = brain.telecharger_donnees(tk, periode="5d")
                if d is not None:
                    var = ((d['Close'].iloc[-1] - d['Close'].iloc[-2])/d['Close'].iloc[-2])*100
                    c = "green" if var > 0 else "red"
                    cols[i].markdown(f"**{n}**<br><span style='color:{c}'>{var:+.2f}%</span>", unsafe_allow_html=True)

    with c_rente:
        st.subheader("💰 Calculateur de Rente")
        if portfolio.positions:
            if st.button("Calculer Dividendes"):
                total_div = 0
                res_div = []
                for t, i in portfolio.positions.items():
                    yld, rate, date = brain.analyser_dividendes(t)
                    rente = rate * i['quantite'] if rate else 0
                    if rente == 0 and yld > 0:
                         df = brain.telecharger_donnees(t)
                         if df is not None:
                             rente = (df['Close'].iloc[-1] * (yld/100)) * i['quantite']
                    
                    total_div += rente
                    res_div.append({"Action": t, "Yield": f"{yld}%", "Rente/An": rente})
                
                st.metric("Rente Annuelle Estimée", f"{total_div:.2f} $")
                st.dataframe(pd.DataFrame(res_div), use_container_width=True)
        else: st.info("Il faut des actions pour toucher des dividendes !")

# T4 : HISTORIQUE
with tabs[3]:
    st.header("Historique des Ventes")
    if os.path.exists("historique_ventes.csv"):
        try:
            dfh = pd.read_csv("historique_ventes.csv")
            st.dataframe(dfh, use_container_width=True)
        except: st.info("Erreur lecture historique.")
    else: st.info("Aucune vente enregistrée pour le moment.")

# T5 : ALPACA
with tabs[4]:
    st.header("🎓 Alpaca Paper Trading")
    if alpaca.connected:
        ac = alpaca.get_account()
        c1, c2 = st.columns(2)
        c1.metric("Equity", f"{float(ac.equity):,.2f} $")
        c2.metric("Buying Power", f"{float(ac.buying_power):,.2f} $")
        
        st.divider()
        
        c_act, c_pos = st.columns([1, 2])
        with c_act:
            st.subheader("Passer Ordre")
            tk = st.text_input("Action", "AAPL", key="alp_tk").upper()
            qty = st.number_input("Qté", 1, 100, key="alp_qty")
            if st.button("Acheter (Paper)"):
                if alpaca.buy(tk, qty): st.success("Ordre envoyé !")
                else: st.error("Erreur ordre")
        
        with c_pos:
            st.subheader("Positions Ouvertes")
            pos = alpaca.list_positions()
            if pos:
                d = [{"Sym": p.symbol, "Qté": p.qty, "P/L ($)": float(p.unrealized_pl), "P/L (%)": float(p.unrealized_plpc)*100} for p in pos]
                st.dataframe(pd.DataFrame(d).style.format({"P/L (%)": "{:.2f}%"}), use_container_width=True)
                if st.button("🚨 TOUT VENDRE (Urgence)"):
                    alpaca.close_all()
                    st.warning("Ordre de liquidation envoyé.")
                    time.sleep(2); st.rerun()
            else: st.info("Aucune position active sur Alpaca.")
    else:
        st.error("Alpaca non connecté. Vérifiez vos clés API dans alpaca_training.py")

# T6 : LOGS CERVEAU IA (NEW !)
with tabs[5]:
    st.header("🧠 Cerveau IA en Direct (X-RAY)")
    if st.checkbox("🔴 Auto-Refresh (5s)", value=True):
        time.sleep(5)
        st.rerun()

    try:
        with open("monitor_log.json", 'r') as f: 
            state = json.load(f)
            thoughts = state.get("live_thoughts", [])
            
            # KPIs en haut
            c1, c2 = st.columns(2)
            status_icon = "🟢" if "Scan" in state.get("status","") else "💤"
            c1.metric("Statut Moteur", f"{status_icon} {state.get('status')}", state.get("last_update"))
            c2.metric("Tendance IA", state.get("market_trend", "N/A"))
            
            st.divider()
            
            # Tableau X-RAY
            if thoughts:
                t_data = []
                for t in thoughts:
                    act = "HOLD"
                    if t['action_raw'] == 1: act = "🚀 BUY"
                    if t['action_raw'] == 2: act = "📉 SELL"
                    
                    t_data.append({
                        "Heure": t['time'],
                        "Ticker": t['ticker'],
                        "Modèle": t['model'].replace("brain_", "").upper(),
                        "Intention IA": act,
                        "Llama Verdict": t['llama_verdict'],
                        "Décision Finale": t['status']
                    })
                st.dataframe(pd.DataFrame(t_data), use_container_width=True)
            else:
                st.info("Le cerveau est vide (lancez l'auto-trader).")
                
    except:
        st.warning("Fichier de logs introuvable. Lancez l'auto-trader avec le bouton à gauche.")
