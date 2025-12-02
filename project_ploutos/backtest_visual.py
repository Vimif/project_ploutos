# backtest_visual.py (VERSION RÉPARÉE & GOD MODE COMPATIBLE)
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Support optionnel LSTM
try:
    from sb3_contrib import RecurrentPPO
    USE_LSTM = True
except ImportError:
    USE_LSTM = False
    from stable_baselines3 import PPO

from trading_env import StockTradingEnv
from trading_bot import TradingBrain

# CONFIGURATION
st.set_page_config(page_title="Ploutos Backtester", layout="wide", page_icon="⏳")
st.markdown("""<style>.stApp { background-color: #0E1117; }</style>""", unsafe_allow_html=True)

st.title("⏳ Simulateur Temporel (God Mode)")

# --- FONCTION DE PRÉPARATION AUTONOME ---
def prepare_data_for_backtest(ticker):
    """Recrée la logique de préparation des données de l'usine"""
    brain = TradingBrain()
    df = brain.telecharger_donnees(ticker)
    
    if df is None or df.empty: return pd.DataFrame()
    
    df = df.copy()
    close = df['Close']
    
    # Indicateurs identiques à ai_trainer.py
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_Ratio'] = close / close.rolling(50).mean()
    e1 = close.ewm(span=12).mean(); e2 = close.ewm(span=26).mean()
    df['MACD'] = e1 - e2
    
    return df.dropna().reset_index(drop=True)

# --- INTERFACE ---
c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.text_input("Action à tester", "NVDA").upper()
with c2:
    # Choix du cerveau (Secteur)
    brain_choice = st.selectbox("Cerveau IA", ["brain_tech", "brain_energy", "brain_defensive", "brain_crypto", "ppo_trading_brain"])
with c3:
    capital = st.number_input("Capital ($)", 1000, 100000, 10000)

# --- MOTEUR ---
def run_simulation(ticker, model_name, initial_capital):
    # 1. Données
    df = prepare_data_for_backtest(ticker)
    if df.empty or len(df) < 50: return None, None, "Pas assez de données."

    # 2. Environnement
    # On doit recréer l'environnement exactement comme à l'entraînement
    env = DummyVecEnv([lambda: StockTradingEnv(df, initial_balance=initial_capital)])
    
    # Gestion Normalisation (VecNormalize)
    vecnorm_path = model_name + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        try:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False 
            env.norm_reward = False
            st.toast("Stats de Normalisation chargées ✅")
        except: st.warning("Échec chargement stats normalisation.")

    # 3. Modèle
    model_path = model_name # SB3 ajoute .zip tout seul
    if not os.path.exists(model_path + ".zip"):
        return None, None, f"Fichier {model_name}.zip introuvable."

    try:
        if USE_LSTM:
            model = RecurrentPPO.load(model_path)
        else:
            model = PPO.load(model_path)
    except:
        # Fallback croisé
        try: model = PPO.load(model_path)
        except: return None, None, "Erreur chargement modèle (Format incompatible ?)"

    # 4. Boucle de Simulation
    obs = env.reset()
    
    # Gestion LSTM States
    _states = None
    dones = np.array([False])
    
    history_ai = []
    history_price = []
    
    # On simule jour par jour
    # Attention : StockTradingEnv commence à window_size (5e jour)
    # On doit aligner les graphiques
    
    done = False
    while not done:
        # Prédiction
        action, _states = model.predict(obs, state=_states, deterministic=True)
        
        # Action
        obs, rewards, dones, infos = env.step(action)
        
        # Tracking
        # On accède à l'environnement interne (sous les wrappers)
        raw_env = env.envs[0]
        history_ai.append(raw_env.net_worth)
        
        current_step = raw_env.current_step
        if current_step < len(df):
            history_price.append(df['Close'].iloc[current_step])
            
        if dones[0]: done = True

    return history_ai, history_price, df.index[-len(history_ai):]

# --- AFFICHAGE ---
if st.button("🚀 Lancer Backtest", use_container_width=True):
    with st.spinner("Simulation du passé..."):
        port_curve, price_curve, dates = run_simulation(ticker, brain_choice, capital)
        
        if port_curve:
            # Calculs Performance
            final_ai = port_curve[-1]
            perf_ai = ((final_ai - capital) / capital) * 100
            
            # Buy & Hold
            start_price = price_curve[0]
            end_price = price_curve[-1]
            shares_bh = capital / start_price
            final_bh = shares_bh * end_price
            perf_bh = ((final_bh - capital) / capital) * 100
            
            # Métriques
            col_a, col_b = st.columns(2)
            color = "green" if perf_ai > 0 else "red"
            col_a.markdown(f"### 🤖 IA Ploutos : <span style='color:{color}'>{perf_ai:+.2f}%</span>", unsafe_allow_html=True)
            col_a.metric("Capital Final", f"{final_ai:,.2f} $", f"{final_ai-capital:+.2f}")
            
            col_b.markdown(f"### 😴 Marché (Hold) : {perf_bh:+.2f}%")
            col_b.metric("Capital Passif", f"{final_bh:,.2f} $")
            
            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=port_curve, mode='lines', name='🤖 IA Strategy', line=dict(color='#00CC96', width=3)))
            
            # Normalisation courbe prix pour superposition
            bh_normalized = [(p/start_price)*capital for p in price_curve]
            fig.add_trace(go.Scatter(y=bh_normalized, mode='lines', name='😴 Market Price', line=dict(color='gray', dash='dot')))
            
            fig.update_layout(title=f"Backtest : {ticker} vs {brain_choice}", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Erreur Simulation (Modèle manquant ou données vides)")
