# auto_trader.py
# ---------------------------------------------------------
# PILOTE AUTOMATIQUE V10 (X-RAY MONITORING)
# ---------------------------------------------------------
import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Support IA
try:
    from sb3_contrib import RecurrentPPO
    USE_LSTM = True
except ImportError:
    from stable_baselines3 import PPO
    USE_LSTM = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Imports internes
import ai_trainer 
from trading_env import StockTradingEnv
from alpaca_training import AlpacaTrainer
from trading_bot import TradingBrain
import news_analyst 
import llm_analyst 
import notifier
from config import * 

# --- MONITORING SYSTEM ---
MONITOR_FILE = "monitor_log.json"
BOT_STATE = {
    "last_update": "",
    "status": "Initialisation",
    "market_trend": "N/A",
    "vix": 0,
    "buying_power": 0,
    "portfolio_value": 0,
    "scanned_tickers": [],
    "last_decisions": [],
    "active_positions": [],
    "live_thoughts": [] # <--- NOUVEAU : Flux de pensée
}

def update_monitor(state):
    try:
        with open(MONITOR_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    except: pass

# --- FONCTIONS MÉTIER ---
MODELS = {} 

def prepare_data_for_trading(ticker):
    brain = TradingBrain()
    df = brain.telecharger_donnees(ticker)
    if df is None or df.empty: return pd.DataFrame()
    
    df = df.copy()
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_Ratio'] = close / close.rolling(50).mean()
    e1 = close.ewm(span=12).mean(); e2 = close.ewm(span=26).mean()
    df['MACD'] = e1 - e2
    return df.dropna().reset_index(drop=True)

def get_model_name_for_ticker(ticker):
    SECTORS = {
        "TECH": ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "QQQ", "AMZN", "GOOGL", "META", "NFLX"],
        "DEFENSIVE": ["KO", "PG", "JNJ", "MCD", "WMT", "XLV", "PEP", "COST"],
        "ENERGY": ["XOM", "CVX", "COP", "SLB", "XLE", "CAT", "BP", "SHEL"],
        "CRYPTO": ["COIN", "MSTR", "MARA", "RIOT", "BITO", "HOOD"]
    }
    target = "ppo_trading_brain"
    for sec, tickers in SECTORS.items():
        if ticker in tickers:
            candidate = f"brain_{sec.lower()}"
            if os.path.exists(candidate + ".zip"): target = candidate; break
    
    if not os.path.exists(target + ".zip"):
        if os.path.exists("ppo_trading_brain.zip"): target = "ppo_trading_brain"
        else: return None
    return target

# Charge le modèle ET retourne l'env pour simulation
def get_model_and_env(model_name, df):
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    vecnorm_path = model_name + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        try:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False; env.norm_reward = False
        except: pass
    
    if model_name not in MODELS:
        try:
            if USE_LSTM: MODELS[model_name] = RecurrentPPO.load(model_name)
            else: MODELS[model_name] = PPO.load(model_name)
        except: MODELS[model_name] = PPO.load(model_name)
            
    return MODELS[model_name], env

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def charger_liste_cible(alpaca):
    cibles = set()
    if os.path.exists(FILE_SCAN_RESULTS):
        try:
            with open(FILE_SCAN_RESULTS, 'r') as f:
                scan = json.load(f)
                for s in scan: cibles.add(s)
        except: pass
    if alpaca.connected:
        for p in alpaca.list_positions(): cibles.add(p.symbol)
    if not cibles:
        for s in DEFAULT_WATCHLIST: cibles.add(s)
    return list(cibles)

def run_bot():
    log("🦅 AUTO-TRADER V10 (X-RAY)...")
    alpaca = AlpacaTrainer()
    if not alpaca.connected: return
    brain = TradingBrain()

    while True:
        BOT_STATE["last_update"] = datetime.now().strftime('%H:%M:%S')
        BOT_STATE["status"] = "Analyse Macro..."
        update_monitor(BOT_STATE)
        
        # MACRO
        trend_spy = brain.analyser_tendance_marche()
        BOT_STATE["market_trend"] = trend_spy
        try:
            vix_df = brain.telecharger_donnees("^VIX")
            if vix_df is not None:
                vix = vix_df['Close'].iloc[-1]
                BOT_STATE["vix"] = round(vix, 2)
                if vix > 32:
                    BOT_STATE["status"] = "PAUSE (VIX DANGER)"
                    update_monitor(BOT_STATE)
                    time.sleep(CHECK_INTERVAL); continue
        except: pass

        # MICRO
        BOT_STATE["status"] = "Scan Actifs..."
        try:
            acct = alpaca.get_account()
            BOT_STATE["buying_power"] = float(acct.buying_power)
            BOT_STATE["portfolio_value"] = float(acct.portfolio_value)
            positions = []
            for p in alpaca.list_positions():
                positions.append({"symbol": p.symbol, "qty": p.qty, "pl": f"{float(p.unrealized_plpc)*100:.2f}%"})
            BOT_STATE["active_positions"] = positions
            update_monitor(BOT_STATE)
        except: pass

        targets = charger_liste_cible(alpaca)
        
        for ticker in targets:
            BOT_STATE["scanned_tickers"].insert(0, ticker)
            BOT_STATE["scanned_tickers"] = BOT_STATE["scanned_tickers"][:5]
            update_monitor(BOT_STATE)
            
            try:
                df = prepare_data_for_trading(ticker)
                if df.empty or len(df) < 50: continue

                model_name = get_model_name_for_ticker(ticker)
                if not model_name: continue

                # INFERENCE X-RAY
                model, env = get_model_and_env(model_name, df)
                obs = env.reset()
                _states = None
                # Replay
                for _ in range(len(df) - 1):
                    action, _states = model.predict(obs, state=_states, deterministic=True)
                    obs, rewards, dones, info = env.step(action)
                
                # Prédiction Finale
                final_action_raw, _ = model.predict(obs, state=_states, deterministic=True)
                final_action = final_action_raw[0] # 0=Hold, 1=Buy, 2=Sell
                current_price = df['Close'].iloc[-1]

                # --- CERVEAU EXPLICITE (X-RAY) ---
                thought = {
                    "time": datetime.now().strftime('%H:%M:%S'),
                    "ticker": ticker,
                    "model": model_name,
                    "action_raw": int(final_action),
                    "llama_verdict": "-",
                    "status": ""
                }

                # Positions actuelles
                qty_held = 0
                for p in positions:
                    if p['symbol'] == ticker: qty_held = int(p['qty'])

                # LOGIQUE DÉCISIONNELLE
                if final_action == 1: # BUY
                    if qty_held > 0:
                        thought["status"] = "Déjà possédé"
                    elif trend_spy == "BEAR":
                        thought["status"] = "Refus Macro (Bear)"
                    elif news_analyst.get_sentiment(ticker) < -0.15:
                        thought["status"] = "Refus VADER (News)"
                    else:
                        # LLAMA
                        log(f"📞 Llama check {ticker}...")
                        verdict_llm, reason_llm = llm_analyst.ask_the_oracle(ticker)
                        thought["llama_verdict"] = verdict_llm
                        
                        if verdict_llm == "BEARISH":
                            thought["status"] = "VETO LLAMA ⛔"
                        else:
                            thought["status"] = "VALIDÉ ✅"
                            
                            # EXÉCUTION
                            qty = int(MAX_POSITION_SIZE / current_price)
                            cost = qty * current_price
                            if qty > 0 and BOT_STATE["buying_power"] > cost:
                                log(f"🚀 ACHAT : {ticker}")
                                alpaca.buy(ticker, qty)
                                BOT_STATE["buying_power"] -= cost
                                
                                # Log Transaction
                                dec = {"time": datetime.now().strftime('%H:%M'), "type": "ACHAT", "ticker": ticker, "reason": "IA + Llama OK"}
                                BOT_STATE["last_decisions"].insert(0, dec)

                elif final_action == 2: # SELL
                    if qty_held > 0:
                        thought["status"] = "VENTE PROFIT 💰"
                        log(f"📉 VENTE : {ticker}")
                        alpaca.client.close_position(ticker)
                        dec = {"time": datetime.now().strftime('%H:%M'), "type": "VENTE", "ticker": ticker, "reason": "Signal Vente"}
                        BOT_STATE["last_decisions"].insert(0, dec)
                    else:
                        thought["status"] = "Signal Vente (Pas de position)"
                
                else:
                    thought["status"] = "Hold (Rien à faire)"

                # Mise à jour Flux de pensée
                BOT_STATE["live_thoughts"].insert(0, thought)
                BOT_STATE["live_thoughts"] = BOT_STATE["live_thoughts"][:15] # Garder les 15 derniers
                update_monitor(BOT_STATE)

            except Exception as e: pass

        BOT_STATE["status"] = "En Veille"
        update_monitor(BOT_STATE)
        log(f"💤 Pause {int(CHECK_INTERVAL/60)} min...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()
