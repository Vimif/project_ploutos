# night_scan.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import yfinance as yf
import pandas as pd
import json
import time
from alpaca_training import AlpacaTrainer

def get_tickers():
    print("🌍 Récupération liste marché...")
    alp = AlpacaTrainer()
    if alp.connected: return alp.get_all_tickers()
    return ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META"]

def scan_market():
    all_tickers = get_tickers()
    BATCH_SIZE = 300 
    pepites = []
    print(f"🚀 Scan de {len(all_tickers)} actions...")
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i + BATCH_SIZE]
        print(f"📦 Paquet {i}/{len(all_tickers)}...")
        try:
            data = yf.download(batch, period="6mo", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            for t in batch:
                try:
                    if len(batch) == 1: df = data
                    else: df = data.get(t)
                    if df is None or df.empty: continue
                    df = df.dropna()
                    if len(df) < 50: continue
                    
                    last_close = float(df['Close'].iloc[-1])
                    if last_close < 5.0: continue 

                    # Indicateurs Manuels (Pas de pandas-ta)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss.replace(0, 0.001)
                    rsi = 100 - (100 / (1 + rs))
                    last_rsi = float(rsi.iloc[-1])
                    
                    sma_50 = float(df['Close'].rolling(50).mean().iloc[-1])
                    last_vol = float(df['Volume'].iloc[-1])
                    vol_ma = float(df['Volume'].rolling(20).mean().iloc[-1])

                    if last_rsi < 35 and last_close > sma_50:
                        print(f"   💎 DIP : {t}")
                        pepites.append(t)

                    if last_vol > (vol_ma * 3.0) and 50 < last_rsi < 80:
                        print(f"   🚀 BOOM : {t}")
                        pepites.append(t)
                except: pass
        except: time.sleep(1)

    print(f"🏁 Terminé ! {len(pepites)} pépites.")
    with open("shortlist_ia.json", "w") as f: json.dump(list(set(pepites)), f)

if __name__ == "__main__":
    # 1. On scanne le marché pour trouver les cibles
    scan_market()
    
    # 2. On met à jour le cerveau de l'IA (Nouveau !)
    print("\n🌙 Phase 2 : Amélioration Continue de l'IA...")
    try:
        import ai_trainer
        ai_trainer.retrain_model()
    except Exception as e:
        print(f"⚠️ Échec mise à jour IA : {e}")
