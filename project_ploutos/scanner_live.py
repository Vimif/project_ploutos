# scanner_live.py (V5 - Avec Portefeuille Miroir)
# ---------------------------------------------------------
# Gère l'Achat ET la Vente de vos positions eToro
# ---------------------------------------------------------

from trading_bot import TradingBrain
from gestion_portefeuille import PortefeuilleManager # <--- NOUVEAU
import pandas as pd
import time
import datetime
import os
import requests

# --- CONFIGURATION ---
WATCHLIST = ["IBM", "AAPL", "TSLA", "MSFT", "NVDA", "AMD", "INTC", "AMZN", "GOOGL", "META", "F", "PFE", "KO"]
FICHIER_LOG = "journal_trading.csv"
CAPITAL_TOTAL = 10000 
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1445133574801195183/HvQKgnHcASHAoPq8X6C0_bUVhHTku-E9cMR0dqmfkN04M16fpfnmq5BldGtkEVt83BRz" 

def envoyer_alerte_discord(titre, ticker, prix, message, couleur):
    if "discord.com" not in DISCORD_WEBHOOK_URL: return
    
    data = {
        "username": "eToro Assistant 🤖",
        "embeds": [{
            "title": f"{titre} : {ticker}",
            "description": f"**Prix Actuel :** {prix:.2f} $\n\n{message}",
            "color": couleur
        }]
    }
    try: requests.post(DISCORD_WEBHOOK_URL, json=data)
    except: pass

def lancer_radar():
    brain = TradingBrain()
    portfolio = PortefeuilleManager() # <--- On charge le portefeuille
    
    print("\n🌍 ANALYSE MACRO-ÉCONOMIQUE...")
    regime_marche = brain.analyser_tendance_marche()
    emoji_meteo = "☀️" if regime_marche == "BULL" else "⛈️"
    print(f"VERDICT S&P 500 : {regime_marche} {emoji_meteo}")
    
    print(f"\n🚀 Démarrage du SCANNER (Positions ouvertes: {len(portfolio.positions)})...")
    print("="*130)
    print(f"{'ACTION':<6} | {'PRIX':<8} | {'ETAT':<12} | {'VERDICT':<25} | {'CONSEIL'}")
    print("="*130)
    
    for ticker in WATCHLIST:
        df = brain.telecharger_donnees(ticker)
        
        if df is not None and len(df) > 50:
            # Analyse technique de base pour tout le monde
            df = brain.ajouter_indicateurs(df)
            last = df.iloc[-1]
            prix_actuel = last['Close']
            
            # --- CAS 1 : ON A DÉJÀ L'ACTION (Mode GARDIEN) ---
            if portfolio.est_en_position(ticker):
                infos = portfolio.get_infos_position(ticker)
                prix_achat = infos['prix_achat']
                perf_latente = (prix_actuel - prix_achat) / prix_achat * 100
                
                etat_str = f"POS {perf_latente:+.1f}%"
                verdict = "GARDER 🔒"
                conseil = "..."
                couleur_discord = 0
                
                # Vérification Vente
                # 1. Stop Loss (Protection)
                if prix_actuel < infos['stop_loss']:
                    verdict = "VENTE (STOP LOSS) 🛑"
                    conseil = "Vendre sur eToro !"
                    portfolio.retirer_position(ticker, prix_actuel) # On sort du JSON
                    couleur_discord = 15158332 # Rouge
                    envoyer_alerte_discord("🛑 STOP LOSS TOUCHÉ", ticker, prix_actuel, f"Perte : {perf_latente:.2f}%", couleur_discord)
                
                # 2. Take Profit (Gain)
                elif prix_actuel > infos['take_profit']:
                    verdict = "VENTE (TAKE PROFIT) 💰"
                    conseil = "Encaisser les gains !"
                    portfolio.retirer_position(ticker, prix_actuel)
                    couleur_discord = 16776960 # Jaune Or
                    envoyer_alerte_discord("💰 TAKE PROFIT TOUCHÉ", ticker, prix_actuel, f"Gain : {perf_latente:.2f}%", couleur_discord)
                
                # 3. Signal Technique de Sortie (RSI Surchauffe)
                elif last['RSI'] > 75:
                    verdict = "VENTE (RSI SURACHAT) ⚠️"
                    conseil = "Vendre (Indicateur Technique)"
                    portfolio.retirer_position(ticker, prix_actuel)
                    couleur_discord = 15105570 # Orange
                    envoyer_alerte_discord("⚠️ SIGNAL DE VENTE", ticker, prix_actuel, f"RSI trop haut ({last['RSI']:.1f})", couleur_discord)
            
            # --- CAS 2 : ON N'A PAS L'ACTION (Mode CHASSEUR) ---
            else:
                # On essaie de charger la mémoire, sinon on recalcule
                try:
                    best_buy, best_sell = brain.charger_memoire(ticker)
                except:
                    best_buy, best_sell = brain.trouver_parametres_optimaux(df)

                if regime_marche == "BEAR": best_buy -= 5
                
                etat_str = "LIQUIDE"
                verdict = "NEUTRE"
                conseil = ""
                
                # Logique d'Achat
                score = 0
                if last['RSI'] < best_buy:
                    if last['MACD'] > last['Signal_Line']:
                        verdict = "ACHAT FORT 🚀"
                        score = 2
                    else:
                        verdict = "ACHAT (RSI seul)"
                        score = 1
                
                # Filtre Fondamental
                if score >= 1:
                    sante_txt, sante_score = brain.verifier_sante_fondamentale(ticker)
                    if sante_score < 2: 
                        verdict = "IGNORÉ (Fragile)"
                        score = 0
                
                # Action
                if score == 2:
                    nb_actions, cout = brain.calculer_position_ideale(CAPITAL_TOTAL, prix_actuel)
                    conseil = f"Acheter {nb_actions} ({cout:.0f}$)"
                    
                    # Simulation d'achat automatique dans le JSON (Miroir)
                    portfolio.ajouter_position(ticker, prix_actuel, nb_actions)
                    
                    msg = f"Signal Technique Validé.\nSanté : OK\n**Ordre : Acheter {nb_actions} actions**"
                    envoyer_alerte_discord("🚀 NOUVELLE POSITION", ticker, prix_actuel, msg, 65280) # Vert

            print(f"{ticker:<6} | {prix_actuel:<8.2f} | {etat_str:<12} | {verdict:<25} | {conseil}")
            
        else:
            print(f"{ticker:<6} ... Données insuffisantes.")
        
        time.sleep(0.2)

    print("="*130)

if __name__ == "__main__":
    lancer_radar()
