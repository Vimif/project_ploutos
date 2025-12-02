# backtest_pro.py
# ---------------------------------------------------------
# Ce script simule un investissement de 10 000$ sur le passé.
# Il permet de vérifier si le bot est rentable sur une action précise.
# ---------------------------------------------------------

from trading_bot import TradingBrain
import matplotlib.pyplot as plt # (Optionnel) pour voir des courbes si vous l'installez

ACTION_A_TESTER = "NVDA" # Changez le nom ici
CAPITAL_DEPART = 10000

def lancer_backtest():
    brain = TradingBrain()
    
    print(f"🔬 Démarrage du BACKTEST PRO sur {ACTION_A_TESTER}...")
    
    # 1. On récupère 5 ans d'historique
    df = brain.telecharger_donnees(ACTION_A_TESTER, periode="5y")
    df = brain.ajouter_indicateurs(df)
    
    # 2. L'IA trouve les réglages optimaux
    best_buy, best_sell = brain.trouver_parametres_optimaux(df)
    print(f"⚙️ Paramètres optimaux trouvés : Achat < {best_buy} | Vente > {best_sell}")
    
    # 3. Simulation Jour par Jour
    capital = CAPITAL_DEPART
    position = 0 # Nombre d'actions
    nb_trades = 0
    
    historique_valeur = []
    
    for i, row in df.iterrows():
        prix = row['Close']
        rsi = row['RSI']
        
        # VENTE
        if position > 0 and rsi > best_sell:
            capital = position * prix
            position = 0
            nb_trades += 1
        
        # ACHAT
        elif position == 0 and rsi < best_buy:
            # On ajoute le filtre MACD pour être plus réaliste
            if row['MACD'] > row['Signal_Line']:
                position = capital / prix
                capital = 0
                nb_trades += 1
        
        # Calcul valeur totale
        valeur_totale = capital + (position * prix)
        historique_valeur.append(valeur_totale)
        
    # 4. Résultats
    valeur_finale = historique_valeur[-1]
    perf = ((valeur_finale - CAPITAL_DEPART) / CAPITAL_DEPART) * 100
    
    # Benchmark (Buy & Hold)
    prix_debut = df.iloc[0]['Close']
    prix_fin = df.iloc[-1]['Close']
    perf_marche = ((prix_fin - prix_debut) / prix_debut) * 100
    
    print("\n📊 --- RAPPORT FINAL (5 ANS) ---")
    print(f"Capital Final   : {valeur_finale:.2f} $")
    print(f"Performance Bot : {perf:+.2f}%")
    print(f"Perf. Marché    : {perf_marche:+.2f}%")
    print(f"Nombre de trades: {nb_trades}")
    
    if perf > perf_marche:
        print("🏆 SUCCÈS : Le Bot bat le marché !")
    else:
        print("📉 INFO : Le marché a fait mieux (mais le bot est moins risqué).")

if __name__ == "__main__":
    lancer_backtest()
