# fix_secteurs.py
# ---------------------------------------------------------
# SCRIPT DE RÉPARATION
# Met à jour les secteurs "Inconnu" dans portefeuille.json
# ---------------------------------------------------------

import json
from trading_bot import TradingBrain

FICHIER_PORTFOLIO = "portefeuille.json"
brain = TradingBrain()

def reparer_secteurs():
    try:
        with open(FICHIER_PORTFOLIO, 'r') as f:
            data = json.load(f)
    except:
        print("❌ Fichier introuvable.")
        return

    compteur = 0
    print("🔍 Analyse du portefeuille en cours...")

    for ticker, infos in data.items():
        secteur_actuel = infos.get('secteur', 'Inconnu')
        
        if secteur_actuel == "Inconnu" or secteur_actuel == "Divers":
            print(f"🔧 Réparation de {ticker}...")
            
            # On demande à Yahoo
            nouveau_secteur, _, _ = brain.recuperer_infos_avancees(ticker)
            
            if nouveau_secteur != "Inconnu":
                data[ticker]['secteur'] = nouveau_secteur
                print(f"   ✅ Corrigé : {nouveau_secteur}")
                compteur += 1
            else:
                print("   ⚠️ Pas trouvé sur Yahoo.")
    
    # Sauvegarde
    if compteur > 0:
        with open(FICHIER_PORTFOLIO, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n🎉 Terminé ! {compteur} positions mises à jour.")
    else:
        print("\n✅ Tout est déjà propre. Rien à faire.")

if __name__ == "__main__":
    reparer_secteurs()
