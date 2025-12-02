# gestion_portefeuille.py
import json
import os
from datetime import datetime
import logger_bot

FICHIER_PORTFOLIO = "portefeuille.json"
FICHIER_HISTORIQUE = "historique_ventes.csv"

class PortefeuilleManager:
    def __init__(self):
        self.positions = self._charger_portfolio()

    def _charger_portfolio(self):
        if os.path.exists(FICHIER_PORTFOLIO):
            try:
                with open(FICHIER_PORTFOLIO, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def _sauvegarder(self):
        with open(FICHIER_PORTFOLIO, 'w') as f: json.dump(self.positions, f, indent=4)

    def ajouter_position(self, ticker, prix_achat, quantite, secteur="Inconnu"):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.positions[ticker] = {
            "prix_achat": prix_achat,
            "quantite": quantite,
            "secteur": secteur,
            "date_achat": date,
            "stop_loss": prix_achat * 0.95,
            "highest_price": prix_achat
        }
        self._sauvegarder()
        logger_bot.log(f"Achat Manuel : {quantite} {ticker} @ {prix_achat}$", "SUCCESS")

    def verifier_trailing_stop(self, ticker, prix_actuel):
        if ticker not in self.positions: return False
        info = self.positions[ticker]
        high = info.get("highest_price", info["prix_achat"])
        stop = info.get("stop_loss", info["prix_achat"] * 0.95)
        
        if prix_actuel > high:
            new_stop = prix_actuel * 0.95
            if new_stop > stop:
                self.positions[ticker]["highest_price"] = prix_actuel
                self.positions[ticker]["stop_loss"] = new_stop
                self._sauvegarder()
                return True
        return False

    def retirer_position(self, ticker, prix_vente):
        if ticker in self.positions:
            i = self.positions[ticker]
            gain = (prix_vente - i["prix_achat"]) * i["quantite"]
            perf = (prix_vente - i["prix_achat"]) / i["prix_achat"] * 100
            
            ligne = f"{ticker},{i['date_achat']},{datetime.now()},{i['prix_achat']},{prix_vente},{i['quantite']},{gain:.2f},{perf:.2f}\n"
            with open(FICHIER_HISTORIQUE, "a") as f:
                if os.stat(FICHIER_HISTORIQUE).st_size == 0:
                    f.write("Ticker,Date Achat,Date Vente,Prix Achat,Prix Vente,Qté,Gain Net,Perf %\n")
                f.write(ligne)
            
            del self.positions[ticker]
            self._sauvegarder()
            logger_bot.log(f"Vente : {ticker}. P/L: {gain:.2f}$", "TRADE")
            return gain, perf
        return 0, 0
