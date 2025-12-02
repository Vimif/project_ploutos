# notifier.py
# ---------------------------------------------------------
# SYSTÈME D'ALERTE DISCORD
# ---------------------------------------------------------
import requests
import json
from datetime import datetime
from config import DISCORD_WEBHOOK_URL

def send_discord(msg, type="INFO"):
    """
    Envoie un message formaté sur Discord.
    Types: INFO (Gris), TRADE (Vert), ALERT (Rouge), PROFIT (Or)
    """
    if not DISCORD_WEBHOOK_URL or "https" not in DISCORD_WEBHOOK_URL:
        return # Pas de config, on ne fait rien

    colors = {
        "INFO": 3447003,   # Bleu
        "TRADE": 5763719,  # Vert
        "ALERT": 15548997, # Rouge
        "PROFIT": 16776960 # Jaune/Or
    }

    data = {
        "username": "Ploutos AI",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/4712/4712009.png", # Icone Robot
        "embeds": [{
            "title": f"📢 {type}",
            "description": msg,
            "color": colors.get(type, 3447003),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Ploutos Trading System"}
        }]
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Erreur Discord: {e}")

# Test rapide
if __name__ == "__main__":
    send_discord("Ceci est un test de connexion.", "INFO")
    send_discord("ACHAT NVDA @ 120$", "TRADE")
