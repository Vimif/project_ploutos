# discord_bot.py
# ---------------------------------------------------------
# MODULE DE NOTIFICATION DISCORD
# ---------------------------------------------------------
import requests
import datetime

# --- REMPLACEZ PAR VOTRE URL WEBHOOK ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1445133574801195183/HvQKgnHcASHAoPq8X6C0_bUVhHTku-E9cMR0dqmfkN04M16fpfnmq5BldGtkEVt83BRz"

def envoyer_notif(titre, message, couleur="BLUE"):
    """
    Envoie une belle 'Embed' (Carte) sur Discord.
    Couleurs : GREEN (Gain/Achat), RED (Perte/Vente), BLUE (Info), GOLD (Alerte)
    """
    if "VOTRE_URL" in DISCORD_WEBHOOK_URL:
        print("❌ Erreur : Webhook Discord non configuré.")
        return

    # Codes couleurs Discord (Décimal)
    colors = {
        "GREEN": 3066993,  # Vert
        "RED": 15158332,   # Rouge
        "BLUE": 3447003,   # Bleu
        "GOLD": 15844367   # Or
    }
    
    col_dec = colors.get(couleur, 3447003)
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    data = {
        "embeds": [
            {
                "title": f"{titre}",
                "description": message,
                "color": col_dec,
                "footer": {"text": f"🦅 Ploutos Station • {timestamp}"}
            }
        ]
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Erreur Discord: {e}")

# Petit test si on lance le fichier directement
if __name__ == "__main__":
    envoyer_notif("Test de Connexion", "Le système Ploutos est connecté à Discord !", "GREEN")
