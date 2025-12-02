# logger_bot.py
# ---------------------------------------------------------
# BOÎTE NOIRE DU SYSTÈME
# ---------------------------------------------------------
import datetime

LOG_FILE = "activity.log"

def log(message, type="INFO"):
    """Types: INFO, SUCCESS, WARNING, ERROR"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icone = "ℹ️"
    if type == "SUCCESS": icone = "✅"
    elif type == "WARNING": icone = "⚠️"
    elif type == "ERROR": icone = "❌"
    elif type == "TRADE": icone = "💰"
    
    ligne = f"[{timestamp}] {icone} {message}\n"
    
    # Écriture Fichier
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(ligne)
    except: pass

def lire_logs(lignes=50):
    """Lit les X dernières lignes"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.readlines()[-lignes:]
    except: return []
