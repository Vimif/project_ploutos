# config.py
# ---------------------------------------------------------
# CENTRE DE CONFIGURATION PLOUTOS
# ---------------------------------------------------------
import os

# --- CLES API (À REMPLIR UNE SEULE FOIS ICI) ---
ALPACA_API_KEY = "PKTEM6CRKRFNOE4HA7GU2IWFJ5"
ALPACA_SECRET_KEY = "DShnhnKy9pZabABkKE1rZX9yEtZpUeet1jFmLbWQGrCC"
ALPACA_PAPER = True

# --- FICHIERS ---
FILE_PORTFOLIO = "portefeuille.json"
FILE_WATCHLIST = "watchlist.json"
FILE_SCAN_RESULTS = "shortlist_ia.json"
FILE_MODEL_BRAIN = "ppo_trading_brain"
FILE_HISTORY = "historique_ventes.csv"
DIR_CACHE = "cache_data"

# --- PARAMETRES TRADING ---
MAX_POSITION_SIZE = 2000     # Max $ par position
CHECK_INTERVAL = 300         # 15 minutes entre chaque check auto
RISK_REWARD_RATIO = 2.0      # Objectif de gain vs perte

# --- UNIVERS D'INVESTISSEMENT ---
# Les ETFs sectoriels pour analyser la rotation
SECTOR_ETFS = {
    "Tech": "XLK", "Finance": "XLF", "Santé": "XLV",
    "Energie": "XLE", "Conso": "XLY", "Industrie": "XLI"
}

# Liste de secours si le scan est vide
DEFAULT_WATCHLIST = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN"]

# Création auto des dossiers
if not os.path.exists(DIR_CACHE): os.makedirs(DIR_CACHE)

# --- NOTIFICATIONS ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1445133574801195183/HvQKgnHcASHAoPq8X6C0_bUVhHTku-E9cMR0dqmfkN04M16fpfnmq5BldGtkEVt83BRz" 