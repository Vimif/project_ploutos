# trading_bot.py
# ---------------------------------------------------------
# CERVEAU DE TRADING (AVEC CACHE ANTI-BAN)
# ---------------------------------------------------------
import yfinance as yf
import pandas as pd
import datetime
import json
import os
import time

class TradingBrain:
    def __init__(self):
        # Création du dossier cache si inexistant
        if not os.path.exists("cache_data"):
            os.makedirs("cache_data")
            
        self.params = self._charger_ai_config()

    def _charger_ai_config(self):
        return {'rsi_buy': 30, 'rsi_sell': 70, 'sma_trend': 50}

    def telecharger_donnees(self, ticker, periode="ignored"):
        """
        Télécharge les données avec un système de cache intelligent.
        Si le fichier existe et a moins de 1h, on l'utilise.
        Sinon, on télécharge et on sauvegarde.
        """
        cache_file = f"cache_data/{ticker}.csv"
        
        # 1. Vérification du Cache
        if os.path.exists(cache_file):
            age_sec = time.time() - os.path.getmtime(cache_file)
            if age_sec < 3600: # 1 Heure de validité
                try:
                    # On recharge depuis le disque (Ultra rapide)
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    return df
                except: pass # Si fichier corrompu, on continue

        # 2. Téléchargement Yahoo
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
        try:
            # Pause de courtoisie pour éviter le "Rate Limit"
            time.sleep(0.5) 
            
            df = yf.download(ticker, start=start_date, interval="1d", progress=False, auto_adjust=True)
            
            if df.empty: return None
            
            # Nettoyage MultiIndex (souvent le cas avec les nouvelles versions de yfinance)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            # 3. Sauvegarde dans le cache
            df.to_csv(cache_file)
            
            return df
        except Exception as e:
            print(f"Erreur download {ticker}: {e}")
            return None

    def ajouter_indicateurs(self, df):
        if df is None or df.empty: return df
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moyennes Mobiles
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        return df.dropna()

    def analyser_tendance_weekly(self, ticker):
        # Pour le weekly, on ne cache pas car c'est une requête rare
        try:
            time.sleep(0.5)
            df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
            if df is None or len(df) < 20: return "NEUTRE"
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            sma = df['Close'].rolling(30).mean()
            if df['Close'].iloc[-1] > sma.iloc[-1]: return "HAUSSIER"
            return "BAISSIER"
        except: return "NEUTRE"
    
    def recuperer_infos_avancees(self, ticker):
        try:
            info = yf.Ticker(ticker).info
            return info.get('sector', 'Divers'), info.get('recommendationKey', 'none').upper(), info.get('targetMeanPrice', 0)
        except: return "Inconnu", "AUCUN AVIS", 0

    def recuperer_profil_entreprise(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            desc = info.get('longBusinessSummary', "Pas de description.")[:300] + "..."
            earn = "Inconnue"
            if 'earningsTimestamp' in info and info['earningsTimestamp']:
                earn = datetime.datetime.fromtimestamp(info['earningsTimestamp']).strftime('%d/%m/%Y')
            return desc, earn
        except: return "Indisponible", "Inconnue"

    def analyser_dividendes(self, ticker):
        try:
            info = yf.Ticker(ticker).info
            yld = info.get('dividendYield', 0)
            if yld is None: yld = 0
            return round(yld * 100, 2), info.get('dividendRate', 0) or 0, "Inconnue"
        except: return 0.0, 0.0, "Inconnue"

    def analyser_tendance_marche(self):
        s = self.telecharger_donnees("SPY")
        if s is None: return "NEUTRE"
        s['SMA_200'] = s['Close'].rolling(200).mean()
        if s['Close'].iloc[-1] > s['SMA_200'].iloc[-1]: return "BULL"
        return "BEAR"

    def analyser_rotation_sectorielle(self):
            """
            Retourne un dictionnaire des performances des secteurs sur 5 jours.
            Permet à l'IA de savoir où va l'argent.
            """
            sectors = {
                "Technologie (XLK)": "XLK",
                "Finance (XLF)": "XLF",
                "Santé (XLV)": "XLV",
                "Energie (XLE)": "XLE",
                "Conso Discrétionnaire (XLY)": "XLY",
                "Industrie (XLI)": "XLI"
            }
            
            resultats = {}
            for nom, ticker in sectors.items():
                df = self.telecharger_donnees(ticker)
                if df is not None and len(df) > 5:
                    # Performance sur 5 jours glissants
                    perf = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
                    resultats[ticker] = round(perf, 2)
            
            # On trie du meilleur au pire
            sorted_sectors = dict(sorted(resultats.items(), key=lambda item: item[1], reverse=True))
            return sorted_sectors