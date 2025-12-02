# news_analyst.py
# ---------------------------------------------------------
# ANALYSEUR DE SENTIMENT (NEWS & MÉDIAS)
# ---------------------------------------------------------
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    """
    Retourne un score de sentiment entre -1 (Très Négatif) et +1 (Très Positif).
    """
    try:
        # On récupère les news via l'objet Ticker de yfinance
        t = yf.Ticker(ticker)
        news_list = t.news
        
        if not news_list:
            return 0.0 # Neutre si pas de news
        
        total_score = 0
        count = 0
        
        # On analyse les titres des 5 dernières news
        for n in news_list[:5]:
            title = n.get('title', '')
            # Analyse VADER
            vs = analyzer.polarity_scores(title)
            total_score += vs['compound']
            count += 1
            
        if count == 0: return 0.0
        
        avg_score = total_score / count
        return round(avg_score, 2)

    except Exception as e:
        print(f"Erreur Sentiment {ticker}: {e}")
        return 0.0

# Test rapide si lancé directement
if __name__ == "__main__":
    tk = "TSLA"
    print(f"Sentiment actuel sur {tk} : {get_sentiment(tk)}")
