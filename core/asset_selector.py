"""
Sélectionneur d'actifs universel
Choisit automatiquement les meilleurs assets selon le contexte
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class UniversalAssetSelector:
    """Sélectionne les meilleurs assets selon le régime de marché"""
    
    # Univers d'investissement (classé par catégorie)
    UNIVERSE = {
        'growth_stocks': ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'CRM'],
        'defensive_stocks': ['PG', 'KO', 'JNJ', 'WMT', 'PEP', 'MCD', 'VZ', 'T'],
        'range_bound': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK'],
        'low_beta': ['VZ', 'KO', 'PG', 'JNJ', 'MCD', 'NEE', 'DUK', 'SO']
    }
    
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        
    def select_assets(self, n_assets=20, lookback_days=90):
        """Sélectionne les n meilleurs assets"""
        
        # 1. Obtenir régime actuel
        regime_info = self.regime_detector.detect()
        regime = regime_info['regime']
        strategy = self.regime_detector.get_optimal_strategy()
        
        # 2. Filtrer l'univers selon la catégorie
        category = strategy['asset_selection']
        candidates = self.UNIVERSE.get(category, self.UNIVERSE['range_bound'])
        
        # 3. Scorer chaque asset
        scored_assets = []
        
        for ticker in candidates:
            try:
                score = self._score_asset(ticker, regime)
                scored_assets.append({
                    'ticker': ticker,
                    'score': score
                })
            except Exception as e:
                print(f"⚠️ Erreur {ticker}: {e}")
                continue
        
        # 4. Trier et retourner top N
        df_scores = pd.DataFrame(scored_assets)
        df_scores = df_scores.sort_values('score', ascending=False).head(n_assets)
        
        selected = df_scores['ticker'].tolist()
        
        print(f"\n🎯 ASSETS SÉLECTIONNÉS ({regime}):")
        for idx, row in df_scores.iterrows():
            print(f"  {row['ticker']:6s} → Score: {row['score']:.2f}")
        
        return selected
    
    def _score_asset(self, ticker, regime):
        """Score un asset (0-100)"""
        
        # Télécharger données 90 jours
        data = yf.download(ticker, period='90d', interval='1d', progress=False)
        
        if len(data) < 30:
            return 0
        
        # Calculer métriques
        returns = data['Close'].pct_change().dropna()
        
        # 1. Performance récente (30j)
        perf_30d = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]
        
        # 2. Volatilité
        volatility = returns.std() * np.sqrt(252)
        
        # 3. Sharpe ratio
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # 4. Volume (liquidité)
        avg_volume = data['Volume'].mean()
        
        # SCORING ADAPTÉ AU RÉGIME
        if regime == 'BULL':
            # Bull : Favoriser momentum et performance
            score = (perf_30d * 50) + (sharpe * 30) + (min(avg_volume / 1e6, 20))
        
        elif regime == 'BEAR':
            # Bear : Favoriser faible volatilité et Sharpe positif
            score = max(0, (1 - volatility) * 40) + (sharpe * 40) + (min(avg_volume / 1e6, 20))
        
        elif regime == 'SIDEWAYS':
            # Sideways : Équilibré
            score = (abs(perf_30d) * 30) + (sharpe * 40) + (min(avg_volume / 1e6, 30))
        
        else:  # HIGH_VOLATILITY
            # High Vol : Minimum variance
            score = max(0, (1 - volatility) * 70) + (min(avg_volume / 1e6, 30))
        
        return max(0, min(score, 100))
