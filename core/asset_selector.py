"""
Sélectionneur d'actifs universel
Choisit automatiquement les meilleurs assets selon le contexte
"""

import pandas as pd
import numpy as np  # ✅ AJOUT ICI
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

class UniversalAssetSelector:
    """Sélectionne les meilleurs assets selon le régime de marché"""
    
    # Univers d'investissement étendu (classé par catégorie)
    UNIVERSE = {
        'growth_stocks': [
            'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 
            'AMD', 'CRM', 'ADBE', 'NFLX', 'AVGO', 'ORCL'
        ],
        'defensive_stocks': [
            'PG', 'KO', 'JNJ', 'WMT', 'PEP', 'MCD', 'VZ', 'T',
            'NEE', 'DUK', 'SO', 'AEP'
        ],
        'range_bound': [
            'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 
            'XLV', 'XLP', 'XLU'
        ],
        'low_beta': [
            'VZ', 'KO', 'PG', 'JNJ', 'MCD', 'NEE', 'DUK', 'SO',
            'WMT', 'PEP', 'T', 'AEP'
        ]
    }
    
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        self.last_selection = None
        self.selection_history = []
        
    def select_assets(self, n_assets=20, lookback_days=90):
        """
        Sélectionne les n meilleurs assets
        
        Args:
            n_assets: Nombre d'assets à sélectionner
            lookback_days: Période d'analyse
            
        Returns:
            list: Tickers sélectionnés
        """
        
        print(f"\n🎯 Sélection de {n_assets} assets...")
        
        # 1. Obtenir régime actuel
        regime_info = self.regime_detector.detect()
        regime = regime_info['regime']
        strategy = self.regime_detector.get_optimal_strategy(regime)
        
        # 2. Filtrer l'univers selon la catégorie
        category = strategy['asset_selection']
        candidates = self.UNIVERSE.get(category, self.UNIVERSE['range_bound'])
        
        print(f"  📊 Catégorie : {category} ({len(candidates)} candidats)")
        
        # 3. Scorer chaque asset EN PARALLÈLE
        scored_assets = self._score_assets_parallel(candidates, regime, lookback_days)
        
        # 4. Filtrer les assets avec score nul (pas de données)
        scored_assets = [a for a in scored_assets if a and a.get('score', 0) > 0]
        
        # ✅ FIX : Si aucun score valide, utiliser fallback
        if len(scored_assets) == 0:
            print("⚠️ Aucun asset scoré, fallback sur SPY + QQQ + NVDA")
            return ['SPY', 'QQQ', 'NVDA']
        
        # 5. Trier et retourner top N
        df_scores = pd.DataFrame(scored_assets)
        df_scores = df_scores.sort_values('score', ascending=False).head(n_assets)
        
        selected = df_scores['ticker'].tolist()
        
        # Affichage
        print(f"\n  🏆 TOP {len(selected)} ASSETS SÉLECTIONNÉS ({regime}):")
        for idx, row in df_scores.head(10).iterrows():
            print(f"    {row['ticker']:6s} → Score: {row['score']:5.1f} | "
                  f"Perf: {row.get('perf_30d', 0)*100:+5.1f}% | "
                  f"Sharpe: {row.get('sharpe', 0):4.2f}")
        
        # Sauvegarder
        self.last_selection = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'assets': selected,
            'scores': df_scores.to_dict('records')
        }
        self.selection_history.append(self.last_selection)
        
        return selected
    
    def _score_assets_parallel(self, candidates, regime, lookback_days):
        """Score les assets en parallèle pour plus de rapidité"""
        
        scored_assets = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._score_asset, ticker, regime, lookback_days): ticker 
                for ticker in candidates
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        scored_assets.append(result)
                except Exception as e:
                    print(f"    ⚠️ Erreur {ticker}: {str(e)[:50]}")
        
        return scored_assets
    
    def _score_asset(self, ticker, regime, lookback_days):
        """
        Score un asset (0-100)
        
        Métriques évaluées :
        - Performance récente (30j)
        - Volatilité
        - Sharpe ratio
        - Volume (liquidité)
        - Max Drawdown
        """
        
        try:
            # Télécharger données
            data = yf.download(
                ticker, 
                period=f'{lookback_days}d', 
                interval='1d', 
                progress=False
            )
            
            # Si MultiIndex, extraire
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs(ticker, axis=1, level=1)
            
            if len(data) < 30:
                return None
            
            # Calculer métriques
            returns = data['Close'].pct_change().dropna()
            
            # 1. Performance récente (30j)
            perf_30d = float((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30])
            
            # 2. Volatilité annualisée
            volatility = float(returns.std() * np.sqrt(252))
            
            # 3. Sharpe ratio
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
            
            # 4. Volume moyen (liquidité)
            avg_volume = float(data['Volume'].mean())
            volume_score = min(avg_volume / 1e6, 20)
            
            # 5. Max Drawdown
            cummax = data['Close'].cummax()
            drawdowns = (data['Close'] - cummax) / cummax
            max_drawdown = float(drawdowns.min())
            
            # SCORING ADAPTÉ AU RÉGIME
            if regime == 'BULL':
                # Bull : Favoriser momentum fort et Sharpe élevé
                score = (
                    max(0, perf_30d * 100) * 0.40 +    # Performance
                    max(0, sharpe * 10) * 0.40 +        # Sharpe
                    volume_score * 0.20                  # Liquidité
                )
            
            elif regime == 'BEAR':
                # Bear : Favoriser faible volatilité, Sharpe positif, faible drawdown
                score = (
                    max(0, (1 - volatility) * 50) * 0.30 +  # Anti-volatilité
                    max(0, sharpe * 10) * 0.40 +             # Sharpe
                    max(0, (1 + max_drawdown) * 50) * 0.20 + # Résistance DD
                    volume_score * 0.10
                )
            
            elif regime == 'SIDEWAYS':
                # Sideways : Équilibré, favoriser Sharpe
                score = (
                    abs(perf_30d * 100) * 0.20 +
                    max(0, sharpe * 10) * 0.50 +
                    volume_score * 0.30
                )
            
            else:  # HIGH_VOLATILITY
                # High Vol : Minimum variance, liquidité maximale
                score = (
                    max(0, (1 - volatility) * 100) * 0.60 +
                    volume_score * 0.40
                )
            
            score = float(max(0, min(score, 100)))
            
            return {
                'ticker': ticker,
                'score': score,
                'perf_30d': perf_30d,
                'volatility': volatility,
                'sharpe': sharpe,
                'volume': avg_volume,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            # Si erreur, retourner None (sera filtré)
            return None
    
    def save_history(self, filepath='data/selection_history.json'):
        """Sauvegarde l'historique des sélections"""
        if len(self.selection_history) > 0:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.selection_history, f, indent=2)
            print(f"💾 Historique sélection sauvegardé : {filepath}")

# À ajouter dans la classe UniversalAssetSelector

def __init__(self, regime_detector, enable_market_scan=False):
    """
    Args:
        enable_market_scan: Si True, active le scan complet du marché
    """
    self.regime_detector = regime_detector
    self.last_selection = None
    self.selection_history = []
    self.enable_market_scan = enable_market_scan
    
    # Initialiser scanner si activé
    if enable_market_scan:
        from core.data_fetcher import UniversalDataFetcher
        from core.market_scanner import MarketScanner
        
        self.fetcher = UniversalDataFetcher()
        self.scanner = MarketScanner(self.fetcher)
        print("🔍 Market Scanner activé")

def select_assets(self, n_assets=20, lookback_days=90, use_market_scan=False):
    """
    Sélectionne les meilleurs assets
    
    Args:
        n_assets: Nombre d'assets à sélectionner
        lookback_days: Période d'analyse
        use_market_scan: Si True, scanne tout le marché US
        
    Returns:
        list: Tickers sélectionnés
    """
    
    # MODE 1 : Scan complet du marché (3000+ actions)
    if use_market_scan and self.enable_market_scan:
        print("\n🔍 MODE : MARKET SCAN COMPLET")
        
        # Scanner tout le marché
        df_scored = self.scanner.scan_full_market(
            min_market_cap=1e9,   # 1 milliard minimum
            min_volume=1e6,       # 1 million volume/jour
            max_stocks=500        # Limiter à top 500 pour vitesse
        )
        
        if len(df_scored) == 0:
            print("⚠️ Scan échoué, fallback sur univers fixe")
            return self._select_from_universe(n_assets, lookback_days)
        
        # Stratégie de sélection selon régime
        regime_info = self.regime_detector.detect()
        regime = regime_info['regime']
        
        if regime in ['BULL', 'SIDEWAYS']:
            # En bull/sideways : Prendre les meilleurs globalement
            selected = self.scanner.get_top_overall(df_scored, n=n_assets)
        else:
            # En bear/high_vol : Diversifier par secteur
            selected = self.scanner.get_top_by_sector(df_scored, n_per_sector=2)
            selected = selected[:n_assets]
        
        # Sauvegarder résultats
        self.scanner.save_results()
        
        return selected
    
    # MODE 2 : Univers fixe (mode classique)
    else:
        print("\n🎯 MODE : UNIVERS FIXE")
        return self._select_from_universe(n_assets, lookback_days)

def _select_from_universe(self, n_assets, lookback_days):
    """Mode classique (code existant)"""
    
    # ... Le code existant de select_assets() va ici ...
    # (Tout ce que tu avais avant, inchangé)
    
    regime_info = self.regime_detector.detect()
    regime = regime_info['regime']
    strategy = self.regime_detector.get_optimal_strategy(regime)
    
    category = strategy['asset_selection']
    candidates = self.UNIVERSE.get(category, self.UNIVERSE['range_bound'])
    
    print(f"  📊 Catégorie : {category} ({len(candidates)} candidats)")
    
    scored_assets = self._score_assets_parallel(candidates, regime, lookback_days)
    scored_assets = [a for a in scored_assets if a and a.get('score', 0) > 0]
    
    if len(scored_assets) == 0:
        print("⚠️ Aucun asset scoré, fallback sur SPY + QQQ + NVDA")
        return ['SPY', 'QQQ', 'NVDA']
    
    df_scores = pd.DataFrame(scored_assets)
    df_scores = df_scores.sort_values('score', ascending=False).head(n_assets)
    
    selected = df_scores['ticker'].tolist()
    
    print(f"\n  🏆 TOP {len(selected)} ASSETS ({regime}):")
    for idx, row in df_scores.head(10).iterrows():
        print(f"    {row['ticker']:6s} → Score: {row['score']:5.1f}")
    
    return selected
