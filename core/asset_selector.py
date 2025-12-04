"""
Sélectionneur d'actifs universel
Choisit automatiquement les meilleurs assets selon le contexte
"""

import pandas as pd
import numpy as np
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
    
    def __init__(self, regime_detector, enable_market_scan=False):
        """
        Args:
            regime_detector: Instance de MarketRegimeDetector
            enable_market_scan: Si True, active le scan complet du marché
        """
        self.regime_detector = regime_detector
        self.last_selection = None
        self.selection_history = []
        self.enable_market_scan = enable_market_scan
        
        # Initialiser scanner si activé
        self.fetcher = None
        self.scanner = None
        
        if enable_market_scan:
            try:
                from core.data_fetcher import UniversalDataFetcher
                from core.market_scanner import MarketScanner
                
                self.fetcher = UniversalDataFetcher()
                self.scanner = MarketScanner(self.fetcher)
                print("🔍 Market Scanner activé")
            except ImportError as e:
                print(f"⚠️ Market Scanner non disponible : {e}")
                print("  → Installer : pip install alpaca-trade-api requests")
                self.enable_market_scan = False
        
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
        if use_market_scan and self.enable_market_scan and self.scanner:
            print("\n🔍 MODE : MARKET SCAN COMPLET")
            
            try:
                # Scanner tout le marché
                df_scored = self.scanner.scan_full_market(
                    min_market_cap=1e9,
                    min_volume=1e6,
                    max_stocks=500
                )
                
                if len(df_scored) == 0:
                    print("⚠️ Scan échoué, fallback sur univers fixe")
                    return self._select_from_universe(n_assets, lookback_days)
                
                # Stratégie selon régime
                regime_info = self.regime_detector.detect()
                regime = regime_info['regime']
                
                if regime in ['BULL', 'SIDEWAYS']:
                    selected = self.scanner.get_top_overall(df_scored, n=n_assets)
                else:
                    selected = self.scanner.get_top_by_sector(df_scored, n_per_sector=2)
                    selected = selected[:n_assets]
                
                self.scanner.save_results()
                return selected
                
            except Exception as e:
                print(f"⚠️ Erreur Market Scan : {e}")
                print("  → Fallback sur univers fixe")
                return self._select_from_universe(n_assets, lookback_days)
        
        # MODE 2 : Univers fixe
        else:
            if use_market_scan:
                print("⚠️ Market Scan demandé mais non disponible")
            print("\n🎯 MODE : UNIVERS FIXE")
            return self._select_from_universe(n_assets, lookback_days)
    
    def _select_from_universe(self, n_assets, lookback_days):
        """Mode classique avec univers fixe"""
        
        print(f"\n🎯 Sélection de {n_assets} assets...")
        
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
            print(f"    {row['ticker']:6s} → Score: {row['score']:5.1f} | "
                  f"Perf: {row.get('perf_30d', 0)*100:+5.1f}% | "
                  f"Sharpe: {row.get('sharpe', 0):4.2f}")
        
        self.last_selection = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'assets': selected,
            'scores': df_scores.to_dict('records')
        }
        self.selection_history.append(self.last_selection)
        
        return selected
    
    def _score_assets_parallel(self, candidates, regime, lookback_days):
        """Score les assets en parallèle"""
        
        scored_assets = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
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
        """Score un asset (0-100)"""
        
        try:
            # Télécharger
            data = yf.download(ticker, period=f'{lookback_days}d', interval='1d', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 30:
                print(f"    ⚠️ {ticker}: Pas assez de données ({len(data)} jours)")
                return None
            
            returns = data['Close'].pct_change().dropna()
            
            # Momentum 30j
            if len(data) < 30:
                perf_30d = 0.0
            else:
                perf_30d = float((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30])
            
            # Volatilité
            volatility = float(returns.std() * np.sqrt(252))
            
            # Sharpe
            if returns.std() > 0:
                sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
            else:
                sharpe = 0.0
            
            # Volume
            avg_volume = float(data['Volume'].mean())
            volume_score = min(avg_volume / 1e6, 20)
            
            # Drawdown
            cummax = data['Close'].cummax()
            drawdowns = (data['Close'] - cummax) / cummax
            max_drawdown = float(drawdowns.min())
            
            # DEBUG
            print(f"    📊 {ticker}: perf={perf_30d:.2%}, vol={volatility:.2f}, sharpe={sharpe:.2f}")
            
            # SCORING selon régime
            if regime == 'BULL':
                score = (
                    max(0, perf_30d * 100) * 0.40 +
                    max(0, sharpe * 10) * 0.40 +
                    volume_score * 0.20
                )
            elif regime == 'BEAR':
                score = (
                    max(0, (1 - volatility) * 50) * 0.30 +
                    max(0, sharpe * 10) * 0.40 +
                    max(0, (1 + max_drawdown) * 50) * 0.20 +
                    volume_score * 0.10
                )
            elif regime == 'SIDEWAYS':
                score = (
                    abs(perf_30d * 100) * 0.20 +
                    max(0, sharpe * 10) * 0.50 +
                    volume_score * 0.30
                )
            else:  # HIGH_VOLATILITY
                score = (
                    max(0, (1 - volatility) * 100) * 0.60 +
                    volume_score * 0.40
                )
            
            score = float(max(0, min(score, 100)))
            
            print(f"    ✅ {ticker}: score final = {score:.1f}")
            
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
            print(f"    ❌ {ticker}: {str(e)}")
            return None
    
    def save_history(self, filepath='data/selection_history.json'):
        """Sauvegarde l'historique des sélections"""
        if len(self.selection_history) > 0:
            import json
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.selection_history, f, indent=2)
            print(f"💾 Historique sélection sauvegardé : {filepath}")
