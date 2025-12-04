"""
Market Scanner - Découverte Automatique de Pépites
Scanne 3000+ actions US pour trouver les opportunités par secteur
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MarketScanner:
    """
    Scanne tout le marché US pour trouver les meilleures opportunités
    Analyse technique + fondamentale + sectorielle
    """
    
    # Secteurs GICS (Global Industry Classification Standard)
    SECTORS = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Energy': 'XLE',
        'Materials': 'XLB',
        'Consumer Staples': 'XLP',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Communication Services': 'XLC'
    }
    
    def __init__(self, data_fetcher=None):
        """
        Args:
            data_fetcher: Instance de UniversalDataFetcher (optionnel)
        """
        self.fetcher = data_fetcher
        self.scan_results = None
        
    def scan_full_market(self, min_market_cap=1e9, min_volume=1e6, max_stocks=500):
        """
        Scanne TOUTES les actions US
        
        Args:
            min_market_cap: Cap min en $ (1e9 = 1 milliard)
            min_volume: Volume quotidien min
            max_stocks: Limite pour accélérer (500 = top 500 par cap)
            
        Returns:
            pd.DataFrame avec toutes les actions scorées
        """
        
        print("\n" + "="*70)
        print("🔍 MARKET SCANNER - FULL US EQUITIES")
        print("="*70)
        
        # 1. Obtenir la liste complète
        print("\n📋 Phase 1 : Récupération liste complète...")
        all_tickers = self._get_all_us_stocks()
        print(f"  → {len(all_tickers)} actions trouvées")
        
        # 2. Filtrer par fondamentaux
        print(f"\n🔬 Phase 2 : Filtrage (market cap > ${min_market_cap/1e9:.1f}B, volume > {min_volume/1e6:.1f}M)")
        df_filtered = self._filter_by_fundamentals(all_tickers, min_market_cap, min_volume, max_stocks)
        print(f"  → {len(df_filtered)} actions retenues")
        
        if len(df_filtered) == 0:
            print("❌ Aucune action ne passe les filtres")
            return pd.DataFrame()
        
        # 3. Analyse technique
        print(f"\n📊 Phase 3 : Analyse technique (momentum, volatilité, tendances)...")
        df_scored = self._score_all_stocks(df_filtered)
        print(f"  → {len(df_scored)} actions scorées")
        
        # Sauvegarder
        self.scan_results = df_scored
        
        # Statistiques
        print(f"\n📈 Statistiques :")
        print(f"  Secteurs couverts : {df_scored['sector'].nunique()}")
        print(f"  Score moyen       : {df_scored['score'].mean():.1f}/100")
        print(f"  Score max         : {df_scored['score'].max():.1f}")
        
        return df_scored
    
    def _get_all_us_stocks(self):
        """
        Récupère la liste complète des actions US tradables
        
        Sources :
        1. Alpaca universe (si disponible)
        2. NASDAQ Screener API (fallback)
        3. Liste hardcodée (dernier recours)
        """
        
        tickers = []
        
        # Méthode 1 : Alpaca Universe (le plus fiable)
        if self.fetcher and self.fetcher.alpaca_api:
            try:
                print("  Tentative Alpaca...")
                assets = self.fetcher.alpaca_api.list_assets(
                    status='active',
                    asset_class='us_equity'
                )
                
                alpaca_tickers = [
                    a.symbol for a in assets 
                    if a.tradable and a.fractionable and not a.symbol.startswith('$')
                ]
                
                tickers.extend(alpaca_tickers)
                print(f"    ✅ Alpaca : {len(alpaca_tickers)} actions")
                
            except Exception as e:
                print(f"    ⚠️ Alpaca échec : {str(e)[:50]}")
        
        # Méthode 2 : NASDAQ Screener (API publique)
        if len(tickers) == 0:
            try:
                print("  Tentative NASDAQ Screener...")
                import requests
                
                # NASDAQ
                url_nasdaq = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=NASDAQ"
                resp = requests.get(url_nasdaq, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    nasdaq_tickers = [row['symbol'] for row in data['data']['table']['rows']]
                    tickers.extend(nasdaq_tickers)
                    print(f"    ✅ NASDAQ : {len(nasdaq_tickers)} actions")
                
                # NYSE
                url_nyse = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=NYSE"
                resp = requests.get(url_nyse, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    nyse_tickers = [row['symbol'] for row in data['data']['table']['rows']]
                    tickers.extend(nyse_tickers)
                    print(f"    ✅ NYSE : {len(nyse_tickers)} actions")
                    
            except Exception as e:
                print(f"    ⚠️ NASDAQ Screener échec : {str(e)[:50]}")
        
        # Méthode 3 : Liste Top 500 hardcodée (dernier recours)
        if len(tickers) == 0:
            print("  Fallback : Liste SP500 + NASDAQ100...")
            tickers = self._get_hardcoded_universe()
            print(f"    ✅ Liste hardcodée : {len(tickers)} actions")
        
        # Nettoyage
        # Retirer les ETFs, warrants, préférées, etc.
        tickers = [
            t for t in tickers 
            if '-' not in t and '.' not in t and '^' not in t 
            and len(t) <= 5 and not t.endswith('W')
        ]
        
        # Dédupliquer
        tickers = list(set(tickers))
        
        return tickers
    
    def _get_hardcoded_universe(self):
        """Liste des principales actions US (fallback)"""
        
        # Top 100 par market cap (mise à jour déc 2024)
        return [
            # Mega Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'V', 'UNH', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'XOM', 'HD', 'CVX',
            'LLY', 'MRK', 'ABBV', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'ORCL',
            'ACN', 'MCD', 'CSCO', 'ABT', 'DHR', 'NKE', 'TXN', 'VZ', 'DIS',
            'ADBE', 'NFLX', 'CRM', 'NVO', 'PM', 'INTC', 'AMD', 'QCOM', 'UPS',
            
            # Large Cap Growth
            'INTU', 'AMGN', 'RTX', 'HON', 'SPGI', 'LOW', 'NEE', 'BMY', 'UNP',
            'LIN', 'BA', 'GS', 'CAT', 'ELV', 'SBUX', 'DE', 'MDT', 'AXP', 'BLK',
            
            # Tech & Software
            'NOW', 'ISRG', 'GILD', 'PLD', 'ADI', 'TJX', 'MDLZ', 'REGN', 'VRTX',
            'BKNG', 'MMC', 'CI', 'CVS', 'SCHW', 'SYK', 'CB', 'ZTS', 'TMUS',
            'MU', 'PGR', 'SO', 'DUK', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS',
            
            # Mid Cap avec potentiel
            'PANW', 'CRWD', 'DDOG', 'NET', 'SNOW', 'PLTR', 'RBLX', 'COIN',
            'ABNB', 'UBER', 'DASH', 'SQ', 'SHOP', 'ROKU', 'ZM', 'DOCU',
            
            # Energy & Materials
            'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'FCX', 'NEM', 'APD',
            
            # Healthcare & Biotech
            'MRNA', 'BIIB', 'ILMN', 'ALXN', 'IQV', 'IDXX', 'DGX', 'MTD',
            
            # Industrials
            'GE', 'MMM', 'EMR', 'ITW', 'PH', 'WM', 'ETN', 'CMI', 'NSC',
            
            # Consumer
            'SBUX', 'LULU', 'ROST', 'ULTA', 'DG', 'DLTR', 'YUM', 'ORLY',
            
            # Finance
            'MS', 'C', 'BAC', 'WFC', 'USB', 'PNC', 'TFC', 'COF', 'AIG',
            
            # REITs & Utilities
            'AMT', 'CCI', 'EQIX', 'PSA', 'DLR', 'SPG', 'O', 'WELL', 'ARE'
        ]
    
    def _filter_by_fundamentals(self, tickers, min_market_cap, min_volume, max_stocks):
        """
        Filtre par fondamentaux (parallélisé pour vitesse)
        
        Returns:
            DataFrame avec [ticker, market_cap, volume, sector, industry]
        """
        
        filtered = []
        
        def check_ticker(ticker):
            """Vérifie un ticker (fondamentaux)"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extraire fondamentaux
                market_cap = info.get('marketCap', 0)
                avg_volume = info.get('averageVolume', 0) or info.get('volume', 0)
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                # Filtres
                if market_cap >= min_market_cap and avg_volume >= min_volume:
                    return {
                        'ticker': ticker,
                        'market_cap': market_cap,
                        'volume': avg_volume,
                        'sector': sector,
                        'industry': industry
                    }
                    
            except Exception:
                pass
            
            return None
        
        # Parallélisation massive (50 threads)
        print("    Processing...", end='', flush=True)
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(check_ticker, tickers))
        
        print(" Done")
        
        filtered = [r for r in results if r is not None]
        
        # Convertir en DataFrame
        df = pd.DataFrame(filtered)
        
        if len(df) == 0:
            return df
        
        # Trier par market cap et limiter
        df = df.sort_values('market_cap', ascending=False).head(max_stocks)
        
        return df
    
    def _score_all_stocks(self, df_filtered):
        """
        Score technique pour chaque action (parallélisé)
        
        Métriques :
        - Momentum 30j
        - Volatilité annualisée
        - RSI
        - Tendance (MA50 vs prix)
        - Volume trend
        """
        
        def score_ticker(row):
            """Score une action (analyse technique)"""
            ticker = row['ticker']
            
            try:
                # Télécharger 90 jours
                data = yf.download(ticker, period='90d', interval='1d', progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if len(data) < 50:
                    return None
                
                # Métriques
                returns = data['Close'].pct_change().dropna()
                
                # 1. Momentum (30 jours)
                momentum = float((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30])
                
                # 2. Volatilité annualisée
                volatility = float(returns.std() * np.sqrt(252))
                
                # 3. RSI (14 jours)
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                rsi = float(100 - (100 / (1 + rs.iloc[-1])))
                
                # 4. Tendance (prix vs MA50)
                ma50 = data['Close'].rolling(50).mean().iloc[-1]
                trend = float((data['Close'].iloc[-1] - ma50) / ma50)
                
                # 5. Volume trend
                volume_ma = data['Volume'].rolling(20).mean().iloc[-1]
                volume_current = data['Volume'].iloc[-5:].mean()
                volume_trend = float((volume_current - volume_ma) / volume_ma)
                
                # 6. Sharpe Ratio (90j)
                sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0
                
                # SCORE COMPOSITE (0-100)
                score = (
                    max(0, momentum * 100) * 0.25 +         # Momentum fort
                    max(0, (1 - volatility) * 50) * 0.15 +  # Faible volatilité
                    (100 - abs(rsi - 50)) * 0.20 +          # RSI équilibré (autour 50)
                    max(0, trend * 100) * 0.20 +            # Tendance haussière
                    max(0, volume_trend * 50) * 0.10 +      # Volume croissant
                    max(0, sharpe * 10) * 0.10              # Bon Sharpe
                )
                
                return {
                    'ticker': ticker,
                    'score': float(score),
                    'momentum': momentum,
                    'volatility': volatility,
                    'rsi': rsi,
                    'trend': trend,
                    'volume_trend': volume_trend,
                    'sharpe': sharpe,
                    'sector': row['sector'],
                    'industry': row['industry'],
                    'market_cap': row['market_cap']
                }
                
            except Exception:
                return None
        
        # Parallélisation
        print("    Scoring...", end='', flush=True)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(score_ticker, [row for _, row in df_filtered.iterrows()]))
        
        print(" Done")
        
        scored = [r for r in results if r is not None]
        
        df_scored = pd.DataFrame(scored)
        df_scored = df_scored.sort_values('score', ascending=False)
        
        return df_scored
    
    def get_top_by_sector(self, df_scored=None, n_per_sector=3):
        """
        Sélectionne les N meilleures actions PAR SECTEUR
        
        Args:
            df_scored: DataFrame scoré (ou utilise self.scan_results)
            n_per_sector: Nombre d'actions par secteur
            
        Returns:
            List de tickers (diversifié sectoriellement)
        """
        
        if df_scored is None:
            df_scored = self.scan_results
        
        if df_scored is None or len(df_scored) == 0:
            raise ValueError("Aucun scan disponible, lancer scan_full_market() d'abord")
        
        top_picks = []
        
        print(f"\n🏆 TOP {n_per_sector} PAR SECTEUR")
        print("="*70)
        
        for sector in df_scored['sector'].unique():
            if sector == 'Unknown':
                continue
            
            sector_stocks = df_scored[df_scored['sector'] == sector].head(n_per_sector)
            
            print(f"\n📁 {sector}")
            for _, row in sector_stocks.iterrows():
                print(f"  {row['ticker']:6s} → Score: {row['score']:5.1f} | "
                      f"Momentum: {row['momentum']*100:+5.1f}% | "
                      f"Sharpe: {row['sharpe']:4.2f}")
            
            top_picks.append(sector_stocks)
        
        result = pd.concat(top_picks).sort_values('score', ascending=False)
        
        print(f"\n✅ {len(result)} actions sélectionnées (diversifiées)")
        
        return result['ticker'].tolist()
    
    def get_top_overall(self, df_scored=None, n=20):
        """Retourne simplement les N meilleures actions (sans contrainte sectorielle)"""
        
        if df_scored is None:
            df_scored = self.scan_results
        
        if df_scored is None:
            raise ValueError("Aucun scan disponible")
        
        top = df_scored.head(n)
        
        print(f"\n🏆 TOP {n} GLOBAL")
        print("="*70)
        
        for _, row in top.iterrows():
            print(f"{row['ticker']:6s} ({row['sector']:20s}) → "
                  f"Score: {row['score']:5.1f} | "
                  f"Momentum: {row['momentum']*100:+5.1f}%")
        
        return top['ticker'].tolist()
    
    def save_results(self, filepath='reports/market_scan_latest.csv'):
        """Sauvegarde les résultats du scan"""
        
        if self.scan_results is None:
            print("⚠️ Aucun résultat à sauvegarder")
            return
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.scan_results.to_csv(filepath, index=False)
        print(f"💾 Résultats sauvegardés : {filepath}")
