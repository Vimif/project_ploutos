"""Scanner S&P 500 par secteur GICS

Selectionne les meilleures actions de chaque secteur GICS du S&P 500
en se basant sur le Sharpe ratio annualise.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

from core.data_fetcher import UniversalDataFetcher


class SP500Scanner:
    """Scanne le S&P 500 et selectionne les top performers par secteur GICS."""

    GICS_SECTORS = [
        'Information Technology',
        'Health Care',
        'Financials',
        'Consumer Discretionary',
        'Communication Services',
        'Industrials',
        'Consumer Staples',
        'Energy',
        'Utilities',
        'Real Estate',
        'Materials',
    ]

    def __init__(self, cache_dir: str = 'data/sp500_cache', lookback_days: int = 252):
        self.cache_dir = Path(cache_dir)
        self.lookback_days = lookback_days
        self.risk_free_rate = 0.04
        self.fetcher = UniversalDataFetcher()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # S&P 500 list
    # ------------------------------------------------------------------
    def fetch_sp500_list(self) -> pd.DataFrame:
        """Recupere la liste des constituants S&P 500 depuis Wikipedia."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0][['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        # Nettoyer les symboles (BRK.B -> BRK-B pour yfinance)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df

    # ------------------------------------------------------------------
    # Sharpe ratio
    # ------------------------------------------------------------------
    def _calculate_sharpe(self, ticker: str) -> float:
        """Calcule le Sharpe ratio annualise sur 12 mois (daily data)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 60)

        try:
            df = self.fetcher.fetch(
                ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1d',
            )

            if df is None or len(df) < int(self.lookback_days * 0.7):
                return np.nan

            returns = df['Close'].pct_change().dropna().tail(self.lookback_days)
            if len(returns) < 50:
                return np.nan

            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)

            if annual_vol <= 0:
                return 0.0

            return (annual_return - self.risk_free_rate) / annual_vol

        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------
    def scan_sectors(
        self,
        stocks_per_sector: int = 2,
        max_workers: int = 5,
    ) -> Dict[str, Any]:
        """Scanne tous les secteurs GICS et selectionne les top performers.

        Args:
            stocks_per_sector: Nombre d'actions a selectionner par secteur.
            max_workers: Workers paralleles pour le calcul des Sharpe.

        Returns:
            Dictionnaire avec les resultats du scan.
        """
        sys.stdout.reconfigure(encoding='utf-8')

        print("Fetching S&P 500 constituents...")
        constituents = self.fetch_sp500_list()
        print(f"Found {len(constituents)} stocks")

        groups = constituents.groupby('GICS Sector')

        results: Dict[str, Any] = {
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'lookback_days': self.lookback_days,
            'stocks_per_sector': stocks_per_sector,
            'sectors': {},
            'sharpe_ratios': {},
            'total_stocks': 0,
        }

        for sector_name in self.GICS_SECTORS:
            if sector_name not in groups.groups:
                print(f"  [WARN] Sector '{sector_name}' not found in data")
                continue

            group = groups.get_group(sector_name)
            tickers = group['Symbol'].tolist()
            print(f"\nScanning {sector_name} ({len(tickers)} stocks)...")

            sharpe_scores: List[tuple] = []

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self._calculate_sharpe, t): t for t in tickers}
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        sharpe = future.result()
                    except Exception:
                        sharpe = np.nan
                    if not np.isnan(sharpe):
                        sharpe_scores.append((ticker, round(sharpe, 4)))
                        results['sharpe_ratios'][ticker] = round(sharpe, 4)

            sharpe_scores.sort(key=lambda x: x[1], reverse=True)
            top = [t for t, _ in sharpe_scores[:stocks_per_sector]]

            if top:
                results['sectors'][sector_name] = top
                results['total_stocks'] += len(top)
                for t, s in sharpe_scores[:stocks_per_sector]:
                    print(f"  >> {t}: Sharpe = {s:.2f}")
            else:
                print(f"  [WARN] No valid stocks in {sector_name}")

        print(f"\nScan complete: {results['total_stocks']} stocks selected")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_top_stocks(self, scan_results: Optional[Dict] = None) -> List[str]:
        """Liste plate de tous les tickers selectionnes."""
        if scan_results is None:
            scan_results = self.scan_sectors()
        tickers: List[str] = []
        for stocks in scan_results['sectors'].values():
            tickers.extend(stocks)
        return tickers

    def save_results(self, results: Dict, filepath: Optional[str] = None):
        """Sauvegarde les resultats en JSON."""
        path = Path(filepath) if filepath else self.cache_dir / 'latest_scan.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved: {path}")

    def load_cached_results(self, max_age_days: int = 30) -> Optional[Dict]:
        """Charge les resultats caches si assez recents."""
        path = self.cache_dir / 'latest_scan.json'
        if not path.exists():
            return None

        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age.days > max_age_days:
            print(f"Cache too old ({age.days} days)")
            return None

        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded cached scan ({age.days} days old, {data.get('total_stocks', '?')} stocks)")
        return data


# ======================================================================
# CLI standalone
# ======================================================================
if __name__ == '__main__':
    import argparse

    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='S&P 500 Sector Scanner')
    parser.add_argument('--stocks-per-sector', type=int, default=2)
    parser.add_argument('--lookback-days', type=int, default=252)
    parser.add_argument('--force', action='store_true', help='Ignore cache')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    scanner = SP500Scanner(lookback_days=args.lookback_days)

    if not args.force:
        cached = scanner.load_cached_results(max_age_days=30)
        if cached:
            print("\nUsing cached results. Use --force to re-scan.\n")
            for sector, tickers in cached['sectors'].items():
                sharpes = [f"{t} ({cached['sharpe_ratios'].get(t, '?')})" for t in tickers]
                print(f"  {sector:35s} {', '.join(sharpes)}")
            print(f"\nTotal: {cached['total_stocks']} stocks")
            sys.exit(0)

    results = scanner.scan_sectors(stocks_per_sector=args.stocks_per_sector)
    scanner.save_results(results, filepath=args.output)

    print("\n" + "=" * 70)
    print("SCAN RESULTS")
    print("=" * 70)
    for sector, tickers in results['sectors'].items():
        sharpes = [f"{t} ({results['sharpe_ratios'].get(t, '?')})" for t in tickers]
        print(f"  {sector:35s} {', '.join(sharpes)}")
    print(f"\nTotal: {results['total_stocks']} stocks")
