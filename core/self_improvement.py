#!/usr/bin/env python3
"""
🧠 SELF-IMPROVEMENT SYSTEM

L'IA qui s'analyse et s'améliore automatiquement !

Fonctionnalités:
1. Analyse des trades perdants
2. Détection des patterns d'erreurs
3. Suggestions d'améliorations
4. Re-entraînement automatique sur cas difficiles
5. A/B testing des versions

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SelfImprovementEngine:
    """Moteur d'auto-amélioration pour le bot de trading"""
    
    # Seuils de détection
    MIN_TRADES = 20  # Minimum de trades pour analyser
    BAD_WIN_RATE = 0.45  # Win rate < 45% = problème
    BAD_SHARPE = 0.3  # Sharpe < 0.3 = problème
    MAX_DRAWDOWN = 0.15  # Drawdown > 15% = problème
    
    def __init__(self, trades_log_dir='logs/trades', models_dir='models'):
        """
        Args:
            trades_log_dir: Dossier contenant les logs JSON des trades
            models_dir: Dossier des modèles sauvegardés
        """
        self.trades_log_dir = Path(trades_log_dir)
        self.models_dir = Path(models_dir)
        
        self.current_metrics = {}
        self.issues_detected = []
        self.suggestions = []
        
        logger.info("🧠 Self-Improvement Engine initialisé")
    
    def analyze_recent_performance(self, days=7) -> Dict:
        """
        Analyser les performances récentes
        
        Args:
            days: Nombre de jours à analyser
        
        Returns:
            Dict avec métriques et diagnostics
        """
        logger.info(f"📊 Analyse des {days} derniers jours...")
        
        # Charger les trades
        trades = self._load_recent_trades(days)
        
        if len(trades) < self.MIN_TRADES:
            logger.warning(f"⚠️  Pas assez de trades: {len(trades)}/{self.MIN_TRADES}")
            return {'status': 'insufficient_data', 'trades_count': len(trades)}
        
        # Calculer métriques
        metrics = self._calculate_metrics(trades)
        self.current_metrics = metrics
        
        # Détecter problèmes
        issues = self._detect_issues(metrics, trades)
        self.issues_detected = issues
        
        # Générer suggestions
        suggestions = self._generate_suggestions(issues, metrics, trades)
        self.suggestions = suggestions
        
        # Résultat
        result = {
            'status': 'analyzed',
            'trades_count': len(trades),
            'metrics': metrics,
            'issues': issues,
            'suggestions': suggestions,
            'health_score': self._calculate_health_score(metrics)
        }
        
        logger.info(f"✅ Analyse terminée: {len(issues)} problème(s) détecté(s)")
        
        return result
    
    def _load_recent_trades(self, days: int) -> List[Dict]:
        """Charger les trades des N derniers jours"""
        trades = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            filename = self.trades_log_dir / f"trades_{date.strftime('%Y-%m-%d')}.json"
            
            if filename.exists():
                try:
                    with open(filename, 'r') as f:
                        daily_trades = json.load(f)
                        trades.extend(daily_trades)
                except Exception as e:
                    logger.warning(f"⚠️  Erreur lecture {filename}: {e}")
        
        # Filtrer pour avoir des paires BUY/SELL complètes
        processed_trades = self._match_buy_sell_pairs(trades)
        
        return processed_trades
    
    def _match_buy_sell_pairs(self, trades: List[Dict]) -> List[Dict]:
        """Matcher les BUY avec leurs SELL correspondants"""
        matched = []
        positions = defaultdict(list)  # ticker -> [buy_orders]
        
        for trade in sorted(trades, key=lambda t: t['timestamp']):
            ticker = trade['symbol']
            action = trade['action']
            
            if action == 'BUY':
                positions[ticker].append(trade)
            
            elif action == 'SELL' and positions[ticker]:
                # Prendre le premier BUY (FIFO)
                buy_trade = positions[ticker].pop(0)
                
                # Calculer PnL
                buy_price = buy_trade['price']
                sell_price = trade['price']
                pnl = (sell_price - buy_price) / buy_price
                
                matched.append({
                    'ticker': ticker,
                    'buy_time': buy_trade['timestamp'],
                    'sell_time': trade['timestamp'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'pnl': pnl,
                    'amount': trade['amount'],
                    'holding_duration': self._calculate_duration(
                        buy_trade['timestamp'], 
                        trade['timestamp']
                    )
                })
        
        return matched
    
    def _calculate_duration(self, start: str, end: str) -> float:
        """Calculer durée en heures"""
        try:
            t1 = datetime.fromisoformat(start)
            t2 = datetime.fromisoformat(end)
            return (t2 - t1).total_seconds() / 3600
        except:
            return 0
    
    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculer métriques de performance"""
        if not trades:
            return {}
        
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Win rate
        win_rate = len(wins) / len(pnls) if pnls else 0
        
        # Average win/loss
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplifié)
        sharpe = (np.mean(pnls) / np.std(pnls)) if len(pnls) > 1 else 0
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Par ticker
        by_ticker = defaultdict(list)
        for t in trades:
            by_ticker[t['ticker']].append(t['pnl'])
        
        ticker_stats = {
            ticker: {
                'trades': len(pnls_list),
                'win_rate': sum(1 for p in pnls_list if p > 0) / len(pnls_list),
                'avg_pnl': np.mean(pnls_list)
            }
            for ticker, pnls_list in by_ticker.items()
        }
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_pnl': sum(pnls),
            'by_ticker': ticker_stats,
            'avg_holding_hours': np.mean([t['holding_duration'] for t in trades])
        }
    
    def _detect_issues(self, metrics: Dict, trades: List[Dict]) -> List[Dict]:
        """Détecter les problèmes de performance"""
        issues = []
        
        # 1. Win rate faible
        if metrics['win_rate'] < self.BAD_WIN_RATE:
            issues.append({
                'type': 'low_win_rate',
                'severity': 'high',
                'value': metrics['win_rate'],
                'description': f"Win rate trop bas: {metrics['win_rate']:.1%}"
            })
        
        # 2. Sharpe faible
        if metrics['sharpe_ratio'] < self.BAD_SHARPE:
            issues.append({
                'type': 'low_sharpe',
                'severity': 'medium',
                'value': metrics['sharpe_ratio'],
                'description': f"Sharpe ratio faible: {metrics['sharpe_ratio']:.2f}"
            })
        
        # 3. Drawdown élevé
        if metrics['max_drawdown'] > self.MAX_DRAWDOWN:
            issues.append({
                'type': 'high_drawdown',
                'severity': 'high',
                'value': metrics['max_drawdown'],
                'description': f"Drawdown excessif: {metrics['max_drawdown']:.1%}"
            })
        
        # 4. Tickers sous-performants
        for ticker, stats in metrics['by_ticker'].items():
            if stats['win_rate'] < 0.4 and stats['trades'] >= 5:
                issues.append({
                    'type': 'bad_ticker',
                    'severity': 'medium',
                    'ticker': ticker,
                    'value': stats['win_rate'],
                    'description': f"{ticker} sous-performe: {stats['win_rate']:.1%} win rate"
                })
        
        # 5. Avg loss > avg win (mauvais risk/reward)
        if metrics['avg_loss'] > metrics['avg_win'] * 1.5:
            issues.append({
                'type': 'bad_risk_reward',
                'severity': 'high',
                'description': f"Pertes moyennes trop élevées: {metrics['avg_loss']:.2%} vs gains {metrics['avg_win']:.2%}"
            })
        
        return issues
    
    def _generate_suggestions(self, issues: List[Dict], metrics: Dict, trades: List[Dict]) -> List[Dict]:
        """Générer suggestions d'amélioration"""
        suggestions = []
        
        for issue in issues:
            if issue['type'] == 'low_win_rate':
                suggestions.append({
                    'issue': issue['type'],
                    'action': 'adjust_decision_threshold',
                    'description': "Augmenter le seuil de confiance pour prendre moins de trades mais meilleurs",
                    'hyperparams': {'min_confidence': 0.6, 'risk_per_trade': 0.15}
                })
                
                suggestions.append({
                    'issue': issue['type'],
                    'action': 'retrain_with_hard_negatives',
                    'description': "Re-entraîner sur les trades perdants pour apprendre de ses erreurs",
                    'data': 'losing_trades'
                })
            
            elif issue['type'] == 'high_drawdown':
                suggestions.append({
                    'issue': issue['type'],
                    'action': 'reduce_position_size',
                    'description': "Réduire la taille des positions pour limiter le risque",
                    'hyperparams': {'buy_pct': 0.15}  # Au lieu de 0.2
                })
            
            elif issue['type'] == 'bad_ticker':
                suggestions.append({
                    'issue': issue['type'],
                    'action': 'blacklist_ticker',
                    'description': f"Blacklister temporairement {issue['ticker']}",
                    'ticker': issue['ticker'],
                    'duration_days': 7
                })
            
            elif issue['type'] == 'bad_risk_reward':
                suggestions.append({
                    'issue': issue['type'],
                    'action': 'implement_stop_loss',
                    'description': "Implémenter stop-loss plus strict",
                    'hyperparams': {'stop_loss_pct': 0.05}
                })
        
        # Suggestion générale: Re-entraînement si trop de problèmes
        if len(issues) >= 3:
            suggestions.append({
                'issue': 'multiple_issues',
                'action': 'full_retrain',
                'description': "Re-entraînement complet du modèle recommandé",
                'priority': 'high'
            })
        
        return suggestions
    
    def _calculate_health_score(self, metrics: Dict) -> float:
        """Score de santé global (0-100)"""
        if not metrics:
            return 0
        
        score = 0
        
        # Win rate (40 points)
        score += min(metrics['win_rate'] * 80, 40)
        
        # Sharpe (30 points)
        score += min(metrics['sharpe_ratio'] * 30, 30)
        
        # Drawdown (20 points)
        dd_score = max(0, 20 - metrics['max_drawdown'] * 100)
        score += dd_score
        
        # Profit factor (10 points)
        pf_score = min(metrics['profit_factor'] * 5, 10)
        score += pf_score
        
        return min(score, 100)
    
    def export_report(self, output_file='logs/self_improvement_report.json'):
        """Exporter rapport complet"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.current_metrics,
            'issues': self.issues_detected,
            'suggestions': self.suggestions,
            'health_score': self._calculate_health_score(self.current_metrics)
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Rapport exporté: {output_file}")
        
        return report


# ★ SCRIPT STANDALONE
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 === SELF-IMPROVEMENT ANALYSIS ===\n")
    
    engine = SelfImprovementEngine()
    result = engine.analyze_recent_performance(days=7)
    
    if result['status'] == 'insufficient_data':
        print(f"⚠️  Pas assez de données: {result['trades_count']} trades")
    else:
        print(f"📊 Trades analysés: {result['trades_count']}")
        print(f"🏥 Health Score: {result['health_score']:.1f}/100\n")
        
        print("📈 MÉTRIQUES:")
        m = result['metrics']
        print(f"  • Win Rate: {m['win_rate']:.1%}")
        print(f"  • Sharpe: {m['sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {m['max_drawdown']:.1%}")
        print(f"  • Profit Factor: {m['profit_factor']:.2f}\n")
        
        if result['issues']:
            print(f"⚠️  PROBLÈMES DÉTECTÉS ({len(result['issues'])})")
            for issue in result['issues']:
                print(f"  • [{issue['severity'].upper()}] {issue['description']}")
            print()
        
        if result['suggestions']:
            print(f"💡 SUGGESTIONS ({len(result['suggestions'])})")
            for sug in result['suggestions']:
                print(f"  • {sug['description']}")
            print()
        
        # Export
        engine.export_report()
        print("✅ Rapport complet exporté\n")
