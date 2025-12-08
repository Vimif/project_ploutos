#!/usr/bin/env python3
"""
🧠 SELF-IMPROVEMENT ORCHESTRATOR

Script complet d'auto-amélioration :
1. Analyse les performances
2. Détecte les problèmes
3. Applique les corrections automatiquement
4. Notifie via Discord
5. Export rapport

Usage:
  python3 scripts/run_self_improvement.py              # Analyse + application
  python3 scripts/run_self_improvement.py --dry-run     # Simulation
  python3 scripts/run_self_improvement.py --days 14     # Analyser 14 jours

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

# Import des modules
from core.self_improvement import SelfImprovementEngine
from core.auto_fixer import AutoFixer

try:
    from notifications.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def notify_improvement_report(discord: 'DiscordNotifier', result: dict, changes: dict, dry_run: bool):
    """Envoyer rapport d'amélioration sur Discord"""
    if not discord or not discord.enabled:
        return
    
    health_score = result.get('health_score', 0)
    metrics = result.get('metrics', {})
    issues = result.get('issues', [])
    
    # Couleur selon health score
    if health_score >= 70:
        color = 3066993  # Vert
    elif health_score >= 50:
        color = 16776960  # Jaune
    else:
        color = 15158332  # Rouge
    
    embed = {
        'title': '🧠 Rapport d\'Auto-Amélioration',
        'color': color,
        'timestamp': datetime.utcnow().isoformat(),
        'fields': []
    }
    
    # Health Score
    emoji = '🟢' if health_score >= 70 else ('🟡' if health_score >= 50 else '🔴')
    embed['fields'].append({
        'name': f'{emoji} Health Score',
        'value': f"**{health_score:.1f}/100**",
        'inline': True
    })
    
    # Trades analysés
    embed['fields'].append({
        'name': '📊 Trades Analysés',
        'value': f"{result['trades_count']} trades",
        'inline': True
    })
    
    # Métriques clés
    if metrics:
        metrics_text = f"**Win Rate:** {metrics['win_rate']:.1%}\n"
        metrics_text += f"**Sharpe:** {metrics['sharpe_ratio']:.2f}\n"
        metrics_text += f"**Max DD:** {metrics['max_drawdown']:.1%}"
        
        embed['fields'].append({
            'name': '📈 Métriques',
            'value': metrics_text,
            'inline': False
        })
    
    # Problèmes détectés
    if issues:
        issues_text = "\n".join([
            f"• [{i['severity'].upper()}] {i['description']}"
            for i in issues[:5]  # Max 5
        ])
        
        embed['fields'].append({
            'name': f'⚠️  Problèmes ({len(issues)})',
            'value': issues_text,
            'inline': False
        })
    
    # Changements appliqués
    if changes.get('changes'):
        changes_text = "\n".join([
            f"• {c.get('action', 'unknown')}: {c.get('description', '')[:100]}"
            for c in changes['changes'][:5]
        ])
        
        status = '💡 Simulés' if dry_run else '✅ Appliqués'
        
        embed['fields'].append({
            'name': f'{status} ({len(changes["changes"])})',
            'value': changes_text or 'Aucun changement',
            'inline': False
        })
    
    embed['footer'] = {'text': 'Ploutos Self-Improvement System'}
    
    discord.send_message(embed=embed)


def main():
    parser = argparse.ArgumentParser(description='Self-Improvement System')
    parser.add_argument('--days', type=int, default=7, help='Jours à analyser (défaut: 7)')
    parser.add_argument('--dry-run', action='store_true', help='Simulation sans appliquer')
    parser.add_argument('--no-discord', action='store_true', help='Désactiver Discord')
    parser.add_argument('--auto-apply', action='store_true', help='Appliquer automatiquement les fixes')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧠 PLOUTOS SELF-IMPROVEMENT SYSTEM")
    print("="*70 + "\n")
    
    # 1. ANALYSE
    print(f"🔍 Étape 1/3: Analyse des {args.days} derniers jours...\n")
    
    engine = SelfImprovementEngine()
    result = engine.analyze_recent_performance(days=args.days)
    
    if result['status'] == 'insufficient_data':
        print(f"\u26a0️  Pas assez de données: {result['trades_count']} trades")
        print("   Minimum requis: 20 trades\n")
        return
    
    # Afficher résultats
    print(f"✅ {result['trades_count']} trades analysés")
    print(f"🏥 Health Score: {result['health_score']:.1f}/100\n")
    
    metrics = result['metrics']
    print("📈 MÉTRIQUES CLÉS:")
    print(f"  • Win Rate: {metrics['win_rate']:.1%}")
    print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  • Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"  • Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  • Total PnL: {metrics['total_pnl']:.2%}\n")
    
    # Problèmes
    issues = result['issues']
    if issues:
        print(f"⚠️  PROBLÈMES DÉTECTÉS ({len(issues)}):")
        for issue in issues:
            print(f"  • [{issue['severity'].upper()}] {issue['description']}")
        print()
    else:
        print("✅ Aucun problème détecté !\n")
    
    # Suggestions
    suggestions = result['suggestions']
    if suggestions:
        print(f"💡 SUGGESTIONS ({len(suggestions)}):")
        for sug in suggestions:
            print(f"  • {sug['description']}")
        print()
    
    # 2. APPLICATION DES FIXES
    print(f"🔧 Étape 2/3: Application des corrections...\n")
    
    fixer = AutoFixer()
    
    if not suggestions:
        print("✅ Aucune correction nécessaire\n")
        changes = {'status': 'no_changes', 'changes': []}
    else:
        # Appliquer (avec ou sans dry-run)
        apply_changes = args.auto_apply or (not args.dry_run)
        
        changes = fixer.apply_suggestions(suggestions, dry_run=(not apply_changes))
        
        if changes['changes']:
            status = "SIMULÉS" if not apply_changes else "APPLIQUÉS"
            print(f"✅ {len(changes['changes'])} changement(s) {status}:\n")
            
            for change in changes['changes']:
                print(f"  ✅ {change['action']}: {change.get('description', '')}")
                
                if 'changes' in change:
                    for key, vals in change['changes'].items():
                        print(f"     - {key}: {vals['old']} → {vals['new']}")
            print()
        else:
            print("ℹ️  Aucun changement applicable\n")
    
    # 3. EXPORT & NOTIFICATION
    print("📤 Étape 3/3: Export et notifications...\n")
    
    # Export rapport JSON
    engine.export_report()
    print("✅ Rapport exporté: logs/self_improvement_report.json")
    
    # Discord
    if DISCORD_AVAILABLE and not args.no_discord:
        discord = DiscordNotifier()
        if discord.enabled:
            notify_improvement_report(discord, result, changes, args.dry_run)
            print("✅ Notification Discord envoyée")
    
    print("\n" + "="*70)
    print("✅ AUTO-AMÉLIORATION TERMINÉE")
    print("="*70 + "\n")
    
    # Résumé final
    if result['health_score'] < 50:
        print("⚠️  ATTENTION: Health Score faible !")
        print("   Considérer un re-entraînement complet du modèle.\n")
    elif result['health_score'] < 70:
        print("🟡 Santé moyenne. Continuez à surveiller.\n")
    else:
        print("🟢 Excellent état ! Continuez comme ça.\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        sys.exit(1)
