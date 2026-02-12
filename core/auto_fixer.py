#!/usr/bin/env python3
"""
🔧 AUTO-FIXER

Applique automatiquement les suggestions du Self-Improvement Engine
Modifie la config du bot en temps réel pour améliorer les performances

Auteur: Ploutos AI Team
Date: Dec 2025
"""

from pathlib import Path
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class AutoFixer:
    """Applique automatiquement les améliorations détectées"""
    
    def __init__(self, config_file='config/bot_config.json', backup_dir='config/backups'):
        """
        Args:
            config_file: Fichier de config du bot
            backup_dir: Dossier des backups
        """
        self.config_file = Path(config_file)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.changes_applied = []
        
        logger.info("🔧 Auto-Fixer initialisé")
    
    def _load_config(self) -> Dict:
        """Charger la config actuelle"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Config par défaut
            return {
                'buy_pct': 0.2,
                'min_confidence': 0.5,
                'risk_per_trade': 0.2,
                'stop_loss_pct': None,
                'blacklisted_tickers': [],
                'max_position_size': 0.3
            }
    
    def _backup_config(self):
        """Créer backup de la config"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"config_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"💾 Backup créé: {backup_file}")
        return backup_file
    
    def _save_config(self):
        """Sauvegarder la config modifiée"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"✅ Config sauvegardée: {self.config_file}")
    
    def apply_suggestions(self, suggestions: List[Dict], dry_run=False) -> Dict:
        """
        Appliquer les suggestions d'amélioration
        
        Args:
            suggestions: Liste des suggestions du Self-Improvement Engine
            dry_run: Si True, simule sans appliquer
        
        Returns:
            Dict avec résumé des changements
        """
        if not suggestions:
            logger.info("✅ Aucune suggestion à appliquer")
            return {'status': 'no_changes', 'changes': []}
        
        logger.info(f"🔧 Application de {len(suggestions)} suggestion(s)...")
        
        # Backup avant modifications
        if not dry_run:
            self._backup_config()
        
        changes = []
        
        for sug in suggestions:
            action = sug.get('action')
            
            if action == 'adjust_decision_threshold':
                change = self._adjust_thresholds(sug, dry_run)
                if change:
                    changes.append(change)
            
            elif action == 'reduce_position_size':
                change = self._reduce_position_size(sug, dry_run)
                if change:
                    changes.append(change)
            
            elif action == 'blacklist_ticker':
                change = self._blacklist_ticker(sug, dry_run)
                if change:
                    changes.append(change)
            
            elif action == 'implement_stop_loss':
                change = self._implement_stop_loss(sug, dry_run)
                if change:
                    changes.append(change)
            
            elif action == 'full_retrain':
                changes.append({
                    'action': 'full_retrain',
                    'description': 'Re-entraînement complet recommandé (manuel)',
                    'priority': 'high'
                })
        
        # Sauvegarder si pas dry run
        if not dry_run and changes:
            self._save_config()
            self.changes_applied = changes
        
        result = {
            'status': 'applied' if not dry_run else 'simulated',
            'changes': changes,
            'dry_run': dry_run
        }
        
        logger.info(f"✅ {len(changes)} changement(s) {'simulé(s)' if dry_run else 'appliqué(s)'}")
        
        return result
    
    def _adjust_thresholds(self, suggestion: Dict, dry_run: bool) -> Dict:
        """Ajuster les seuils de décision"""
        hyperparams = suggestion.get('hyperparams', {})
        
        changes = {}
        
        if 'min_confidence' in hyperparams:
            old_val = self.config.get('min_confidence', 0.5)
            new_val = hyperparams['min_confidence']
            
            if new_val != old_val:
                if not dry_run:
                    self.config['min_confidence'] = new_val
                changes['min_confidence'] = {'old': old_val, 'new': new_val}
        
        if 'risk_per_trade' in hyperparams:
            old_val = self.config.get('risk_per_trade', 0.2)
            new_val = hyperparams['risk_per_trade']
            
            if new_val != old_val:
                if not dry_run:
                    self.config['risk_per_trade'] = new_val
                changes['risk_per_trade'] = {'old': old_val, 'new': new_val}
        
        if changes:
            return {
                'action': 'adjust_thresholds',
                'changes': changes,
                'description': suggestion['description']
            }
        return None
    
    def _reduce_position_size(self, suggestion: Dict, dry_run: bool) -> Dict:
        """Réduire la taille des positions"""
        hyperparams = suggestion.get('hyperparams', {})
        
        if 'buy_pct' in hyperparams:
            old_val = self.config.get('buy_pct', 0.2)
            new_val = hyperparams['buy_pct']
            
            if new_val < old_val:
                if not dry_run:
                    self.config['buy_pct'] = new_val
                
                return {
                    'action': 'reduce_position_size',
                    'changes': {'buy_pct': {'old': old_val, 'new': new_val}},
                    'description': suggestion['description']
                }
        
        return None
    
    def _blacklist_ticker(self, suggestion: Dict, dry_run: bool) -> Dict:
        """Blacklister un ticker sous-performant"""
        ticker = suggestion.get('ticker')
        duration_days = suggestion.get('duration_days', 7)
        
        if ticker:
            blacklist = self.config.get('blacklisted_tickers', [])
            
            if ticker not in blacklist:
                if not dry_run:
                    blacklist.append(ticker)
                    self.config['blacklisted_tickers'] = blacklist
                
                return {
                    'action': 'blacklist_ticker',
                    'ticker': ticker,
                    'duration_days': duration_days,
                    'description': suggestion['description']
                }
        
        return None
    
    def _implement_stop_loss(self, suggestion: Dict, dry_run: bool) -> Dict:
        """Implémenter stop-loss"""
        hyperparams = suggestion.get('hyperparams', {})
        
        if 'stop_loss_pct' in hyperparams:
            old_val = self.config.get('stop_loss_pct')
            new_val = hyperparams['stop_loss_pct']
            
            if not dry_run:
                self.config['stop_loss_pct'] = new_val
            
            return {
                'action': 'implement_stop_loss',
                'changes': {'stop_loss_pct': {'old': old_val, 'new': new_val}},
                'description': suggestion['description']
            }
        
        return None
    
    def get_current_config(self) -> Dict:
        """Obtenir la config actuelle"""
        return self.config.copy()
    
    def rollback(self) -> bool:
        """Restaurer le dernier backup"""
        backups = sorted(self.backup_dir.glob('config_*.json'), reverse=True)
        
        if not backups:
            logger.warning("⚠️  Aucun backup trouvé")
            return False
        
        latest_backup = backups[0]
        
        try:
            shutil.copy(latest_backup, self.config_file)
            self.config = self._load_config()
            logger.info(f"✅ Rollback depuis: {latest_backup.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur rollback: {e}")
            return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 === AUTO-FIXER TEST ===\n")
    
    fixer = AutoFixer()
    
    # Exemple de suggestions
    test_suggestions = [
        {
            'action': 'adjust_decision_threshold',
            'description': 'Augmenter seuil de confiance',
            'hyperparams': {'min_confidence': 0.6, 'risk_per_trade': 0.15}
        },
        {
            'action': 'reduce_position_size',
            'description': 'Réduire taille positions',
            'hyperparams': {'buy_pct': 0.15}
        }
    ]
    
    # Dry run
    print("💡 Simulation (dry run):\n")
    result = fixer.apply_suggestions(test_suggestions, dry_run=True)
    
    for change in result['changes']:
        print(f"  • {change['action']}: {change['description']}")
        if 'changes' in change:
            for key, vals in change['changes'].items():
                print(f"    - {key}: {vals['old']} → {vals['new']}")
    
    print("\n✅ Test terminé\n")
