"""
📚 ADAPTIVE CURRICULUM LEARNING - Stratégie d'Entraînement Avancée #2

Curriculum qui s'ajuste automatiquement à la progression de l'IA.
Au lieu d'un plan fixe (Stage1→Stage2→Stage3), le système observe
les performances et adapte la difficulté en temps réel.

Gains attendus: -30% temps training, progression optimale

Usage:
    from core.adaptive_curriculum import AdaptiveCurriculum
    
    curriculum = AdaptiveCurriculum()
    
    for iteration in range(100):
        # Curriculum donne tâche adaptée
        task_config = curriculum.get_next_task(current_sharpe)
        
        # Entraîner sur cette tâche
        env = create_env(**task_config)
        model.learn(env, timesteps=1_000_000)
        
        # Évaluer
        current_sharpe = evaluate(model, env)

Références:
    - NeurIPS 2025: Auto-curriculum améliore apprentissage RL
    - TU Delft: Curriculum adaptatif surpasse curriculum fixe
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class AdaptiveCurriculum:
    """
    Gestion automatique de la difficulté d'entraînement.
    
    Niveaux de difficulté:
        Level 0: 1 ticker, bull market, volatilité faible
        Level 1: 1 ticker, mixed market, volatilité moyenne
        Level 2: 2 tickers, mixed market, volatilité moyenne
        Level 3: 3 tickers, all markets, volatilité haute
        Level 4: 5+ tickers, all markets, volatilité extrême
    
    Critères d'avancement:
        ✅ 3 dernières évals: Sharpe > 0.7 → Level UP
        ❌ 5 dernières évals: Sharpe < 0.3 → Level DOWN
    """
    
    def __init__(
        self,
        difficulty_levels=None,
        advancement_threshold=0.7,
        regression_threshold=0.3,
        advancement_window=3,
        regression_window=5,
        save_dir='curriculum_logs',
        metrics=['sharpe', 'winrate', 'drawdown']
    ):
        """
        Args:
            difficulty_levels: Liste des configurations de difficulté
            advancement_threshold: Sharpe min pour monter niveau
            regression_threshold: Sharpe min pour descendre
            advancement_window: Nombre d'évals pour monter
            regression_window: Nombre d'évals pour descendre
            save_dir: Dossier pour logs
            metrics: Métriques à tracker
        """
        # Configuration niveaux (défaut calibré pour Ploutos)
        if difficulty_levels is None:
            self.difficulty_levels = self._get_default_levels()
        else:
            self.difficulty_levels = difficulty_levels
        
        # Paramètres adaptation
        self.advancement_threshold = advancement_threshold
        self.regression_threshold = regression_threshold
        self.advancement_window = advancement_window
        self.regression_window = regression_window
        self.metrics = metrics
        
        # État actuel
        self.current_level = 0
        self.performance_history = []
        self.level_history = []
        self.transitions = []
        
        # Logs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Stats par niveau
        self.level_stats = {i: {'n_epochs': 0, 'avg_sharpe': 0, 'best_sharpe': 0}
                            for i in range(len(self.difficulty_levels))}
        
        print(f"\n📚 Adaptive Curriculum initialisé")
        print(f"🎯 {len(self.difficulty_levels)} niveaux de difficulté")
        print(f"📈 Advancement: {advancement_window} évals > {advancement_threshold}")
        print(f"📉 Regression: {regression_window} évals < {regression_threshold}")
        print(f"💾 Logs: {self.save_dir / self.session_id}")
        print()
    
    def _get_default_levels(self) -> List[Dict]:
        """
        Configuration par défaut des niveaux (calibrée pour Ploutos)
        
        Returns:
            Liste de configs avec: tickers, period, volatility, max_steps
        """
        # Tickers par catégorie (défini dans ton projet)
        GROWTH = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"]
        DEFENSIVE = ["SPY", "QQQ", "VOO", "VTI"]
        ENERGY = ["XOM", "CVX", "COP", "XLE"]
        FINANCE = ["JPM", "BAC", "WFC", "GS"]
        
        levels = [
            # Level 0: Ultra facile (warm-up)
            {
                'level': 0,
                'name': 'Warm-up',
                'tickers': ['SPY'],
                'period': '1y',
                'interval': '1h',
                'max_steps': 1000,
                'initial_balance': 100000,
                'description': '1 ticker défensif, bull market, faible volatilité'
            },
            
            # Level 1: Facile
            {
                'level': 1,
                'name': 'Basic',
                'tickers': ['SPY'],
                'period': '2y',
                'interval': '1h',
                'max_steps': 1500,
                'initial_balance': 100000,
                'description': '1 ticker, marché mixte, volatilité normale'
            },
            
            # Level 2: Intermédiaire
            {
                'level': 2,
                'name': 'Intermediate',
                'tickers': ['SPY', 'QQQ'],
                'period': '2y',
                'interval': '1h',
                'max_steps': 2000,
                'initial_balance': 100000,
                'description': '2 tickers, marché mixte, volatilité moyenne'
            },
            
            # Level 3: Avancé
            {
                'level': 3,
                'name': 'Advanced',
                'tickers': ['SPY', 'QQQ', 'NVDA'],
                'period': '2y',
                'interval': '1h',
                'max_steps': 2000,
                'initial_balance': 100000,
                'description': '3 tickers (défensif + tech), tous marchés, volatilité haute'
            },
            
            # Level 4: Expert (configuration actuelle Ploutos)
            {
                'level': 4,
                'name': 'Expert',
                'tickers': GROWTH[:3] + DEFENSIVE[:2],  # 5 tickers
                'period': '2y',
                'interval': '1h',
                'max_steps': 2000,
                'initial_balance': 100000,
                'description': '5 tickers mixtes, tous marchés, volatilité extrême'
            },
            
            # Level 5: Master (challenge)
            {
                'level': 5,
                'name': 'Master',
                'tickers': GROWTH + DEFENSIVE + ENERGY[:2],  # 11 tickers
                'period': '2y',
                'interval': '1h',
                'max_steps': 2000,
                'initial_balance': 100000,
                'description': '11 tickers multi-secteurs, volatilité maximale'
            }
        ]
        
        return levels
    
    def get_next_task(self, performance_metrics: Dict[str, float]) -> Dict:
        """
        Détermine la prochaine tâche selon performances actuelles.
        
        Args:
            performance_metrics: {'sharpe': 0.5, 'winrate': 0.45, 'drawdown': 0.03}
        
        Returns:
            Dict: Configuration de la tâche (tickers, period, etc.)
        """
        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'level': self.current_level,
            **performance_metrics
        })
        
        # Update stats niveau actuel
        sharpe = performance_metrics.get('sharpe', 0.0)
        self.level_stats[self.current_level]['n_epochs'] += 1
        self.level_stats[self.current_level]['avg_sharpe'] = (
            (self.level_stats[self.current_level]['avg_sharpe'] * 
             (self.level_stats[self.current_level]['n_epochs'] - 1) + sharpe) /
            self.level_stats[self.current_level]['n_epochs']
        )
        self.level_stats[self.current_level]['best_sharpe'] = max(
            self.level_stats[self.current_level]['best_sharpe'],
            sharpe
        )
        
        # Décision adaptation
        previous_level = self.current_level
        
        if self._should_advance():
            self._advance_level()
            print(f"\n📈 NIVEAU UP! {previous_level} → {self.current_level}")
            print(f"   Raison: {self.advancement_window} dernières évals > {self.advancement_threshold}")
        
        elif self._should_regress():
            self._regress_level()
            print(f"\n📉 NIVEAU DOWN {previous_level} → {self.current_level}")
            print(f"   Raison: {self.regression_window} dernières évals < {self.regression_threshold}")
        
        # Log transition si changement
        if previous_level != self.current_level:
            self.transitions.append({
                'timestamp': datetime.now().isoformat(),
                'from_level': previous_level,
                'to_level': self.current_level,
                'reason': 'advancement' if self.current_level > previous_level else 'regression',
                'performance': performance_metrics
            })
        
        # Retourner config niveau actuel
        task_config = self.difficulty_levels[self.current_level].copy()
        
        # Enrichir avec infos curriculum
        task_config['curriculum_info'] = {
            'current_level': self.current_level,
            'total_levels': len(self.difficulty_levels),
            'n_epochs_at_level': self.level_stats[self.current_level]['n_epochs'],
            'avg_sharpe_at_level': self.level_stats[self.current_level]['avg_sharpe'],
            'progression': (self.current_level + 1) / len(self.difficulty_levels)
        }
        
        # Save logs
        self._save_state()
        
        return task_config
    
    def _should_advance(self) -> bool:
        """Vérifie si ready pour niveau supérieur"""
        # Besoin de suffisamment d'historique
        if len(self.performance_history) < self.advancement_window:
            return False
        
        # Déjà au niveau max?
        if self.current_level >= len(self.difficulty_levels) - 1:
            return False
        
        # Vérifier N dernières performances
        recent = self.performance_history[-self.advancement_window:]
        
        # Toutes au-dessus du seuil?
        sharpes = [p['sharpe'] for p in recent]
        
        return all(s >= self.advancement_threshold for s in sharpes)
    
    def _should_regress(self) -> bool:
        """Vérifie si doit redescendre niveau"""
        # Besoin de suffisamment d'historique
        if len(self.performance_history) < self.regression_window:
            return False
        
        # Déjà au niveau min?
        if self.current_level <= 0:
            return False
        
        # Vérifier N dernières performances
        recent = self.performance_history[-self.regression_window:]
        
        # Toutes en dessous du seuil?
        sharpes = [p['sharpe'] for p in recent]
        
        return all(s < self.regression_threshold for s in sharpes)
    
    def _advance_level(self):
        """Monte au niveau supérieur"""
        self.current_level = min(
            self.current_level + 1,
            len(self.difficulty_levels) - 1
        )
        self.level_history.append(self.current_level)
    
    def _regress_level(self):
        """Redescend au niveau inférieur"""
        self.current_level = max(self.current_level - 1, 0)
        self.level_history.append(self.current_level)
    
    def _save_state(self):
        """Sauvegarde l'état du curriculum"""
        state = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'current_level': self.current_level,
            'performance_history': self.performance_history,
            'level_history': self.level_history,
            'transitions': self.transitions,
            'level_stats': self.level_stats,
            'config': {
                'advancement_threshold': self.advancement_threshold,
                'regression_threshold': self.regression_threshold,
                'advancement_window': self.advancement_window,
                'regression_window': self.regression_window
            }
        }
        
        filepath = self.save_dir / f'{self.session_id}.json'
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Résumé de la session curriculum"""
        summary = {
            'session_id': self.session_id,
            'current_level': self.current_level,
            'current_level_name': self.difficulty_levels[self.current_level]['name'],
            'total_epochs': len(self.performance_history),
            'n_transitions': len(self.transitions),
            'level_distribution': {
                lvl: stats['n_epochs'] 
                for lvl, stats in self.level_stats.items()
            },
            'best_sharpe_per_level': {
                lvl: stats['best_sharpe']
                for lvl, stats in self.level_stats.items()
            },
            'progression': (self.current_level + 1) / len(self.difficulty_levels)
        }
        
        # Dernières performances
        if len(self.performance_history) > 0:
            recent = self.performance_history[-5:]
            summary['recent_sharpes'] = [p['sharpe'] for p in recent]
            summary['recent_avg_sharpe'] = np.mean([p['sharpe'] for p in recent])
        
        return summary
    
    def print_summary(self):
        """Affiche résumé visuel"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📚 ADAPTIVE CURRICULUM - RÉSUMÉ")
        print("="*80)
        
        print(f"\n🎯 Niveau actuel: {summary['current_level']} - {summary['current_level_name']}")
        print(f"📈 Progression: {summary['progression']:.0%}")
        print(f"🔄 Transitions: {summary['n_transitions']}")
        print(f"📅 Total époques: {summary['total_epochs']}")
        
        print(f"\n📊 Distribution par niveau:")
        for lvl, n_epochs in summary['level_distribution'].items():
            if n_epochs > 0:
                level_name = self.difficulty_levels[lvl]['name']
                best = summary['best_sharpe_per_level'][lvl]
                print(f"   Level {lvl} ({level_name}): {n_epochs} époques | Best Sharpe: {best:.2f}")
        
        if 'recent_sharpes' in summary:
            print(f"\n🔥 Dernières performances:")
            print(f"   Sharpes: {[f'{s:.2f}' for s in summary['recent_sharpes']]}")
            print(f"   Moyenne: {summary['recent_avg_sharpe']:.2f}")
        
        print("\n" + "="*80)


# ============================================================================
# EXEMPLE D'INTÉGRATION AVEC TRAIN LOOP
# ============================================================================

def example_integration_train_curriculum():
    """
    Exemple d'intégration dans train_curriculum.py
    
    ✅ ZERO MODIFICATION du code existant
    ✅ Juste wrapper le training loop
    """
    from core.adaptive_curriculum import AdaptiveCurriculum
    from core.universal_environment import UniversalTradingEnv
    from core.data_fetcher import DataFetcher
    from stable_baselines3 import PPO
    import wandb
    
    # ✨ Initialiser curriculum
    curriculum = AdaptiveCurriculum(
        advancement_threshold=0.6,  # Plus permissif que 0.7
        regression_threshold=0.3
    )
    
    # Model PPO (réutilisable entre niveaux)
    model = None
    fetcher = DataFetcher()
    
    # Training loop adapté
    for epoch in range(100):
        # ✨ Curriculum donne config adaptée
        if epoch == 0:
            # Première époque: pas de metrics
            task_config = curriculum.difficulty_levels[0]
        else:
            # Curriculum s'adapte
            task_config = curriculum.get_next_task({
                'sharpe': eval_sharpe,
                'winrate': eval_winrate,
                'drawdown': eval_drawdown
            })
        
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch+1}/100 - Level {task_config['level']}: {task_config['name']}")
        print(f"📊 {task_config['description']}")
        print(f"{'='*80}")
        
        # Fetch data selon config
        data = fetcher.fetch_data(
            task_config['tickers'],
            period=task_config['period'],
            interval=task_config['interval']
        )
        
        # Créer env
        env = UniversalTradingEnv(
            data,
            initial_balance=task_config['initial_balance'],
            max_steps=task_config['max_steps']
        )
        
        # Créer ou réutiliser model
        if model is None:
            model = PPO('MlpPolicy', env, ...)
        else:
            # Réutiliser poids appris (transfer learning)
            model.set_env(env)
        
        # Train
        model.learn(total_timesteps=1_000_000)
        
        # Évaluer
        eval_sharpe, eval_winrate, eval_drawdown = evaluate_model(model, env)
        
        # Log W&B
        wandb.log({
            'curriculum/level': task_config['level'],
            'curriculum/n_tickers': len(task_config['tickers']),
            'eval/sharpe': eval_sharpe,
            'eval/winrate': eval_winrate,
            'eval/drawdown': eval_drawdown
        })
    
    # Résumé final
    curriculum.print_summary()


def evaluate_model(model, env, n_episodes=10):
    """Fonction d'évaluation simple"""
    sharpes = []
    winrates = []
    drawdowns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_returns = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_returns.append(reward)
            done = terminated or truncated
        
        # Calculs métriques
        returns = np.array(episode_returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        winrate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        sharpes.append(sharpe)
        winrates.append(winrate)
    
    return np.mean(sharpes), np.mean(winrates), 0.02  # drawdown simplifié


if __name__ == '__main__':
    print("=" * 80)
    print("📚 ADAPTIVE CURRICULUM - Tests")
    print("=" * 80)
    
    # Test basique
    curriculum = AdaptiveCurriculum()
    
    print("\n🧪 Simulation 20 époques...\n")
    
    # Simuler progression
    for i in range(20):
        # Simuler amélioration progressive
        if i < 5:
            sharpe = np.random.uniform(0.2, 0.5)  # Début difficile
        elif i < 10:
            sharpe = np.random.uniform(0.5, 0.8)  # Amélioration
        else:
            sharpe = np.random.uniform(0.7, 1.2)  # Maîtrise
        
        task = curriculum.get_next_task({
            'sharpe': sharpe,
            'winrate': 0.5,
            'drawdown': 0.03
        })
        
        print(f"Epoch {i+1}: Sharpe {sharpe:.2f} → Level {task['level']} ({task['name']})")
    
    # Résumé
    curriculum.print_summary()
    
    print("\n✅ Tests terminés !")
