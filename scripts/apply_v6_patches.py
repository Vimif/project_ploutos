#!/usr/bin/env python3
"""
Apply V6 Advanced Patches - Automatic Integration Script
========================================================

Ce script applique automatiquement tous les patchs V6 au fichier d'environnement.
Il :
1. Sauvegarde l'original
2. Ajoute les imports
3. Intègre les 3 nouveaux modules
4. Met à jour les méthodes

Usage:
    python scripts/apply_v6_patches.py
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
ENV_FILE = "core/universal_environment_v6_better_timing.py"
BACKUP_FILE = f"core/universal_environment_v6_better_timing.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Templates des patchs
PATCH_IMPORTS = '''
from core.observation_builder_v7 import ObservationBuilderV7
from core.reward_calculator_advanced import DifferentialSharpeRewardCalculator
from core.normalization import AdaptiveNormalizer
'''

PATCH_INIT_ADDITIONS = '''
        # --- PATCH V6 ACTIVATION ---
        
        # 1. Calculateur de Reward Avancé (Differential Sharpe)
        self.reward_calc = DifferentialSharpeRewardCalculator(
            decay=0.99,
            window=252,
            dsr_weight=0.6,
            sortino_weight=0.2,
            win_rate_weight=0.1,
            risk_weight=0.05,
            trade_penalty_weight=0.05,
        )
        print("  ✅ V6 Reward Calculator initialized")
        
        # 2. Normalizer (pour nettoyer les données)
        self.normalizer = AdaptiveNormalizer()
        print("  ✅ V6 Normalizer initialized")
        
        # 3. Constructeur d'Observation 3D (pour le Transformer)
        self.obs_builder = None  # Sera initialisé dans _prepare_features_v2
        print("  ✅ V6 Observation Builder ready")
'''

PATCH_PREPARE_FEATURES = '''
        # --- PATCH V6: Fit Normalizer & Setup ObsBuilder ---
        print("  🔄 V6: Fitting Adaptive Normalizer...")
        self.normalizer.fit(self.processed_data)
        
        print("  🏗️ V6: Initializing 3D Observation Builder...")
        self.obs_builder = ObservationBuilderV7(
            n_tickers=self.n_assets,
            lookback=self.lookback_period if hasattr(self, 'lookback_period') else 60,
            feature_columns=self.feature_columns,
            normalize=True
        )
        
        # Update observation space for Transformer compatibility
        obs_dim = self.obs_builder.get_observation_space_size()
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        print(f"  ✅ V6: Observation space updated to {obs_dim} dims")
'''

PATCH_GET_OBSERVATION = '''
    def _get_observation(self) -> np.ndarray:
        """V6: Délègue la construction à ObservationBuilderV7"""
        return self.obs_builder.build_observation(
            processed_data=self.processed_data,
            tickers=self.tickers,
            current_step=self.current_step,
            portfolio=self.portfolio,
            balance=self.balance,
            equity=self.equity,
            initial_balance=self.initial_balance,
            peak_value=self.peak_value,
        )
'''

PATCH_REWARD_IN_STEP = '''
        # --- PATCH V6: Reward Avancé ---
        # Calcule le rendement du step
        prev_equity = self.portfolio_value_history[-2] if len(self.portfolio_value_history) > 1 else self.initial_balance
        step_return = (self.equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        
        # Calcule le max drawdown actuel
        max_dd = (self.peak_value - self.equity) / self.peak_value if self.peak_value > 0 else 0
        
        # Utilise le nouveau calculateur
        reward = self.reward_calc.calculate(
            step_return=step_return,
            winning_trades=self.winning_trades,
            total_trades=self.total_trades,
            max_drawdown=-max_dd,
            trades_executed=trades_executed
        )
'''


def backup_file(filepath):
    """Crée une sauvegarde du fichier original"""
    if os.path.exists(filepath):
        shutil.copy(filepath, BACKUP_FILE)
        print(f"✅ Sauvegarde créée: {BACKUP_FILE}")
        return True
    return False


def add_imports(content):
    """Ajoute les imports V6 au début du fichier"""
    # Trouve la position après les imports existants
    lines = content.split('\n')
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith('from core.advanced_features_v2'):
            insert_pos = i + 1
            break
    
    # Insère les nouveaux imports
    lines.insert(insert_pos, PATCH_IMPORTS)
    return '\n'.join(lines)


def add_init_patch(content):
    """Ajoute les initialisations V6 dans __init__"""
    # Trouve où insérer (avant l'observation_space)
    if 'self.observation_space = gym.spaces.Box(' in content:
        content = content.replace(
            'self.observation_space = gym.spaces.Box(',
            PATCH_INIT_ADDITIONS + '\n        self.observation_space = gym.spaces.Box(',
            1
        )
    return content


def patch_prepare_features(content):
    """Ajoute le patch à _prepare_features_v2"""
    if 'self.feature_columns = [' in content:
        # Trouve la fin de la méthode
        pos = content.find('print(f"  ✅ {len(self.feature_columns)} features')
        if pos != -1:
            # Trouve la fin de la ligne
            end_pos = content.find('\n', pos) + 1
            content = content[:end_pos] + PATCH_PREPARE_FEATURES + content[end_pos:]
    return content


def patch_get_observation(content):
    """Remplace la méthode _get_observation"""
    # Trouve la méthode
    start = content.find('    def _get_observation(self) -> np.ndarray:')
    if start == -1:
        return content
    
    # Trouve la fin de la méthode (prochaine méthode ou fin du fichier)
    end = content.find('\n    def ', start + 1)
    if end == -1:
        end = len(content)
    
    # Remplace
    content = content[:start] + PATCH_GET_OBSERVATION + '\n' + content[end:]
    return content


def patch_step_reward(content):
    """Ajoute le patch de récompense dans step()"""
    # Trouve la section du reward dans step()
    if 'reward = self._calculate_reward(' in content:
        # Remplace l'appel à l'ancienne méthode
        content = content.replace(
            'reward = self._calculate_reward(total_reward, trades_executed)',
            PATCH_REWARD_IN_STEP,
            1
        )
    return content


def remove_old_calculate_reward(content):
    """Supprime l'ancienne méthode _calculate_reward"""
    # Trouve la méthode
    start = content.find('    def _calculate_reward(self, trade_reward: float, trades_executed: int) -> float:')
    if start == -1:
        return content
    
    # Trouve la fin de la méthode
    end = content.find('\n    def ', start + 1)
    if end == -1:
        end = len(content)
    
    # Supprime
    content = content[:start] + content[end:]
    return content


def verify_patches(content):
    """Vérifie que tous les patchs ont été appliqués"""
    checks = [
        ('ObservationBuilderV7' in content, 'ObservationBuilderV7 import'),
        ('DifferentialSharpeRewardCalculator' in content, 'DifferentialSharpeRewardCalculator import'),
        ('AdaptiveNormalizer' in content, 'AdaptiveNormalizer import'),
        ('self.reward_calc' in content, 'Reward calculator initialization'),
        ('self.obs_builder' in content, 'Observation builder initialization'),
        ('self.obs_builder.build_observation' in content, '_get_observation patch'),
    ]
    
    print("\n🔍 Vérification des patchs:")
    all_ok = True
    for check, name in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {name}")
        if not check:
            all_ok = False
    
    return all_ok


def main():
    print("\n" + "="*70)
    print("  🚀 PLOUTOS V6 PATCHES - AUTOMATIC INTEGRATION")
    print("="*70 + "\n")
    
    # Vérifie que le fichier existe
    if not os.path.exists(ENV_FILE):
        print(f"❌ Erreur: {ENV_FILE} non trouvé!")
        sys.exit(1)
    
    # Sauvegarde l'original
    print("📦 Sauvegarde de l'original...")
    if not backup_file(ENV_FILE):
        print(f"❌ Impossible de sauvegarder {ENV_FILE}")
        sys.exit(1)
    
    # Lit le fichier
    print("📖 Lecture du fichier...")
    with open(ENV_FILE, 'r') as f:
        content = f.read()
    
    # Applique les patchs
    print("🔧 Application des patchs...\n")
    
    print("  1️⃣ Ajout des imports...")
    content = add_imports(content)
    
    print("  2️⃣ Patch __init__...")
    content = add_init_patch(content)
    
    print("  3️⃣ Patch _prepare_features_v2...")
    content = patch_prepare_features(content)
    
    print("  4️⃣ Remplacement _get_observation...")
    content = patch_get_observation(content)
    
    print("  5️⃣ Patch reward dans step...")
    content = patch_step_reward(content)
    
    print("  6️⃣ Suppression ancienne _calculate_reward...")
    content = remove_old_calculate_reward(content)
    
    # Vérifie les patchs
    if not verify_patches(content):
        print("\n⚠️  ATTENTION: Certains patchs n'ont pas pu être appliqués!")
        print(f"   Fichier de secours: {BACKUP_FILE}")
        sys.exit(1)
    
    # Écrit le fichier modifié
    print("\n💾 Écriture du fichier modifié...")
    with open(ENV_FILE, 'w') as f:
        f.write(content)
    
    print(f"\n✅ SUCCÈS! Tous les patchs ont été appliqués à {ENV_FILE}")
    print(f"   Backup: {BACKUP_FILE}")
    print("\n" + "="*70)
    print("  🎯 PROCHAINES ÉTAPES:")
    print("="*70)
    print("\n1. Vérifiez les modifications:")
    print(f"   git diff {ENV_FILE}\n")
    print("2. Testez l'environnement:")
    print("   python -c \"from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming; print('OK')\"\n")
    print("3. Lancez le training:")
    print("   python scripts/train_v6_extended_with_optimizations.py\n")


if __name__ == "__main__":
    main()
