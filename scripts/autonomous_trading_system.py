"""
Système de trading autonome complet
Meta-agent qui orchestre tout
"""

from core.market_regime import MarketRegimeDetector
from core.asset_selector import UniversalAssetSelector
from core.auto_optimizer import AutoOptimizer
from core.universal_environment import UniversalTradingEnv
from stable_baselines3 import PPO

class AutonomousTradingSystem:
    """Système de trading 100% autonome"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.asset_selector = UniversalAssetSelector(self.regime_detector)
        
    def run_full_cycle(self):
        """Cycle complet : Analyse → Sélection → Optimisation → Entraînement"""
        
        print("="*70)
        print("🤖 AUTONOMOUS TRADING SYSTEM - FULL CYCLE")
        print("="*70)
        
        # 1. Détecter régime
        print("\n📊 PHASE 1 : Market Regime Detection")
        regime_info = self.regime_detector.detect()
        print(f"  Régime : {regime_info['regime']}")
        print(f"  Confiance : {regime_info['confidence']:.2%}")
        
        # 2. Sélectionner assets
        print("\n🎯 PHASE 2 : Asset Selection")
        selected_assets = self.asset_selector.select_assets(n_assets=10)
        
        # 3. Auto-optimiser les hyperparamètres
        print("\n⚙️ PHASE 3 : Hyperparameter Optimization")
        optimizer = AutoOptimizer(selected_assets[0], f"data_cache/{selected_assets[0]}.csv")
        best_params = optimizer.optimize(n_trials=20)
        
        # 4. Entraîner le modèle universel
        print("\n🧠 PHASE 4 : Universal Model Training")
        env = UniversalTradingEnv(selected_assets, self.regime_detector)
        
        model = PPO("MlpPolicy", env, verbose=1, device="cuda", **best_params)
        model.learn(total_timesteps=2_000_000, progress_bar=True)
        
        # 5. Sauvegarder
        model.save("models/production/autonomous_universal.zip")
        
        print("\n" + "="*70)
        print("🎉 CYCLE TERMINÉ - Modèle prêt pour production")
        print("="*70)

if __name__ == "__main__":
    system = AutonomousTradingSystem()
    system.run_full_cycle()
