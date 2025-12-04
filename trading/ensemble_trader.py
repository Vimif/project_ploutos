import numpy as np
from stable_baselines3 import PPO
import os

class EnsembleTrader:
    """Vote majoritaire de plusieurs modèles PPO"""
    
    def __init__(self, ticker, ensemble_dir="models/ensemble"):
        """
        Charge tous les modèles d'un ensemble
        
        Args:
            ticker: Nom du ticker (ex: "NVDA")
            ensemble_dir: Dossier contenant les modèles
        """
        self.ticker = ticker
        self.models = []
        
        # Lire le manifest
        manifest_path = f"{ensemble_dir}/{ticker}_manifest.txt"
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            model_paths = [line.strip() for line in f if line.strip()]
        
        # Charger tous les modèles
        print(f"📥 Chargement de {len(model_paths)} modèles pour {ticker}...")
        for path in model_paths:
            if os.path.exists(path):
                model = PPO.load(path)
                self.models.append(model)
                print(f"   ✅ {os.path.basename(path)}")
            else:
                print(f"   ⚠️  SKIP : {path} introuvable")
        
        if len(self.models) == 0:
            raise ValueError("Aucun modèle chargé !")
        
        print(f"✅ Ensemble prêt : {len(self.models)} modèles actifs\n")
    
    def predict(self, observation, deterministic=True):
        """
        Vote majoritaire des modèles
        
        Args:
            observation: État de l'environnement
            deterministic: Si True, chaque modèle vote de manière déterministe
        
        Returns:
            action: Action finale (vote majoritaire)
            confidence: % de confiance (proportion du vote gagnant)
        """
        votes = []
        
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            votes.append(int(action))
        
        # Compter les votes
        vote_counts = np.bincount(votes, minlength=3)  # 3 actions possibles
        
        # Action gagnante
        final_action = np.argmax(vote_counts)
        
        # Confiance (% du vote majoritaire)
        confidence = vote_counts[final_action] / len(self.models)
        
        return final_action, confidence
    
    def predict_all_votes(self, observation):
        """Retourne tous les votes individuels (pour debug)"""
        votes = []
        for i, model in enumerate(self.models):
            action, _ = model.predict(observation, deterministic=True)
            votes.append({"model_id": i, "action": int(action)})
        return votes

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    from core.environment import TradingEnv
    
    # Charger l'ensemble
    ensemble = EnsembleTrader("NVDA")
    
    # Créer un environnement de test
    env = TradingEnv(csv_path="data_cache/NVDA.csv")
    obs, _ = env.reset()
    
    # Faire une prédiction
    action, confidence = ensemble.predict(obs)
    
    print(f"Action finale : {action} (confiance: {confidence*100:.1f}%)")
    
    # Voir les votes individuels
    all_votes = ensemble.predict_all_votes(obs)
    print("\nVotes individuels :")
    for vote in all_votes:
        print(f"   Modèle {vote['model_id']} → Action {vote['action']}")
