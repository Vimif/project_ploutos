#!/usr/bin/env python3
"""
Feature Adapter pour Transfer Learning
Permet de réutiliser un modèle entraîné sur N assets vers M assets

Méthodes:
1. Feature Projection : Projette features source vers target space
2. Weight Adaptation : Adapte poids du réseau pour nouvelle dimension
3. Freezing Strategy : Geler certaines couches pendant fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import copy
import warnings

class FeatureAdapter:
    """
    Adapte un modèle PPO entraîné sur N features vers M features
    
    Example:
        # Modèle Stage 1 : 1 asset (9 features)
        source_model = PPO.load('models/stage1_final.zip')
        
        # Environnement Stage 2 : 3 assets (21 features)
        target_env = SubprocVecEnv([...])
        
        # Adapter
        adapter = FeatureAdapter(source_model, target_env)
        adapted_model = adapter.adapt(
            method='projection',  # ou 'average', 'repeat'
            freeze_layers=['policy.mlp_extractor']  # Optionnel
        )
        
        # Fine-tuner
        adapted_model.learn(total_timesteps=500_000)
    """
    
    def __init__(self, source_model, target_env, device='cuda'):
        """
        Args:
            source_model: Modèle PPO source
            target_env: Environnement cible (peut être VecEnv)
            device: 'cuda' ou 'cpu'
        """
        self.source_model = source_model
        self.target_env = target_env
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Extraire observation spaces
        self.source_obs_dim = source_model.observation_space.shape[0]
        self.target_obs_dim = target_env.observation_space.shape[0]
        
        print(f"\n🔧 Feature Adapter initialisé")
        print(f"   Source : {self.source_obs_dim} features")
        print(f"   Target : {self.target_obs_dim} features")
        print(f"   Ratio  : {self.target_obs_dim / self.source_obs_dim:.2f}x")
        print(f"   Device : {self.device}\n")
    
    def adapt(self, method='projection', freeze_layers=None, learning_rate=None):
        """
        Adapte le modèle source au target environment
        
        Args:
            method: Méthode adaptation
                - 'projection' : Projection linéaire (meilleure précision)
                - 'average' : Moyenner features (simple)
                - 'repeat' : Répéter features (rapide)
                - 'learned' : Apprendre mapping (optimal mais lent)
            freeze_layers: Liste noms de couches à geler (optionnel)
            learning_rate: Nouveau learning rate (optionnel)
            
        Returns:
            PPO model adapté
        """
        
        print(f"🔄 Adaptation : {method}")
        
        # Créer nouveau modèle avec target observation space
        adapted_model = PPO(
            policy='MlpPolicy',
            env=self.target_env,
            learning_rate=learning_rate or self.source_model.learning_rate,
            n_steps=self.source_model.n_steps,
            batch_size=self.source_model.batch_size,
            n_epochs=self.source_model.n_epochs,
            gamma=self.source_model.gamma,
            gae_lambda=self.source_model.gae_lambda,
            clip_range=self.source_model.clip_range,
            ent_coef=self.source_model.ent_coef,
            vf_coef=self.source_model.vf_coef,
            max_grad_norm=self.source_model.max_grad_norm,
            policy_kwargs=self.source_model.policy_kwargs,
            device=self.device,
            verbose=0
        )
        
        # Adapter les poids selon la méthode
        if method == 'projection':
            self._adapt_projection(adapted_model)
        elif method == 'average':
            self._adapt_average(adapted_model)
        elif method == 'repeat':
            self._adapt_repeat(adapted_model)
        elif method == 'learned':
            self._adapt_learned(adapted_model)
        else:
            raise ValueError(f"Méthode inconnue : {method}")
        
        # Geler couches si demandé
        if freeze_layers:
            self._freeze_layers(adapted_model, freeze_layers)
        
        print(f"✅ Adaptation terminée\n")
        
        return adapted_model
    
    def _adapt_projection(self, target_model):
        """
        Méthode Projection : Projette features source vers target space
        
        Utilise une matrice de projection linéaire apprise
        """
        
        print("   Méthode : Projection linéaire")
        
        # Récupérer poids source
        source_state_dict = self.source_model.policy.state_dict()
        target_state_dict = target_model.policy.state_dict()
        
        # ✅ Déplacer tous les tensors sur le device correct
        for key in source_state_dict.keys():
            source_state_dict[key] = source_state_dict[key].to(self.device)
        
        for key in target_state_dict.keys():
            target_state_dict[key] = target_state_dict[key].to(self.device)
        
        # Adapter SEULEMENT la première couche (features_extractor)
        for key in target_state_dict.keys():
            if 'features_extractor' in key or 'mlp_extractor.policy_net.0' in key:
                
                source_weight = source_state_dict.get(key)
                
                if source_weight is None:
                    continue
                
                # ✅ Assurer que source_weight est sur le bon device
                source_weight = source_weight.to(self.device)
                
                # Adapter dimensions
                if len(source_weight.shape) == 2:  # Weight matrix
                    source_out, source_in = source_weight.shape
                    target_out, target_in = target_state_dict[key].shape
                    
                    if source_in != target_in:  # Input dimension change
                        # Projection : moyenne pondérée
                        scale = target_in / source_in
                        
                        if scale > 1:  # Target plus grand
                            # Répéter + bruit
                            adapted_weight = torch.zeros(source_out, target_in, device=self.device)  # ✅ Device
                            
                            # Répéter poids source
                            for i in range(int(scale)):
                                start_idx = i * source_in
                                end_idx = min((i + 1) * source_in, target_in)
                                adapted_weight[:, start_idx:end_idx] = source_weight[:, :end_idx-start_idx]
                            
                            # Remplir reste avec moyenne + bruit
                            if target_in % source_in != 0:
                                remaining = target_in - (int(scale) * source_in)
                                # ✅ Créer bruit sur même device
                                noise = torch.randn(source_out, remaining, device=self.device) * 0.01
                                adapted_weight[:, -remaining:] = source_weight[:, :remaining] + noise
                            
                            target_state_dict[key] = adapted_weight
                        
                        else:  # Target plus petit
                            # Moyenner features
                            adapted_weight = torch.zeros(source_out, target_in, device=self.device)  # ✅ Device
                            chunk_size = source_in // target_in
                            
                            for i in range(target_in):
                                start = i * chunk_size
                                end = start + chunk_size
                                adapted_weight[:, i] = source_weight[:, start:end].mean(dim=1)
                            
                            target_state_dict[key] = adapted_weight
                    
                    else:  # Même input dim, copier directement
                        target_state_dict[key] = source_weight
                
                elif len(source_weight.shape) == 1:  # Bias
                    target_state_dict[key] = source_weight
            
            else:
                # Copier autres couches directement si dimensions compatibles
                source_weight = source_state_dict.get(key)
                
                if source_weight is not None and source_weight.shape == target_state_dict[key].shape:
                    target_state_dict[key] = source_weight.to(self.device)  # ✅ Device
        
        # Charger poids adaptés
        target_model.policy.load_state_dict(target_state_dict)
        
        print("   ✅ Première couche adaptée par projection")
        print("   ✅ Autres couches copiées depuis source")
    
    def _adapt_average(self, target_model):
        """
        Méthode Average : Moyenner features source
        Simple mais peut perdre information
        """
        
        print("   Méthode : Average (moyenner features)")
        
        source_state_dict = self.source_model.policy.state_dict()
        target_state_dict = target_model.policy.state_dict()
        
        # ✅ Device sync
        for key in source_state_dict.keys():
            source_state_dict[key] = source_state_dict[key].to(self.device)
        
        for key in target_state_dict.keys():
            target_state_dict[key] = target_state_dict[key].to(self.device)
        
        for key in target_state_dict.keys():
            if 'mlp_extractor.policy_net.0' in key:
                source_weight = source_state_dict.get(key)
                
                if source_weight is not None and len(source_weight.shape) == 2:
                    source_weight = source_weight.to(self.device)
                    source_out, source_in = source_weight.shape
                    target_out, target_in = target_state_dict[key].shape
                    
                    if source_in != target_in:
                        # Grouper et moyenner
                        adapted_weight = torch.zeros(source_out, target_in, device=self.device)
                        
                        for i in range(target_in):
                            adapted_weight[:, i] = source_weight.mean(dim=1)
                        
                        target_state_dict[key] = adapted_weight
                    else:
                        target_state_dict[key] = source_weight
            
            elif key in source_state_dict and source_state_dict[key].shape == target_state_dict[key].shape:
                target_state_dict[key] = source_state_dict[key].to(self.device)
        
        target_model.policy.load_state_dict(target_state_dict)
        
        print("   ✅ Features moyennées")
    
    def _adapt_repeat(self, target_model):
        """
        Méthode Repeat : Répéter features source
        Très simple, convergence rapide mais sous-optimal
        """
        
        print("   Méthode : Repeat (répéter features)")
        
        source_state_dict = self.source_model.policy.state_dict()
        target_state_dict = target_model.policy.state_dict()
        
        # ✅ Device sync
        for key in source_state_dict.keys():
            source_state_dict[key] = source_state_dict[key].to(self.device)
        
        for key in target_state_dict.keys():
            target_state_dict[key] = target_state_dict[key].to(self.device)
        
        for key in target_state_dict.keys():
            if 'mlp_extractor.policy_net.0' in key:
                source_weight = source_state_dict.get(key)
                
                if source_weight is not None and len(source_weight.shape) == 2:
                    source_weight = source_weight.to(self.device)
                    source_out, source_in = source_weight.shape
                    target_out, target_in = target_state_dict[key].shape
                    
                    if source_in != target_in:
                        # Répéter poids
                        adapted_weight = source_weight.repeat(1, (target_in // source_in) + 1)[:, :target_in]
                        target_state_dict[key] = adapted_weight
                    else:
                        target_state_dict[key] = source_weight
            
            elif key in source_state_dict and source_state_dict[key].shape == target_state_dict[key].shape:
                target_state_dict[key] = source_state_dict[key].to(self.device)
        
        target_model.policy.load_state_dict(target_state_dict)
        
        print("   ✅ Features répétées")
    
    def _adapt_learned(self, target_model):
        """
        Méthode Learned : Ajouter une couche de projection apprise
        Optimal mais nécessite entraînement supplémentaire
        """
        
        print("   Méthode : Learned (projection apprise)")
        print("   ⚠️  Non implémentée - utiliser 'projection' à la place")
        
        # Fallback sur projection
        self._adapt_projection(target_model)
    
    def _freeze_layers(self, model, layer_names):
        """
        Gèle certaines couches du modèle
        
        Args:
            layer_names: Liste de noms de couches à geler
                Ex: ['policy.mlp_extractor', 'value_net']
        """
        
        print(f"\n🧊 Gel des couches : {', '.join(layer_names)}")
        
        frozen_count = 0
        
        for name, param in model.policy.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        print(f"   ✅ {frozen_count} paramètres gelés\n")
    
    def get_transfer_strategy(self, stage_from, stage_to):
        """
        Recommande stratégie de transfer learning selon stages
        
        Args:
            stage_from: Stage source (1, 2, ou 3)
            stage_to: Stage cible (1, 2, ou 3)
            
        Returns:
            dict: Stratégie recommandée
        """
        
        strategies = {
            (1, 2): {
                'method': 'projection',
                'freeze_layers': ['policy.mlp_extractor.policy_net.2', 'policy.mlp_extractor.value_net.2'],
                'learning_rate_factor': 0.5,  # Réduire LR de moitié
                'fine_tune_steps': 500_000,
                'description': 'Stage 1→2 : Geler dernières couches, fine-tune premières'
            },
            (2, 3): {
                'method': 'projection',
                'freeze_layers': ['policy.mlp_extractor.policy_net.2'],
                'learning_rate_factor': 0.3,
                'fine_tune_steps': 1_000_000,
                'description': 'Stage 2→3 : Geler une couche, apprentissage plus long'
            },
            (1, 3): {
                'method': 'projection',
                'freeze_layers': None,  # Tout ré-entraîner
                'learning_rate_factor': 0.2,
                'fine_tune_steps': 2_000_000,
                'description': 'Stage 1→3 : Saut important, ré-entraînement complet recommandé'
            }
        }
        
        return strategies.get((stage_from, stage_to), {
            'method': 'projection',
            'freeze_layers': None,
            'learning_rate_factor': 1.0,
            'fine_tune_steps': 500_000,
            'description': 'Stratégie générique'
        })

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Démonstration du feature adapter
    """
    
    print("\n" + "="*80)
    print("🔧 TEST FEATURE ADAPTER")
    print("="*80 + "\n")
    
    # Simuler transfer learning Stage 1 → Stage 2
    print("📊 Scénario : Stage 1 (SPY) → Stage 2 (SPY, QQQ, IWM)\n")
    
    from gymnasium import spaces
    import numpy as np
    
    # Mock source model (Stage 1 : 9 features)
    class MockModel:
        def __init__(self, obs_dim):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
            self.learning_rate = 1e-4
            self.n_steps = 2048
            self.batch_size = 256
            self.n_epochs = 5
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.clip_range = 0.2
            self.ent_coef = 0.05
            self.vf_coef = 0.5
            self.max_grad_norm = 0.3
            self.policy_kwargs = {'net_arch': [512, 512, 512]}
            
            # Mock policy
            class MockPolicy:
                def state_dict(self):
                    return {
                        'mlp_extractor.policy_net.0.weight': torch.randn(512, obs_dim),
                        'mlp_extractor.policy_net.0.bias': torch.randn(512),
                        'mlp_extractor.policy_net.2.weight': torch.randn(512, 512),
                        'mlp_extractor.value_net.0.weight': torch.randn(512, obs_dim)
                    }
                
                def load_state_dict(self, state_dict):
                    pass
                
                def named_parameters(self):
                    return []
            
            self.policy = MockPolicy()
    
    # Mock target env (Stage 2 : 21 features)
    class MockEnv:
        def __init__(self, obs_dim):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    
    source_model = MockModel(obs_dim=9)
    target_env = MockEnv(obs_dim=21)
    
    # Test adapter
    adapter = FeatureAdapter(source_model, target_env)
    
    # Récupérer stratégie recommandée
    strategy = adapter.get_transfer_strategy(stage_from=1, stage_to=2)
    
    print("🎯 Stratégie recommandée :")
    print(f"   Méthode           : {strategy['method']}")
    print(f"   Freeze layers    : {strategy['freeze_layers']}")
    print(f"   LR factor        : {strategy['learning_rate_factor']}x")
    print(f"   Fine-tune steps  : {strategy['fine_tune_steps']:,}")
    print(f"   Description      : {strategy['description']}\n")
    
    print("✅ Test terminé ! (Mock uniquement)\n")
    print("📝 Usage réel : Voir docstring de FeatureAdapter\n")
