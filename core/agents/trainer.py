"""
Trainer - Responsable unique de l'entraînement
"""

import sys
from pathlib import Path

# Ajouter root au path
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from typing import Dict, Any, Callable
from datetime import datetime
import json

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

from config.config import PloutosConfig
from utils.logger import PloutosLogger
from utils.helpers import ensure_dir

logger = PloutosLogger().get_logger(__name__)

class ModelTrainer:
    """
    Entraîne des modèles PPO
    
    Responsabilités:
    - Créer environnements vectorisés
    - Configurer PPO avec bons hyperparamètres
    - Gérer callbacks (W&B, checkpoints, etc.)
    - Sauvegarder modèle + metadata
    """
    
    def __init__(self, config: PloutosConfig):
        """
        Args:
            config: Configuration Ploutos
        """
        self.config = config
        self.model = None
        self.training_info = {}
        
        # Vérifier GPU si demandé
        if config.training.device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA demandé mais non disponible, passage CPU")
                self.config.training.device = 'cpu'
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU détecté: {gpu_name} ({gpu_mem:.1f} GB)")
    
    def train(
        self,
        env_class: Callable,
        data: Dict[str, pd.DataFrame],
        hyperparams: Dict[str, Any] = None,
        callbacks: list = None
    ) -> PPO:
        """
        Entraîne un modèle PPO
        
        Args:
            env_class: Classe d'environnement (ex: UniversalTradingEnv)
            data: Données historiques {ticker: DataFrame}
            hyperparams: Hyperparamètres custom (ou None = config)
            callbacks: Liste de callbacks SB3
            
        Returns:
            Modèle PPO entraîné
        """
        
        logger.info("="*80)
        logger.info("DÉBUT ENTRAÎNEMENT")
        logger.info("="*80)
        
        # Hyperparams
        if hyperparams is None:
            hyperparams = self.config.get_ppo_kwargs()
        
        logger.info(f"Assets: {len(data)}")
        logger.info(f"Device: {self.config.training.device}")
        logger.info(f"Timesteps: {self.config.training.timesteps:,}")
        logger.info(f"N Envs: {self.config.training.n_envs}")
        logger.info(f"Learning Rate: {hyperparams['learning_rate']:.6f}")
        logger.info(f"Batch Size: {hyperparams['batch_size']}")
        
        # Créer environnements
        logger.info("Création environnements...")
        
        def make_env():
            return env_class(
                data=data,
                initial_balance=100000,
                commission=0.001,
                max_steps=1000
            )
        
        # Choisir vectorisation
        if self.config.training.n_envs > 1:
            try:
                env = SubprocVecEnv([make_env for _ in range(self.config.training.n_envs)])
                logger.info(f"SubprocVecEnv créé ({self.config.training.n_envs} envs)")
            except Exception as e:
                logger.warning(f"SubprocVecEnv échoué ({e}), fallback DummyVecEnv")
                env = DummyVecEnv([make_env for _ in range(self.config.training.n_envs)])
        else:
            env = DummyVecEnv([make_env])
            logger.info("DummyVecEnv créé (1 env)")
        
        # Créer modèle
        logger.info("Création modèle PPO...")
        
        self.model = PPO(
            "MlpPolicy",
            env,
            device=self.config.training.device,
            tensorboard_log="logs/tensorboard",
            verbose=1,
            **hyperparams
        )
        
        logger.info("Modèle créé avec succès")
        
        # Préparer callbacks
        callback_list = callbacks if callbacks else []
        
        # Entraîner
        logger.info("="*80)
        logger.info("ENTRAÎNEMENT EN COURS...")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            self.model.learn(
                total_timesteps=self.config.training.timesteps,
                callback=CallbackList(callback_list) if callback_list else None,
                progress_bar=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Entraînement terminé en {duration/60:.1f} min")
            
            # Stocker infos
            self.training_info = {
                'duration_seconds': duration,
                'timesteps': self.config.training.timesteps,
                'device': self.config.training.device,
                'n_envs': self.config.training.n_envs,
                'hyperparams': hyperparams
            }
            
        except KeyboardInterrupt:
            logger.warning("Entraînement interrompu par utilisateur")
            raise
            
        except Exception as e:
            logger.error(f"Erreur durant entraînement: {e}", exc_info=True)
            raise
            
        finally:
            env.close()
            logger.info("Environnements fermés")
        
        return self.model
    
    def save(self, path: str, metadata: Dict[str, Any] = None):
        """
        Sauvegarde modèle + metadata
        
        Args:
            path: Chemin de sauvegarde (.zip)
            metadata: Métadonnées additionnelles
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder (entraîner d'abord)")
        
        # Sauvegarder modèle
        model_path = Path(path)
        ensure_dir(model_path.parent)
        
        self.model.save(str(model_path))
        logger.info(f"Modèle sauvegardé: {model_path}")
        
        # Sauvegarder metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'training_info': self.training_info,
            'config': self.config.to_dict()
        }
        
        if metadata:
            meta.update(metadata)
        
        meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Metadata sauvegardée: {meta_path}")
    
    @staticmethod
    def load(path: str) -> PPO:
        """
        Charge un modèle sauvegardé
        
        Args:
            path: Chemin du modèle (.zip)
            
        Returns:
            Modèle PPO chargé
        """
        model = PPO.load(path)
        logger.info(f"Modèle chargé: {path}")
        return model
