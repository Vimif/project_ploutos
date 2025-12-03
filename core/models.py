# core/models.py
"""Gestion des modèles IA"""
from pathlib import Path
from stable_baselines3 import PPO
import logging
from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestionnaire de modèles IA"""
    
    def __init__(self):
        self.models = {}
        self.models_dir = MODELS_DIR
    
    def load_model(self, model_name: str, device='auto'):
        """
        Charger un modèle
        
        Args:
            model_name: Nom du modèle (ex: 'brain_tech')
            device: 'cuda', 'cpu', ou 'auto'
        
        Returns:
            PPO model ou None si erreur
        """
        if model_name in self.models:
            logger.info(f"♻️  Modèle {model_name} déjà chargé (cache)")
            return self.models[model_name]
        
        model_path = self.models_dir / f"{model_name}.zip"
        
        if not model_path.exists():
            logger.error(f"❌ Modèle introuvable: {model_path}")
            return None
        
        try:
            model = PPO.load(model_path, device=device)
            self.models[model_name] = model
            logger.info(f"✅ Modèle {model_name} chargé depuis {model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Erreur chargement {model_name}: {e}")
            return None
    
    def save_model(self, model, model_name: str):
        """
        Sauvegarder un modèle
        
        Args:
            model: Instance PPO
            model_name: Nom du fichier (sans extension)
        """
        model_path = self.models_dir / f"{model_name}.zip"
        model.save(model_path)
        logger.info(f"💾 Modèle sauvegardé: {model_path}")
        
        # Mettre en cache
        self.models[model_name] = model
    
    def list_models(self):
        """Lister tous les modèles disponibles"""
        models = list(self.models_dir.glob("*.zip"))
        return [m.stem for m in models]
    
    def get_model_info(self, model_name: str):
        """Obtenir infos sur un modèle"""
        model_path = self.models_dir / f"{model_name}.zip"
        
        if not model_path.exists():
            return None
        
        stat = model_path.stat()
        return {
            'name': model_name,
            'path': str(model_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime
        }
    
    def clear_cache(self):
        """Vider le cache des modèles chargés"""
        self.models.clear()
        logger.info("🧹 Cache des modèles vidé")
