"""
Deployer - Gère le déploiement des modèles
Création liens symboliques, versioning
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config.config import PloutosConfig
from utils.logger import PloutosLogger
from utils.helpers import ensure_dir

logger = PloutosLogger().get_logger(__name__)

class ModelDeployer:
    """
    Déploie des modèles validés
    
    Responsabilités:
    - Versioning des modèles
    - Création liens symboliques (latest, production)
    - Sauvegarde config associée
    """
    
    def __init__(self, config: PloutosConfig, deployment_dir: str = 'models/production'):
        """
        Args:
            config: Configuration Ploutos
            deployment_dir: Dossier de déploiement
        """
        self.config = config
        self.deployment_dir = Path(deployment_dir)
        ensure_dir(self.deployment_dir)
    
    def deploy(
        self,
        model_path: str,
        metadata: Dict[str, Any] = None,
        create_symlinks: bool = True
    ) -> Dict[str, str]:
        """
        Déploie un modèle
        
        Args:
            model_path: Chemin du modèle à déployer
            metadata: Métadonnées additionnelles
            create_symlinks: Créer liens symboliques
            
        Returns:
            Dict avec chemins de déploiement
        """
        
        logger.info("="*80)
        logger.info("DÉPLOIEMENT MODÈLE")
        logger.info("="*80)
        
        model_src = Path(model_path)
        
        if not model_src.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Timestamp pour versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Chemins destination
        model_dest = self.deployment_dir / f"model_{timestamp}.zip"
        config_dest = self.deployment_dir / f"config_{timestamp}.json"
        
        # Copier modèle
        import shutil
        shutil.copy(model_src, model_dest)
        logger.info(f"Modèle copié: {model_dest}")
        
        # Sauvegarder config associée
        deploy_config = {
            'timestamp': timestamp,
            'source_model': str(model_src),
            'config': self.config.to_dict()
        }
        
        if metadata:
            deploy_config['metadata'] = metadata
        
        with open(config_dest, 'w') as f:
            json.dump(deploy_config, f, indent=2)
        
        logger.info(f"Config sauvegardée: {config_dest}")
        
        # Créer liens symboliques
        symlinks = {}
        
        if create_symlinks:
            # Lien "latest"
            latest_link = self.deployment_dir / "latest.zip"
            latest_config = self.deployment_dir / "latest_config.json"
            
            self._create_symlink(model_dest, latest_link)
            self._create_symlink(config_dest, latest_config)
            
            symlinks['latest'] = str(latest_link)
            logger.info(f"Lien créé: latest.zip → {model_dest.name}")
            
            # Lien "production"
            production_link = self.deployment_dir / "production.zip"
            production_config = self.deployment_dir / "production_config.json"
            
            self._create_symlink(model_dest, production_link)
            self._create_symlink(config_dest, production_config)
            
            symlinks['production'] = str(production_link)
            logger.info(f"Lien créé: production.zip → {model_dest.name}")
        
        logger.info("="*80)
        logger.info("✅ DÉPLOIEMENT RÉUSSI")
        logger.info("="*80)
        
        return {
            'model': str(model_dest),
            'config': str(config_dest),
            **symlinks
        }
    
    def _create_symlink(self, src: Path, dest: Path):
        """Crée un lien symbolique (supprime ancien si existe)"""
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        
        # Lien relatif pour portabilité
        rel_src = os.path.relpath(src, dest.parent)
        dest.symlink_to(rel_src)
    
    def list_deployed_models(self) -> list:
        """Liste tous les modèles déployés"""
        models = sorted(self.deployment_dir.glob("model_*.zip"))
        return [str(m) for m in models]
    
    def get_production_model(self) -> str:
        """Retourne chemin du modèle en production"""
        prod_link = self.deployment_dir / "production.zip"
        
        if not prod_link.exists():
            raise FileNotFoundError("Aucun modèle en production")
        
        return str(prod_link.resolve())
