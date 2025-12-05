#!/usr/bin/env python3
"""
Script de déploiement standalone
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PloutosConfig
from core.agents.deployer import ModelDeployer
from utils.logger import PloutosLogger

logger = PloutosLogger().get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Déployer un modèle validé')
    
    parser.add_argument(
        'model',
        help='Chemin du modèle à déployer (.zip)'
    )
    
    parser.add_argument(
        '--config',
        default='config/autonomous_config.yaml',
        help='Fichier de configuration'
    )
    
    parser.add_argument(
        '--no-symlinks',
        action='store_true',
        help='Ne pas créer liens symboliques'
    )
    
    args = parser.parse_args()
    
    # ════════════════════════════════════════════════════════════════
    # DÉPLOIEMENT
    # ════════════════════════════════════════════════════════════════
    
    logger.info("="*80)
    logger.info("PLOUTOS DEPLOYER")
    logger.info("="*80)
    
    # Config
    config = PloutosConfig.from_yaml(args.config)
    logger.info(f"✅ Config chargée: {args.config}")
    
    # Vérifier modèle existe
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ Modèle non trouvé: {args.model}")
        return 1
    
    # Déployer
    deployer = ModelDeployer(config)
    
    try:
        paths = deployer.deploy(
            model_path=str(model_path),
            create_symlinks=not args.no_symlinks
        )
        
        logger.info("\n" + "="*80)
        logger.info("✅ DÉPLOIEMENT RÉUSSI")
        logger.info("="*80)
        
        for key, path in paths.items():
            logger.info(f"{key}: {path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erreur déploiement: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
