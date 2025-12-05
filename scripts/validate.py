#!/usr/bin/env python3
"""
Script de validation standalone
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PloutosConfig
from core.environments.universal_env import UniversalTradingEnv
from core.agents.validator import ModelValidator
from core.agents.trainer import ModelTrainer
from utils.logger import PloutosLogger
from utils.helpers import load_cached_data

logger = PloutosLogger().get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Valider un modèle entraîné')
    
    parser.add_argument(
        'model',
        help='Chemin du modèle à valider (.zip)'
    )
    
    parser.add_argument(
        '--config',
        default='config/autonomous_config.yaml',
        help='Fichier de configuration'
    )
    
    parser.add_argument(
        '--assets',
        nargs='+',
        help='Assets à valider (ou auto depuis metadata)'
    )
    
    args = parser.parse_args()
    
    # ════════════════════════════════════════════════════════════════
    # CHARGEMENT
    # ════════════════════════════════════════════════════════════════
    
    logger.info("="*80)
    logger.info("PLOUTOS VALIDATOR")
    logger.info("="*80)
    
    # Config
    config = PloutosConfig.from_yaml(args.config)
    logger.info(f"✅ Config chargée: {args.config}")
    
    # Modèle
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ Modèle non trouvé: {args.model}")
        return 1
    
    model = ModelTrainer.load(str(model_path))
    logger.info(f"✅ Modèle chargé: {args.model}")
    
    # Assets
    if args.assets:
        assets = args.assets
    else:
        # Essayer de charger depuis metadata
        meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            assets = meta.get('assets', [])
        else:
            logger.error("❌ Pas d'assets spécifiés et pas de metadata")
            return 1
    
    logger.info(f"Assets: {assets}")
    
    # Données
    data = load_cached_data(assets)
    if len(data) == 0:
        logger.error("❌ Aucune donnée disponible")
        return 1
    
    logger.info(f"✅ {len(data)} datasets chargés")
    
    # ════════════════════════════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════════════════════════════
    
    validator = ModelValidator(config)
    
    results = validator.validate(
        model=model,
        env_class=UniversalTradingEnv,
        data=data
    )
    
    # ════════════════════════════════════════════════════════════════
    # RÉSULTAT
    # ════════════════════════════════════════════════════════════════
    
    if results['is_valid']:
        logger.info("\n✅ MODÈLE VALIDÉ - Prêt pour déploiement")
        return 0
    else:
        logger.warning("\n⚠️ MODÈLE NON VALIDÉ - Performance insuffisante")
        return 2

if __name__ == "__main__":
    sys.exit(main())
