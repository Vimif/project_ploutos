#!/usr/bin/env python3
"""
Script d'entraînement simplifié
Délègue toute la logique aux modules core
"""

import argparse
import sys
from pathlib import Path

# Ajouter root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PloutosConfig
from core.data.fetcher import UniversalDataFetcher
from core.environments.universal_env import UniversalTradingEnv
from core.agents.trainer import ModelTrainer
from core.market.regime_detector import MarketRegimeDetector
from core.market.asset_selector import UniversalAssetSelector
from utils.logger import PloutosLogger
from utils.helpers import load_cached_data, ensure_dir

logger = PloutosLogger().get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Entraîner un modèle de trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 scripts/train.py
  python3 scripts/train.py --config config/custom.yaml
  python3 scripts/train.py --output models/my_model.zip
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/autonomous_config.yaml',
        help='Fichier de configuration'
    )
    
    parser.add_argument(
        '--output',
        default='models/autonomous/trained_model.zip',
        help='Chemin de sauvegarde du modèle'
    )
    
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Utiliser données en cache uniquement'
    )
    
    args = parser.parse_args()
    
    # ════════════════════════════════════════════════════════════════
    # CHARGEMENT CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    
    logger.info("="*80)
    logger.info("PLOUTOS TRAINER")
    logger.info("="*80)
    
    try:
        config = PloutosConfig.from_yaml(args.config)
        logger.info(f"✅ Configuration chargée: {args.config}")
    except Exception as e:
        logger.error(f"Erreur chargement config: {e}")
        return 1
    
    # ════════════════════════════════════════════════════════════════
    # DÉTECTION RÉGIME
    # ════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*80)
    logger.info("PHASE 1: DÉTECTION RÉGIME")
    logger.info("─"*80)
    
    detector = MarketRegimeDetector(config.market.reference_ticker)
    regime_info = detector.detect(config.market.lookback_days)
    
    logger.info(f"✅ Régime: {regime_info['regime']} ({regime_info['confidence']:.1%})")
    
    # ════════════════════════════════════════════════════════════════
    # SÉLECTION ASSETS
    # ════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*80)
    logger.info("PHASE 2: SÉLECTION ASSETS")
    logger.info("─"*80)
    
    selector = UniversalAssetSelector(detector)
    assets = selector.select_assets(
        n_assets=config.assets.n_assets,
        lookback_days=config.market.lookback_days
    )
    
    logger.info(f"✅ {len(assets)} assets sélectionnés: {assets}")
    
    # ════════════════════════════════════════════════════════════════
    # CHARGEMENT DONNÉES
    # ════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*80)
    logger.info("PHASE 3: CHARGEMENT DONNÉES")
    logger.info("─"*80)
    
    if args.use_cache:
        logger.info("Mode cache uniquement")
        data = load_cached_data(assets)
    else:
        logger.info("Téléchargement données (avec cache)")
        fetcher = UniversalDataFetcher()
        data = fetcher.bulk_fetch(assets, save_to_cache=True)
    
    if len(data) == 0:
        logger.error("❌ Aucune donnée disponible")
        return 1
    
    logger.info(f"✅ {len(data)} datasets chargés")
    
    # ════════════════════════════════════════════════════════════════
    # ENTRAÎNEMENT
    # ════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*80)
    logger.info("PHASE 4: ENTRAÎNEMENT")
    logger.info("─"*80)
    
    trainer = ModelTrainer(config)
    
    try:
        model = trainer.train(
            env_class=UniversalTradingEnv,
            data=data,
            hyperparams=config.get_ppo_kwargs()
        )
        
        # Sauvegarder
        metadata = {
            'regime': regime_info,
            'assets': assets,
            'n_assets': len(assets)
        }
        
        trainer.save(args.output, metadata=metadata)
        
        logger.info("\n" + "="*80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("="*80)
        logger.info(f"Modèle: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Entraînement interrompu")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
