"""
Logger centralisé pour Ploutos
Usage: logger = PloutosLogger().get_logger(__name__)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

class PloutosLogger:
    """Singleton logger pour tout le projet"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure le logger principal"""
        
        # Créer dossier logs
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Fichier log avec timestamp
        log_file = log_dir / f"ploutos_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Logger principal
        self.logger = logging.getLogger('ploutos')
        self.logger.setLevel(logging.DEBUG)
        
        # Éviter duplication handlers
        if self.logger.handlers:
            return
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler fichier (DEBUG)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Handler console (INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Format console plus simple
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Ajouter handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialisé → {log_file}")
    
    def get_logger(self, name: str = None):
        """
        Récupère un logger pour un module spécifique
        
        Args:
            name: Nom du module (utiliser __name__)
            
        Returns:
            Logger configuré
        """
        if name:
            return logging.getLogger(f'ploutos.{name}')
        return self.logger
