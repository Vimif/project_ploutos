# core/utils.py
"""Fonctions utilitaires"""
import logging
import torch
import gc
from datetime import datetime

def setup_logging(name: str, log_file: str = None):
    """
    Configurer le logging
    
    Args:
        name: Nom du logger
        log_file: Chemin du fichier de log (optionnel)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        from config.settings import LOGS_DIR
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def cleanup_resources(*objects):
    """
    Nettoyer des ressources (envs, modèles, etc.)
    
    Usage:
        cleanup_resources(env, model, autre_objet)
    """
    for obj in objects:
        if obj is not None:
            # Essayer .close()
            if hasattr(obj, 'close'):
                try:
                    obj.close()
                except:
                    pass
            
            # Supprimer
            try:
                del obj
            except:
                pass
    
    # GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()

def get_gpu_info():
    """Obtenir infos GPU"""
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'available': True,
        'name': torch.cuda.get_device_name(0),
        'memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
    }

def format_duration(seconds: float):
    """Formater une durée en secondes"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def timestamp():
    """Timestamp formaté"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
