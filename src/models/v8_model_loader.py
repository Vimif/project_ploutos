#!/usr/bin/env python3
"""
📦 PLOUTOS V8 - MODEL LOADER

Module utilitaire pour charger les modèles V8 avec chemins absolus
Résout les problèmes de chemins relatifs

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Retourne la racine du projet (là où se trouve le dossier models/)
    Cherche dans plusieurs emplacements possibles
    """
    # 1. Essayer depuis le fichier actuel
    current_file = Path(__file__).resolve()
    
    # Remonter depuis src/models/ vers racine
    potential_root = current_file.parent.parent.parent
    if (potential_root / 'models').exists():
        return potential_root
    
    # 2. Essayer depuis CWD
    cwd = Path.cwd()
    if (cwd / 'models').exists():
        return cwd
    
    # 3. Essayer parent de CWD
    if (cwd.parent / 'models').exists():
        return cwd.parent
    
    # 4. Chemins VPS spécifiques
    vps_paths = [
        Path('/root/ploutos/project_ploutos'),
        Path('/root/ai-factory/tmp/project_ploutos'),
        Path.home() / 'ploutos' / 'project_ploutos',
        Path.home() / 'ai-factory' / 'tmp' / 'project_ploutos'
    ]
    
    for path in vps_paths:
        if path.exists() and (path / 'models').exists():
            return path
    
    # Fallback: retourner CWD
    return cwd


def get_model_path(model_name: str) -> str:
    """
    Retourne le chemin absolu vers un modèle
    
    Args:
        model_name: Nom du fichier (ex: 'v8_lightgbm_intraday.pkl')
    
    Returns:
        Chemin absolu vers le modèle
    """
    root = get_project_root()
    model_path = root / 'models' / model_name
    
    # Debug
    print(f"📁 Project root: {root}")
    print(f"🎯 Model path: {model_path}")
    print(f"✅ Exists: {model_path.exists()}")
    
    return str(model_path)


def list_available_models() -> dict:
    """
    Liste tous les modèles disponibles dans models/
    """
    root = get_project_root()
    models_dir = root / 'models'
    
    if not models_dir.exists():
        return {'error': f'Dossier models/ non trouvé dans {root}'}
    
    models = {
        'v8': [],
        'v7': [],
        'other': []
    }
    
    for file in models_dir.glob('*'):
        if file.is_file():
            name = file.name
            size = file.stat().st_size / 1024  # KB
            
            if name.startswith('v8_'):
                models['v8'].append({'name': name, 'size_kb': round(size, 1)})
            elif name.startswith('v7_'):
                models['v7'].append({'name': name, 'size_kb': round(size, 1)})
            else:
                models['other'].append({'name': name, 'size_kb': round(size, 1)})
    
    return models


if __name__ == '__main__':
    print("🔍 PLOUTOS V8 - MODEL LOADER TEST")
    print("=" * 70 + "\n")
    
    # Test 1: Project root
    root = get_project_root()
    print(f"✅ Project root: {root}\n")
    
    # Test 2: Model paths
    print("🎯 Testing model paths:\n")
    for model_name in ['v8_lightgbm_intraday.pkl', 'v8_xgboost_weekly.pkl']:
        path = get_model_path(model_name)
        print()
    
    # Test 3: List models
    print("\n" + "="*70)
    print("📂 Available models:\n")
    models = list_available_models()
    
    if 'error' in models:
        print(f"❌ {models['error']}")
    else:
        for version, files in models.items():
            if files:
                print(f"\n{version.upper()}:")
                for f in files:
                    print(f"  - {f['name']:40s} ({f['size_kb']:>8.1f} KB)")
