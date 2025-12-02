#!/bin/bash

echo "🏭 PLOUTOS AI FACTORY - TRAINING LAUNCHER"
echo "=========================================="

# Vérification GPU
echo "🎮 Vérification GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Activation environnement
source venv/bin/activate

# Installation dépendances
echo "📦 Installation dépendances..."
pip install -r project_ploutos/requirements.txt

# Lancement entraînement
echo "🚀 Lancement entraînement..."
cd project_ploutos

# Choix du secteur (argument optionnel)
SECTOR=${1:-TECH}

echo "🎯 Secteur: $SECTOR"
python ai_trainer.py $SECTOR 2>&1 | tee ../logs/training_${SECTOR}_$(date +%Y%m%d_%H%M%S).log
