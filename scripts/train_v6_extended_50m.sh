#!/bin/bash
# scripts/train_v6_extended_50m.sh
# ✅ ENTRAÎNEMENT V6 EXTENDED 50M - Convergence Complète

set -e

echo ""
echo "========================================"
echo "🚀 ENTRAÎNEMENT V6 EXTENDED 50M"
echo "========================================"
echo ""
echo "✅ Objectif: Convergence complète pour meilleur BUY timing"
echo ""
echo "⚙️  Configuration:"
echo "  • Total timesteps: 50M (3.3x plus que V6 standard)"
echo "  • Features: 85 par ticker (Features V2)"
echo "  • Tickers: 15"
echo "  • Early stopping: Activé"
echo ""
echo "⏱️  Durée estimée: 15-18h sur RTX 3080"
echo ""
echo "🚨 Recommandation: Lancer en mode background (--nohup)"
echo ""

# Paths
PROJECT_ROOT="/root/ai-factory/tmp/project_ploutos"
VENV_PATH="/root/ai-factory/venv"

cd "$PROJECT_ROOT"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA détecté"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  Pas de GPU détecté, utilisation CPU (LENT!)" 
    echo ""
fi

# Activer venv
echo "🐍 Activation virtualenv..."
source "$VENV_PATH/bin/activate"

# Créer dossiers
mkdir -p models/v6_extended_50m_checkpoints
mkdir -p models/v6_extended_50m_best
mkdir -p models/v6_extended_50m
mkdir -p logs/v6_extended_50m
mkdir -p logs/v6_extended_50m_eval
mkdir -p runs/v6_extended_50m

echo "✅ Dossiers créés"
echo ""

# Parse arguments
NOHUP=false
for arg in "$@"; do
    if [ "$arg" == "--nohup" ]; then
        NOHUP=true
    fi
done

if [ "$NOHUP" = true ]; then
    echo "========================================"
    echo "🚀 DÉMARRAGE EN MODE BACKGROUND"
    echo "========================================"
    echo ""
    echo "  • Logs: logs/v6_extended_50m/training_$(date +%Y%m%d_%H%M%S).log"
    echo "  • Pour suivre: tail -f logs/v6_extended_50m/training_*.log"
    echo "  • Pour arrêter: pkill -f train_v6_extended_50m"
    echo "  • Pour monitorer GPU: watch -n 1 nvidia-smi"
    echo ""
    echo "⏱️  Durée: ~15-18h"
    echo ""
    
    nohup python -u training/train_v6_extended_50m.py \
        --config config/training_config_v6_extended_50m.yaml \
        > "logs/v6_extended_50m/training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    PID=$!
    echo "✅ Entraînement lancé (PID: $PID)"
    echo ""
    echo "📊 Monitorer avec:"
    echo "  tail -f logs/v6_extended_50m/training_*.log"
    echo "  tensorboard --logdir runs/v6_extended_50m/"
    echo ""
else
    echo "========================================"
    echo "🚀 DÉMARRAGE ENTRAÎNEMENT"
    echo "========================================"
    echo ""
    echo "💻 Mode interactif"
    echo "Ctrl+C pour interrompre"
    echo ""
    echo "⚠️  Attention: Entraînement de 15-18h !"
    echo "Recommandé: Relancer avec --nohup"
    echo ""
    
    python -u training/train_v6_extended_50m.py \
        --config config/training_config_v6_extended_50m.yaml
fi

echo ""
echo "========================================"
echo "✅ SCRIPT TERMINÉ"
echo "========================================"
echo ""
