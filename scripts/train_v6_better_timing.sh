#!/bin/bash
# scripts/train_v6_better_timing.sh
# ✅ ENTRAÎNEMENT V6 - MEILLEUR TIMING

set -e

echo ""
echo "========================================"
echo "🚀 ENTRAÎNEMENT V6 - BETTER TIMING"
echo "========================================"
echo ""
echo "✅ Objectif: Résoudre 'buy high' (85% mauvais timing)"
echo ""
echo "✅ Features V2 optimisées:"
echo "  • Support/Resistance dynamiques"
echo "  • Mean Reversion signals"
echo "  • Volume confirmation"
echo "  • Price action patterns"
echo "  • Divergences RSI/Prix"
echo "  • Bollinger squeeze"
echo "  • Entry score composite"
echo ""
echo "  • 60+ features par ticker (vs 37 avant)"
echo ""

# Paths
PROJECT_ROOT="/root/ai-factory/tmp/project_ploutos"
VENV_PATH="/root/ai-factory/venv"

cd "$PROJECT_ROOT"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA détecté"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  Pas de GPU détecté, utilisation CPU"
    echo ""
fi

# Activer venv
echo "🐍 Activation virtualenv..."
source "$VENV_PATH/bin/activate"

# Créer dossiers
mkdir -p models/v6_better_timing_checkpoints
mkdir -p models/v6_better_timing_best
mkdir -p logs/v6_better_timing
mkdir -p runs/v6_better_timing

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
    echo "  • Logs: logs/v6_better_timing/training_$(date +%Y%m%d_%H%M%S).log"
    echo "  • Pour suivre: tail -f logs/v6_better_timing/training_*.log"
    echo "  • Pour arrêter: pkill -f train_v6_better_timing"
    echo ""
    
    nohup python -u training/train_v6_better_timing.py \
        --config config/training_config_v6_better_timing.yaml \
        > "logs/v6_better_timing/training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    PID=$!
    echo "✅ Entraînement lancé (PID: $PID)"
    echo ""
else
    echo "========================================"
    echo "🚀 DÉMARRAGE ENTRAÎNEMENT"
    echo "========================================"
    echo ""
    echo "💻 Mode interactif"
    echo "Ctrl+C pour interrompre"
    echo ""
    
    python -u training/train_v6_better_timing.py \
        --config config/training_config_v6_better_timing.yaml
fi

echo ""
echo "========================================"
echo "✅ SCRIPT TERMINÉ"
echo "========================================"
echo ""
