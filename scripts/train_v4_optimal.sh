#!/bin/bash
# scripts/train_v4_optimal.sh
# ✅ ENTRAÎNEMENT V4 OPTIMAL - Configuration pour Performance Maximale

set -e

echo ""
echo "========================================"
echo "🚀 ENTRAÎNEMENT V4 OPTIMAL"
echo "========================================"
echo ""
echo "✅ Améliorations vs V3:"
echo "  • Entropy coef: 0.01 → 0.08 (+700% exploration)"
echo "  • Max trades/day: 3 → 10 (+233% liberté)"
echo "  • Buy pct: 15% → 25% (+67% capital)"
echo "  • Timesteps: 10M → 20M (+100% apprentissage)"
echo "  • Reward bonus trades réussis"
echo "  • Pénalités réduites"
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

# Vérifier deps
echo "✅ Vérification dépendances..."
python -c "import torch; import stable_baselines3; import gymnasium; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available__()}'); print(f'SB3: {stable_baselines3.__version__}')"
echo ""

# Créer dossiers
mkdir -p models/v4_optimal_checkpoints
mkdir -p models/v4_optimal_best
mkdir -p logs/v4_optimal
mkdir -p runs/v4_optimal

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
    echo "  • Logs: logs/v4_optimal/training_$(date +%Y%m%d_%H%M%S).log"
    echo "  • Pour suivre: tail -f logs/v4_optimal/training_*.log"
    echo "  • Pour arrêter: pkill -f train_v4_optimal"
    echo ""
    
    nohup python -u training/train_v4_optimal.py \
        --config config/training_config_v4_optimal.yaml \
        > "logs/v4_optimal/training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
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
    
    python -u training/train_v4_optimal.py \
        --config config/training_config_v4_optimal.yaml
fi

echo ""
echo "========================================"
echo "✅ SCRIPT TERMINÉ"
echo "========================================"
echo ""
