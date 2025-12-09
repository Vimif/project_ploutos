#!/bin/bash
# scripts/launch_training_ultimate.sh
# Lancement Entraînement V3 ULTIMATE

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 LANCEMENT ENTRAÎNEMENT V3 ULTIMATE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Vérifier qu'on est sur BBC (machine GPU)
if [ ! -d "/root/ai-factory" ]; then
    echo -e "${RED}❌ Cette machine n'est pas BBC (GPU)${NC}"
    echo -e "${YELLOW}Utiliser: ssh bbc${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Machine BBC détectée${NC}"

# Vérifier GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi non trouvé${NC}"
    exit 1
fi

echo -e "${YELLOW}💻 GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Aller dans le projet
cd /root/ai-factory/tmp/project_ploutos || {
    echo -e "${RED}❌ Projet non trouvé${NC}"
    exit 1
}

echo -e "${GREEN}✅ Répertoire projet OK${NC}"

# Activer virtualenv
if [ -f "/root/ai-factory/venv/bin/activate" ]; then
    source /root/ai-factory/venv/bin/activate
    echo -e "${GREEN}✅ Virtualenv activé${NC}"
else
    echo -e "${YELLOW}⚠️  Virtualenv non trouvé${NC}"
fi

# Vérifier dépendances
echo -e "${YELLOW}📎 Vérification dépendances...${NC}"

if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch non installé${NC}"
    exit 1
fi

if ! python3 -c "import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')" 2>/dev/null; then
    echo -e "${RED}❌ Stable-Baselines3 non installé${NC}"
    exit 1
fi

if ! python3 -c "import ta" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  'ta' (Technical Analysis) non installé${NC}"
    echo -e "${YELLOW}Installation...${NC}"
    pip install -q ta
fi

echo -e "${GREEN}✅ Dépendances OK${NC}"

# Créer dossiers
mkdir -p models/v3_checkpoints
mkdir -p models/v3_best
mkdir -p models/v3_ultimate
mkdir -p logs/v3_eval
mkdir -p runs/v3_ultimate

echo -e "${GREEN}✅ Dossiers créés${NC}"

# Afficher config
if [ -f "config/training_config_v3.yaml" ]; then
    echo -e "${YELLOW}📄 Configuration:${NC}"
    cat config/training_config_v3.yaml | grep -E "(total_timesteps|n_envs|learning_rate|net_arch)" | head -10
    echo ""
fi

# Options
CONFIG_PATH="config/training_config_v3.yaml"
NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --nohup)
            NOHUP=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Option inconnue: $1${NC}"
            exit 1
            ;;
    esac
done

# LANCEMENT
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 DÉMARRAGE ENTRAÎNEMENT${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$NOHUP" = true ]; then
    echo -e "${YELLOW}Mode NOHUP (arrière-plan)${NC}"
    nohup python3 training/train_v3_ultimate.py --config "$CONFIG_PATH" > logs/training_v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    PID=$!
    echo -e "${GREEN}✅ Process lancé (PID: $PID)${NC}"
    echo -e "${YELLOW}Logs: tail -f logs/training_v3_*.log${NC}"
else
    python3 training/train_v3_ultimate.py --config "$CONFIG_PATH"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ SCRIPT TERMINÉ${NC}"
echo -e "${BLUE}========================================${NC}"
