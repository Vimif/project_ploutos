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
    echo -e "${YELLOW}⚠️  Répertoire /root/ai-factory non trouvé${NC}"
    echo -e "${YELLOW}Entraînement possible mais optimisé pour GPU${NC}"
fi

# Détecter GPU (optionnel)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU NVIDIA détecté${NC}"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${YELLOW}💻 GPU: $GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠️  Pas de GPU détecté, utilisation CPU${NC}"
    echo -e "${YELLOW}L'entraînement sera plus lent${NC}"
fi

echo ""

# Aller dans le projet
if [ -d "/root/ai-factory/tmp/project_ploutos" ]; then
    cd /root/ai-factory/tmp/project_ploutos
else
    # Essayer depuis le répertoire courant
    if [ ! -f "training/train_v3_ultimate.py" ]; then
        echo -e "${RED}❌ Projet non trouvé${NC}"
        echo -e "${YELLOW}Lancer depuis la racine du projet${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Répertoire projet OK${NC}"

# Activer virtualenv (optionnel)
if [ -f "/root/ai-factory/venv/bin/activate" ]; then
    source /root/ai-factory/venv/bin/activate
    echo -e "${GREEN}✅ Virtualenv activé${NC}"
else
    echo -e "${YELLOW}⚠️  Virtualenv non trouvé, utilisation Python système${NC}"
fi

# Vérifier dépendances
echo -e "${YELLOW}📎 Vérification dépendances...${NC}"

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}❌ PyTorch non installé${NC}"
    echo -e "${YELLOW}Lancer: bash scripts/install_training_deps.sh${NC}"
    exit 1
fi

if ! python3 -c "import stable_baselines3" 2>/dev/null; then
    echo -e "${RED}❌ Stable-Baselines3 non installé${NC}"
    echo -e "${YELLOW}Lancer: bash scripts/install_training_deps.sh${NC}"
    exit 1
fi

if ! python3 -c "import ta" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  'ta' (Technical Analysis) non installé${NC}"
    echo -e "${YELLOW}Installation automatique...${NC}"
    pip install -q ta
fi

echo -e "${GREEN}✅ Dépendances OK${NC}"

# Afficher versions
echo -e "${YELLOW}📚 Versions:${NC}"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import stable_baselines3; print(f'  SB3: {stable_baselines3.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
echo ""

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
    cat config/training_config_v3.yaml | grep -E "(total_timesteps|n_envs|learning_rate)" | head -5
    echo ""
else
    echo -e "${YELLOW}⚠️  Config non trouvée, utilisation valeurs par défaut${NC}"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH    Chemin vers fichier config YAML"
            echo "  --nohup          Lancer en arrière-plan"
            echo "  --help           Afficher cette aide"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Option inconnue: $1${NC}"
            echo "Utiliser --help pour voir les options"
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
    LOG_FILE="logs/training_v3_$(date +%Y%m%d_%H%M%S).log"
    echo -e "${YELLOW}📄 Mode arrière-plan${NC}"
    echo -e "${YELLOW}Logs: $LOG_FILE${NC}"
    
    nohup python3 training/train_v3_ultimate.py --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo -e "${GREEN}✅ Process lancé (PID: $PID)${NC}"
    echo ""
    echo -e "${YELLOW}Commandes utiles:${NC}"
    echo -e "  tail -f $LOG_FILE"
    echo -e "  ps aux | grep train_v3"
    echo -e "  kill $PID"
else
    echo -e "${YELLOW}💻 Mode interactif${NC}"
    echo -e "${YELLOW}Ctrl+C pour interrompre${NC}"
    echo ""
    
    python3 training/train_v3_ultimate.py --config "$CONFIG_PATH"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ SCRIPT TERMINÉ${NC}"
echo -e "${BLUE}========================================${NC}"
