#!/bin/bash
# scripts/install_training_deps.sh
# Installation des dépendances pour entraînement V3

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📦 INSTALLATION DÉPENDANCES V3${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 non installé${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Vérifier pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 non installé${NC}"
    exit 1
fi

echo -e "${GREEN}✅ pip3 installé${NC}"

# Mettre à jour pip
echo -e "${YELLOW}🔄 Mise à jour pip...${NC}"
pip3 install --upgrade pip -q

# Installer PyTorch avec CUDA si disponible
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}🔥 GPU détecté, installation PyTorch avec CUDA...${NC}"
    
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo -e "${GREEN}✅ CUDA Version: $CUDA_VERSION${NC}"
    
    if [ "$(echo "$CUDA_VERSION >= 11.8" | bc)" -eq 1 ]; then
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip3 install torch torchvision torchaudio
    fi
else
    echo -e "${YELLOW}⚠️  Pas de GPU, installation PyTorch CPU only${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✅ PyTorch installé${NC}"

# Installer requirements
if [ -f "requirements_training.txt" ]; then
    echo -e "${YELLOW}📚 Installation des dépendances...${NC}"
    pip3 install -r requirements_training.txt
    echo -e "${GREEN}✅ Dépendances installées${NC}"
else
    echo -e "${YELLOW}⚠️  requirements_training.txt non trouvé${NC}"
    echo -e "${YELLOW}Installation manuelle...${NC}"
    
    pip3 install stable-baselines3
    pip3 install gymnasium
    pip3 install ta
    pip3 install scipy
    pip3 install pandas numpy
    pip3 install yfinance
    pip3 install wandb
    pip3 install tensorboard
    pip3 install PyYAML
    pip3 install alpaca-py
    pip3 install python-dotenv
    pip3 install tqdm
fi

# Vérifier installations
echo -e "${YELLOW}🔍 Vérification...${NC}"
echo ""

python3 << 'EOF'
import sys

packages = [
    ('torch', 'PyTorch'),
    ('stable_baselines3', 'Stable-Baselines3'),
    ('gymnasium', 'Gymnasium'),
    ('ta', 'Technical Analysis'),
    ('scipy', 'SciPy'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('yfinance', 'yfinance'),
    ('wandb', 'Weights & Biases'),
    ('yaml', 'PyYAML')
]

all_ok = True

for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:25s} {version}")
    except ImportError:
        print(f"❌ {name:25s} NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\n✅ Toutes les dépendances sont installées")
else:
    print("\n❌ Certaines dépendances manquent")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ INSTALLATION TERMINÉE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${GREEN}Vous pouvez maintenant lancer:${NC}"
    echo -e "${YELLOW}bash scripts/launch_training_ultimate.sh${NC}"
    echo ""
else
    echo -e "${RED}❌ Échec installation${NC}"
    exit 1
fi
