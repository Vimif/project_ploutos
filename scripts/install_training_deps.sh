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

# Détecter GPU (optionnel)
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo -e "${GREEN}✅ GPU détecté: $GPU_NAME${NC}"
    fi
fi

# Installer PyTorch
if [ "$HAS_GPU" = true ]; then
    echo -e "${YELLOW}🔥 Installation PyTorch avec support CUDA...${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}💻 Installation PyTorch CPU only...${NC}"
    echo -e "${YELLOW}(L'entraînement sera plus lent)${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✅ PyTorch installé${NC}"

# Installer requirements
if [ -f "requirements_training.txt" ]; then
    echo -e "${YELLOW}📚 Installation des dépendances depuis requirements...${NC}"
    pip3 install -r requirements_training.txt -q
    echo -e "${GREEN}✅ Dépendances installées${NC}"
else
    echo -e "${YELLOW}⚠️  requirements_training.txt non trouvé${NC}"
    echo -e "${YELLOW}Installation manuelle des packages essentiels...${NC}"
    
    pip3 install -q stable-baselines3
    pip3 install -q gymnasium
    pip3 install -q ta
    pip3 install -q scipy
    pip3 install -q pandas numpy
    pip3 install -q yfinance
    pip3 install -q wandb
    pip3 install -q tensorboard
    pip3 install -q PyYAML
    pip3 install -q alpaca-py
    pip3 install -q python-dotenv
    pip3 install -q tqdm
    
    echo -e "${GREEN}✅ Packages essentiels installés${NC}"
fi

# Vérifier installations
echo ""
echo -e "${YELLOW}🔍 Vérification des installations...${NC}"
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
print("Package                    Version")
print("="*50)

for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        status = "✅"
        print(f"{status} {name:25s} {version}")
    except ImportError:
        status = "❌"
        print(f"{status} {name:25s} NOT INSTALLED")
        all_ok = False

print("="*50)

# Vérifier CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n🔥 CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("\n💻 CPU only (pas de CUDA détecté)")
except:
    pass

if all_ok:
    print("\n✅ Toutes les dépendances sont installées avec succès")
    sys.exit(0)
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
    echo -e "${YELLOW}  bash scripts/launch_training_ultimate.sh${NC}"
    echo ""
    echo -e "${YELLOW}Ou en mode arrière-plan:${NC}"
    echo -e "${YELLOW}  bash scripts/launch_training_ultimate.sh --nohup${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Échec de l'installation${NC}"
    echo -e "${YELLOW}Vérifiez les erreurs ci-dessus${NC}"
    exit 1
fi
