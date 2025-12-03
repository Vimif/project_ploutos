#!/bin/bash
# scripts/setup_infrastructure.sh

echo "🏗️  SETUP INFRASTRUCTURE PLOUTOS"
echo "=================================="

# Détection OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    OS="unknown"
fi

echo "🖥️  OS détecté: $OS"

# 1. Python & venv
echo ""
echo "1️⃣  Configuration Python..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION"

# Créer venv si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer
source venv/bin/activate

# 2. Dépendances
echo ""
echo "2️⃣  Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Structure des dossiers
echo ""
echo "3️⃣  Création de la structure..."

mkdir -p data/{models,logs,trade_history}
mkdir -p temp_models
mkdir -p tensorboard

echo "✅ Dossiers créés"

# 4. NFS Check (si applicable)
echo ""
echo "4️⃣  Vérification NFS..."

if [ -d "/mnt/shared" ]; then
    echo "✅ NFS monté: /mnt/shared"
    
    # Créer structure sur NFS
    mkdir -p /mnt/shared/ploutos_data/{models,logs,trade_history}
    
    # Symlink si pas déjà fait
    if [ ! -L "data" ]; then
        rm -rf data
        ln -s /mnt/shared/ploutos_data data
        echo "✅ Symlink créé: data -> /mnt/shared/ploutos_data"
    fi
else
    echo "⚠️  NFS non monté (utilisation locale)"
fi

# 5. Git config
echo ""
echo "5️⃣  Configuration Git..."

if [ ! -d ".git" ]; then
    git init
    echo "✅ Git initialisé"
else
    echo "✅ Git déjà configuré"
fi

# 6. Tests
echo ""
echo "6️⃣  Tests basiques..."

python3 -c "from config.settings import BASE_DIR; print(f'✅ Config OK: {BASE_DIR}')"
python3 -c "from core.features import FeatureCalculator; print('✅ Core OK')"

# 7. Résumé
echo ""
echo "=================================="
echo "✅ SETUP TERMINÉ"
echo "=================================="
echo ""
echo "📋 Prochaines étapes:"
echo "   1. Activer l'environnement: source venv/bin/activate"
echo "   2. Entraîner des modèles: python scripts/train_models.py"
echo "   3. Lancer le trader: python scripts/run_trader.py"
echo "   4. Dashboard: streamlit run ui/dashboard.py"
echo ""
