#!/bin/bash
# scripts/setup_runpod.sh
# One-click environment setup for RunPod / Lambda Labs / Vast.ai
# Tested on clean Ubuntu 22.04 + CUDA 11.8/12.x images

# Stop on error
set -e

echo "🚀 Starting Ploutos Cloud Setup..."

# 1. System Dependencies (htop, screen, git, etc.)
echo "📦 Installing system dependencies..."
apt-get update -y
apt-get install -y git htop screen vim curl unzip build-essential python3-venv

# 2. Python Environment (Assuming Python 3.10+ pre-installed)
echo "🐍 Setting up Python Environment..."
python3 -m pip install --upgrade pip

# 3. Clone Repository (if not present)
REPO_DIR="project_ploutos"
REPO_URL="https://github.com/Vimif/project_ploutos.git" 

if [ -d "$REPO_DIR" ]; then
    echo "✅ Repository found. Updating..."
    cd $REPO_DIR
    git pull
else
    echo "📥 Cloning repository from $REPO_URL..."
    # Note: On private repos, you will need to enter credentials or use SSH keys manually first
    git clone $REPO_URL
    cd $REPO_DIR
fi

# 4. Install Project Dependencies
echo "📚 Installing Python dependencies..."
# Core dependencies
pip install -r requirements.txt
# Training specific dependencies (stable-baselines3, torch, etc.)
pip install -r requirements_training.txt
# Install package in editable mode
pip install -e .

# 5. Create Essential Directories
echo "📂 Creating project directories..."
mkdir -p models logs data reports data_cache

# 6. Verify GPU Availability
echo "🖥️  Verifying GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 7. Create Launch Scripts for Convenience
echo "📝 Creating launch helper scripts..."

# Launch PPO Walk-Forward
cat <<EOT > run_ppo.sh
#!/bin/bash
echo "🚀 Launching PPO Walk-Forward..."
nohup python3 training/train_walk_forward.py --config config/training_config_v8.yaml > logs/train_ppo_wfa.log 2>&1 &
echo "✅ Started in background. Logs: tail -f logs/train_ppo_wfa.log"
EOT
chmod +x run_ppo.sh

# Launch RecurrentPPO Walk-Forward
cat <<EOT > run_lstm.sh
#!/bin/bash
echo "🚀 Launching RecurrentPPO (LSTM) Walk-Forward..."
nohup python3 training/train_walk_forward.py --config config/training_config_v8.yaml --recurrent > logs/train_lstm_wfa.log 2>&1 &
echo "✅ Started in background. Logs: tail -f logs/train_lstm_wfa.log"
EOT
chmod +x run_lstm.sh

# Launch Ensemble
cat <<EOT > run_ensemble.sh
#!/bin/bash
echo "🚀 Launching Ensemble Walk-Forward (N=3)..."
nohup python3 training/train_walk_forward.py --config config/training_config_v8.yaml --ensemble 3 > logs/train_ensemble.log 2>&1 &
echo "✅ Started in background. Logs: tail -f logs/train_ensemble.log"
EOT
chmod +x run_ensemble.sh

# Launch Optuna
cat <<EOT > run_optuna.sh
#!/bin/bash
echo "🚀 Launching Optuna Hyperparameter Optimization..."
nohup python3 scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 > logs/optuna.log 2>&1 &
echo "✅ Started in background. Logs: tail -f logs/optuna.log"
EOT
chmod +x run_optuna.sh


echo "✅ Setup Complete!"
echo ""
echo "🔥 Ready to train! Use the helper scripts created:"
echo "   ./run_ppo.sh       - Baseline PPO"
echo "   ./run_lstm.sh      - Recurrent PPO (LSTM)"
echo "   ./run_ensemble.sh  - Ensemble (3 models)"
echo "   ./run_optuna.sh    - Hyperparameter Tuning"
echo ""
echo "👉 You can monitor training with: tail -f logs/<log_file>"
