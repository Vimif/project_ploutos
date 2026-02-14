#!/bin/bash
# start_training.sh
# Lance l'entraînement optimisé pour serveur HPC (256 Cores)

# 1. Empêcher l'explosion de threads (256 process x 1 thread au lieu de 256x256)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# 2. Augmenter la limite de fichiers ouverts (si possible)
ulimit -n 65535 2>/dev/null

echo "🚀 Starting High-Performance Training on RunPod..."
echo "   - Threads per process forced to 1 to avoid resource exhaustion"
echo "   - Walking Forward on 5 years of history"
echo "   - Ensemble of 10 models"

# 3. Lancer le pipeline
python3 scripts/run_pipeline.py --config config/training_config_v8.yaml --auto-scale --ensemble 10
