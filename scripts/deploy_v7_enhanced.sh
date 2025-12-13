#!/bin/bash

#🤖 PLOUTOS V7.1 ENHANCED - Full Deployment Script
# 
# Deploy l'ensemble du système V7.1 avec optimisation des hyperparamétrages.
#
# Usage:
#   chmod +x scripts/deploy_v7_enhanced.sh
#   ./scripts/deploy_v7_enhanced.sh [--skip-optimization] [--quick]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"
MODELS_DIR="$PROJECT_DIR/models"

echo ""
echo "========================================================================="
echo "🤖 PLOUTOS V7.1 ENHANCED - Deployment Pipeline"
echo "========================================================================="
echo ""

# Check GPU
echo "🔍 Vérification GPU..."
if python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null; then
    echo "✅ GPU check OK"
else
    echo "⚠️  No GPU detected, using CPU"
fi

echo ""
echo "========================================================================="
echo "PHASE 1: Prepare Environment"
echo "========================================================================="necho ""

mkdir -p $LOGS_DIR $MODELS_DIR
echo "✅ Directories created"

# Check Python dependencies
echo "📅 Checking dependencies..."
required_packages=("torch" "optuna" "yfinance" "sklearn")
for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "  ✅ $package"
    else
        echo "  ⚠️  $package not found! Install with: pip install $package"
    fi
done

echo ""
echo "========================================================================="
echo "PHASE 2: Hyperparameter Optimization"
echo "========================================================================="
echo ""

if [[ $1 == "--skip-optimization" ]]; then
    echo "⏹  Skipping optimization (--skip-optimization flag set)"
else
    TRIALS=50
    TIMEOUT=3600
    
    if [[ $1 == "--quick" ]]; then
        TRIALS=10
        TIMEOUT=600
        echo "⚡ Quick mode: 10 trials, 600s timeout"
    else
        echo "🔌 Running Optuna optimization (50 trials, ~1 hour)..."
    fi
    
    echo ""
    echo "   Optimizing Momentum Expert..."
    python3 $SCRIPT_DIR/v7_hyperparameter_optimizer.py \
        --expert momentum \
        --trials $TRIALS \
        --timeout $TIMEOUT \
        --tickers "NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NFLX" \
        2>&1 | tee "$LOGS_DIR/optimize_momentum.log"
    
    echo ""
    echo "   Optimizing Reversion Expert..."
    python3 $SCRIPT_DIR/v7_hyperparameter_optimizer.py \
        --expert reversion \
        --trials $TRIALS \
        --timeout $TIMEOUT \
        --tickers "NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NFLX" \
        2>&1 | tee "$LOGS_DIR/optimize_reversion.log"
    
    echo ""
    echo "   Optimizing Volatility Expert..."
    python3 $SCRIPT_DIR/v7_hyperparameter_optimizer.py \
        --expert volatility \
        --trials $TRIALS \
        --timeout $TIMEOUT \
        --tickers "NVDA,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NFLX" \
        2>&1 | tee "$LOGS_DIR/optimize_volatility.log"
    
    echo ""
    echo "🎉 Optimization complete!"
    echo "💾 Results saved to: $LOGS_DIR/v7_*_optimization.json"
fi

echo ""
echo "========================================================================="
echo "PHASE 3: Validation"
echo "========================================================================="
echo ""

echo "🔍 Testing prediction pipeline..."
python3 $SCRIPT_DIR/v7_ensemble_predict.py --ticker NVDA 2>&1 | tee "$LOGS_DIR/test_prediction.log"

echo ""
echo "✅ Validation OK"

echo ""
echo "========================================================================="
echo "🌟 DEPLOYMENT COMPLETE"
echo "========================================================================="
echo ""
echo "📄 Next Steps:"
echo "   1. Review optimization results: cat $LOGS_DIR/v7_*_optimization.json"
echo "   2. Train enhanced models: python scripts/train_v7_enhanced.py"
echo "   3. Deploy dashboard: python web/app.py"
echo "   4. Run predictions: python scripts/v7_ensemble_predict.py --ticker AAPL"
echo ""
echo "💾 Logs available at: $LOGS_DIR/"
echo ""
