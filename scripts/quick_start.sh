#!/bin/bash

# 🚀 PLOUTOS V6 - QUICK START SCRIPT
# No Makefile dependency - Works on any system

set -e  # Exit on error

echo ""
echo "🚀 PLOUTOS V6 ADVANCED - QUICK START"
echo "===================================="
echo ""

echo "What do you want to do?"
echo ""
echo "1. Apply V6 patches (automatic)"
echo "2. Verify patches"
echo "3. Run quick 5M test"
echo "4. Run full 50M training"
echo "5. Watch logs"
echo "6. Monitor GPU"
echo ""
printf "Select (1-6): "
read CHOICE

case $CHOICE in
    1)
        echo ""
        echo "🔧 Applying V6 patches..."
        python scripts/apply_v6_patches.py
        ;;
    2)
        echo ""
        echo "🐍 Verifying patches..."
        python scripts/verify_v6_installation.py
        ;;
    3)
        echo ""
        echo "🚀 Starting 5M test (24h)..."
        mkdir -p models logs/tensorboard
        python scripts/train_v6_extended_with_optimizations.py \
            --config config/training_v6_extended_optimized.yaml \
            --output models/v6_test_5m \
            --device cuda:0 \
            --timesteps 5000000
        ;;
    4)
        echo ""
        echo "⚠️  IMPORTANT: Did the 5M test pass?"
        printf "Continue? (y/n): "
        read CONFIRM
        if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
            echo ""
            echo "🚀 Starting 50M full training (7-14 days)..."
            mkdir -p models logs/tensorboard
            python scripts/train_v6_extended_with_optimizations.py \
                --config config/training_v6_extended_optimized.yaml \
                --output models/v6_extended_full \
                --device cuda:0 \
                --timesteps 50000000
        else
            echo "Cancelled."
        fi
        ;;
    5)
        echo ""
        echo "📃 Watching logs (Ctrl+C to stop)..."
        echo ""
        tail -f logs/train_v6_extended_*.log 2>/dev/null || echo "No logs found"
        ;;
    6)
        echo ""
        echo "📊 Monitoring GPU (Ctrl+C to stop)..."
        echo ""
        watch -n 2 nvidia-smi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
echo ""
