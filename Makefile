.PHONY: help setup-v6 test-v6 train-v6-test train-v6-full validate clean

# 🚀 PLOUTOS V6 ADVANCED - MAKEFILE
# Quick commands for setup, testing, and training

help:
	@echo ""
	@echo "🚀 PLOUTOS V6 ADVANCED - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Configuration:"
	@echo "  make setup-v6              Apply V6 patches automatically"
	@echo "  make test-patches          Verify patches were applied"
	@echo "  make restore-backup        Restore from backup (if patches failed)"
	@echo ""
	@echo "Training:"
	@echo "  make train-v6-test         Quick 5M steps test (24h)"
	@echo "  make train-v6-full         Full 50M steps training (7-14 days)"
	@echo "  make validate              Run walk-forward validation"
	@echo "  make analyze-features      Analyze feature importance"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-vps            Copy model to VPS"
	@echo "  make paper-trade           Start paper trading"
	@echo ""
	@echo "Monitoring:"
	@echo "  make watch-logs            Watch training logs"
	@echo "  make gpu-monitor           Monitor GPU usage"
	@echo "  make tensorboard           Launch TensorBoard"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 Remove old backups and logs"
	@echo ""

setup-v6:
	@echo ""
	@echo "🚀 PLOUTOS V6 SETUP - Step 1: Apply Patches"
	@echo "================================================"
	@echo ""
	@echo "This will:"
	@echo "1. Backup your original environment file"
	@echo "2. Apply all V6 patches automatically"
	@echo "3. Verify the patches are correct"
	@echo ""
	@read -p "Continue? (y/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python scripts/apply_v6_patches.py; \
	fi

test-patches:
	@echo ""
	@echo "🐍 Testing V6 patches..."
	@echo "================================================"
	@echo ""
	@python -c "
	import sys
	try:
		from core.observation_builder_v7 import ObservationBuilderV7
		print('✅ ObservationBuilderV7 - OK')
		from core.reward_calculator_advanced import DifferentialSharpeRewardCalculator
		print('✅ DifferentialSharpeRewardCalculator - OK')
		from core.normalization import AdaptiveNormalizer
		print('✅ AdaptiveNormalizer - OK')
		from core.transformer_encoder import TransformerFeatureExtractor
		print('✅ TransformerFeatureExtractor - OK')
		from core.drift_detector_advanced import ComprehensiveDriftDetector
		print('✅ ComprehensiveDriftDetector - OK')
		print('')
		print('✅ All V6 modules loaded successfully!')
		sys.exit(0)
		except Exception as e:
			print(f'❌ Error: {e}')
			sys.exit(1)
	"

restore-backup:
	@echo ""
	@echo "⚠️  Restoring from backup..."
	@echo "================================================"
	@echo ""
	@bash -c '
		BACKUP=$$(ls -t core/universal_environment_v6_better_timing.py.backup_* 2>/dev/null | head -1);
		if [ -z "$$BACKUP" ]; then
			echo "❌ No backup found!";
			exit 1;
		fi;
		cp "$$BACKUP" core/universal_environment_v6_better_timing.py;
		echo "✅ Restored from: $$BACKUP";
	'

train-v6-test:
	@echo ""
	@echo "🚀 PLOUTOS V6 - Quick Test (5M Steps)"
	@echo "================================================"
	@echo ""
	@echo "This will:"
	@echo "1. Train for 5M steps (fast validation)"
	@echo "2. Check if patches work correctly"
	@echo "3. Monitor convergence"
	@echo ""
	@echo "Expected time: 12-24 hours"
	@echo "Target Sharpe: > 0.8"
	@echo ""
	@read -p "Continue? (y/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		mkdir -p models logs/tensorboard; \
		python scripts/train_v6_extended_with_optimizations.py \
			--config config/training_v6_extended_optimized.yaml \
			--output models/v6_test_5m \
			--device cuda:0; \
	fi

train-v6-full:
	@echo ""
	@echo "🚀 PLOUTOS V6 - FULL TRAINING (50M Steps)"
	@echo "================================================"
	@echo ""
	@echo "This will:"
	@echo "1. Train for 50M steps (full training)"
	@echo "2. Use curriculum learning (3 stages)"
	@echo "3. Integrate all 7 optimizations"
	@echo ""
	@echo "Expected time: 7-14 days continuous"
	@echo "Target Sharpe: > 1.6"
	@echo ""
	@echo "⚠️  Make sure:"
	@echo "   - 5M test passed (Sharpe > 0.8)"
	@echo "   - Enough disk space (50+ GB)"
	@echo "   - GPU available"
	@echo ""
	@read -p "Continue? (y/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		mkdir -p models logs/tensorboard; \
		python scripts/train_v6_extended_with_optimizations.py \
			--config config/training_v6_extended_optimized.yaml \
			--output models/v6_extended_full \
			--device cuda:0; \
	fi

validate:
	@echo ""
	@echo "🐍 Running Walk-Forward Validation..."
	@echo "================================================"
	@echo ""
	@mkdir -p results
	@python scripts/walk_forward_validator.py \
		--model models/v6_extended_full/stage_3_final.zip \
		--data data/historical_daily.csv \
		--output results/walk_forward_results.json

analyze-features:
	@echo ""
	@echo "👁️ Analyzing Feature Importance..."
	@echo "================================================"
	@echo ""
	@mkdir -p results
	@python scripts/feature_importance_analysis.py \
		--model models/v6_extended_full/stage_3_final.zip \
		--output results/feature_importance.json

deploy-vps:
	@echo ""
	@echo 🚀 Deploying to VPS..."
	@echo "================================================"
	@echo ""
	@read -p "VPS Host (default: root@VPS): " VPS_HOST; \
	VPS_HOST=$${VPS_HOST:-root@VPS}; \
	echo "Deploying to: $$VPS_HOST"; \
	rsync -avz models/v6_extended_full/ $$VPS_HOST:/root/ploutos/models/v6_extended_full/

paper-trade:
	@echo ""
	@echo "📋 Starting Paper Trading..."
	@echo "================================================"
	@echo ""
	@read -p "VPS Host (default: root@VPS): " VPS_HOST; \
	VPS_HOST=$${VPS_HOST:-root@VPS}; \
	ssh $$VPS_HOST "sudo systemctl restart ploutos-trader-v2"; \
	echo "✅ Paper trading started!"

watch-logs:
	@echo ""
	@echo "📃 Watching training logs..."
	@echo "================================================"
	@echo "Press Ctrl+C to stop"
	@echo ""
	@tail -f logs/train_v6_extended_*.log 2>/dev/null || echo "No logs found"

gpu-monitor:
	@echo ""
	@echo "📊 GPU Monitoring (Press Ctrl+C to stop)"
	@echo "================================================"
	@watch -n 2 nvidia-smi

tensorboard:
	@echo ""
	@echo "📊 Launching TensorBoard..."
	@echo "================================================"
	@echo "Open your browser at: http://localhost:6006"
	@echo ""
	@tensorboard --logdir=logs/tensorboard

clean:
	@echo ""
	@echo "🗻 Cleaning old files..."
	@echo "================================================"
	@echo ""
	@find . -name "*.backup_202*" -mtime +7 -delete && echo "✅ Deleted old backups"
	@find logs -name "*.log" -mtime +30 -delete && echo "✅ Deleted old logs"
	@echo "✅ Cleanup complete"

get-latest-model:
	@echo ""
	@echo "🗂 Getting latest model..."
	@echo "================================================"
	@echo ""
	@bash -c '
		MODEL=$$(ls -dt models/*/stage_3_final.zip 2>/dev/null | head -1);
		if [ -z "$$MODEL" ]; then
			echo "❌ No models found!";
			exit 1;
		fi;
		echo "✅ Latest model: $$MODEL";
		ls -lh "$$MODEL";
	'

.DEFAULT_GOAL := help
