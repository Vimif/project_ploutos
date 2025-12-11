# 🐍 PLOUTOS V6 - TROUBLESHOOTING GUIDE

**Having issues with V6?** Find your problem below.

---

## 🔧 INSTALLATION ISSUES

### Error: "read: Illegal option -n"

**What it means:**
You're using `make` but your shell is `sh`, not `bash`. The `read` command is incompatible.

**Solution:**
Use the bash script instead:
```bash
bash scripts/quick_start.sh
```

Or run commands directly:
```bash
python scripts/apply_v6_patches.py
python scripts/verify_v6_installation.py
python scripts/train_v6_extended_with_optimizations.py --config config/training_v6_extended_optimized.yaml --output models/v6_test_5m --device cuda:0
```

---

### Error: "No such file or directory: scripts/apply_v6_patches.py"

**What it means:**
You're not in the right directory.

**Solution:**
```bash
cd /root/ploutos/project_ploutos
python scripts/apply_v6_patches.py
```

---

### Error: "ModuleNotFoundError: No module named 'core.observation_builder_v7'"

**What it means:**
The V6 modules haven't been installed yet, OR patches weren't applied.

**Solution:**
```bash
# First, apply patches
python scripts/apply_v6_patches.py

# Then verify
python scripts/verify_v6_installation.py

# If still failing, check you're on the right branch
git branch
# Should show: * feature/v6-advanced-optimization
```

---

### Error: "Backup file already exists"

**What it means:**
You ran the patch script twice. It's safe - it will skip.

**Solution:**
Just run it again:
```bash
python scripts/apply_v6_patches.py
```

The script checks if patches are already applied and skips them.

---

### Error: "Permission denied: scripts/apply_v6_patches.py"

**What it means:**
The script isn't executable.

**Solution:**
```bash
chmod +x scripts/apply_v6_patches.py
python scripts/apply_v6_patches.py
```

Or just use python directly:
```bash
python scripts/apply_v6_patches.py
```

---

## 🚀 TRAINING ISSUES

### Error: "CUDA out of memory"

**What it means:**
Your GPU doesn't have enough VRAM for the batch size.

**Solution #1 - Reduce batch size:**
```bash
# Edit config/training_v6_extended_optimized.yaml
training:
  n_envs: 16          # Reduce from 32 to 16
  batch_size: 1024    # Reduce from 4096 to 2048
```

Then retrain:
```bash
python scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m
```

**Solution #2 - Check GPU memory:**
```bash
nvidia-smi
```

You need at least 24GB VRAM. If you have less:
- Reduce `n_envs` to 8
- Reduce `batch_size` to 512
- Reduce `learning_rate` to 5e-5

---

### Error: "RuntimeError: Expected input_h to be >= 0 but got -N"

**What it means:**
Your observation space dimensions are wrong. This usually means the environment wasn't patched correctly.

**Solution:**
```bash
# Restore the backup
make restore-backup
# Or manually:
BACKUP=$(ls -t core/universal_environment_v6_better_timing.py.backup_* 2>/dev/null | head -1)
cp "$BACKUP" core/universal_environment_v6_better_timing.py

# Re-apply patches
python scripts/apply_v6_patches.py

# Verify
python scripts/verify_v6_installation.py
```

---

### Error: "ValueError: num_envs must be > 0"

**What it means:**
Your config file has `n_envs: 0` or a negative number.

**Solution:**
Edit `config/training_v6_extended_optimized.yaml`:
```yaml
training:
  n_envs: 32  # Must be > 0, typically 8, 16, 32, 64
```

---

### Training is VERY slow (< 100K steps/hour)

**What it means:**
CPU bottleneck or not using GPU.

**Solution #1 - Check GPU is being used:**
```bash
watch -n 2 nvidia-smi
```

You should see GPU utilization > 50%.

If GPU % is low:
```bash
# Increase parallelism
training:
  n_envs: 64  # Increase from 32
  batch_size: 4096  # Increase
```

**Solution #2 - Check CPU:**
```bash
top  # or htop
```

If CPU is pinned at 100%, reduce `n_envs` to 16.

---

### Training crashes with "Segmentation fault"

**What it means:**
Memory issue or corrupted data.

**Solution:**
```bash
# 1. Check RAM
free -h

# 2. Kill any other processes
killall python

# 3. Reduce n_envs
training:
  n_envs: 8  # Start small

# 4. Clear cache
rm -rf ~/.cache/pip

# 5. Reinstall dependencies
pip install --upgrade --force-reinstall torch stable-baselines3

# 6. Try again
python scripts/train_v6_extended_with_optimizations.py ...
```

---

### Error: "ImportError: cannot import name 'TransformerFeatureExtractor'"

**What it means:**
The Transformer module wasn't found.

**Solution:**
```bash
# Check the file exists
ls -la core/transformer_encoder.py

# If not, you're not on the right branch
git checkout feature/v6-advanced-optimization
git pull

# If it exists, check imports
grep -n "class TransformerFeatureExtractor" core/transformer_encoder.py
```

---

## 📋 LOGGING & MONITORING ISSUES

### Error: "tail: cannot open for reading: No such file or directory"

**What it means:**
No logs exist yet. Training hasn't started.

**Solution:**
Start training first:
```bash
python scripts/train_v6_extended_with_optimizations.py --output models/v6_test_5m
```

Then in another terminal:
```bash
tail -f logs/train_v6_extended_*.log
```

---

### Logs show: "Warning: Feature column X not found"

**What it means:**
The features calculated don't match what the environment expects.

**Solution:**
Check `_prepare_features_v2` in your environment. All feature names must exist:
```bash
grep "feature_columns =" core/universal_environment_v6_better_timing.py
```

Make sure all these columns are actually calculated in your data.

---

### No logs are being written

**What it means:**
Logging might be misconfigured or directory doesn't exist.

**Solution:**
```bash
# Create logs directory
mkdir -p logs/tensorboard

# Verify training is running
ps aux | grep train_v6_extended

# Check if process is there
pgrep -f train_v6_extended_with_optimizations.py
```

---

## 🗙️ RESTORATION & RECOVERY

### "I want to restore the original environment file"

**Solution:**
```bash
# Find the backup
ls -lah core/universal_environment_v6_better_timing.py.backup_*

# Restore the most recent one
BACKUP=$(ls -t core/universal_environment_v6_better_timing.py.backup_* 2>/dev/null | head -1)
cp "$BACKUP" core/universal_environment_v6_better_timing.py

echo "✅ Restored from: $BACKUP"
```

Or use make:
```bash
make restore-backup
```

---

### "The patches didn't apply correctly"

**What it means:**
The automatic patch script had issues.

**Solution:**
```bash
# 1. Restore backup
BACKUP=$(ls -t core/universal_environment_v6_better_timing.py.backup_* 2>/dev/null | head -1)
cp "$BACKUP" core/universal_environment_v6_better_timing.py

# 2. Check what went wrong
echo "Checking patch application..."
grep "from core.observation_builder_v7 import" core/universal_environment_v6_better_timing.py

# 3. Re-apply
python scripts/apply_v6_patches.py

# 4. Verify
python scripts/verify_v6_installation.py
```

---

## 👁️ VERIFICATION ISSUES

### Error: "All checks FAILED"

**What it means:**
Something is seriously wrong with the installation.

**Solution - Nuclear Option:**
```bash
# 1. Start fresh
git checkout feature/v6-advanced-optimization
git reset --hard HEAD

# 2. Clean caches
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf ~/.cache/pip

# 3. Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements_training.txt

# 4. Apply patches
python scripts/apply_v6_patches.py

# 5. Verify
python scripts/verify_v6_installation.py
```

---

## 📈 PERFORMANCE ISSUES

### "Sharpe ratio is still low after patches"

**What it means:**
Either:
1. Training hasn't converged yet (normal for first few hours)
2. Patches weren't applied correctly
3. Config is suboptimal

**Solution:**
```bash
# 1. Verify patches
python scripts/verify_v6_installation.py

# 2. Check logs for convergence
tail -50 logs/train_v6_extended_*.log | grep -i sharpe

# 3. If Sharpe increases over time, it's working
# It typically takes 1-2 million steps to see improvement

# 4. If stuck at 0.6-0.7, check config
cat config/training_v6_extended_optimized.yaml | grep -A5 "training:"
```

---

### "Training seems to diverge (loss keeps increasing)"

**What it means:**
Learning rate is too high or data is corrupted.

**Solution:**
```bash
# 1. Reduce learning rate
training:
  learning_rate: 5e-5  # Reduce from 1e-4

# 2. Reduce batch size
  batch_size: 2048    # Reduce from 4096

# 3. Stop training
killall python

# 4. Remove corrupted checkpoint
rm -rf models/v6_test_5m/

# 5. Retrain
python scripts/train_v6_extended_with_optimizations.py ...
```

---

## 🧠 ADVANCED DEBUGGING

### "I want to see detailed error messages"

**Solution:**
Run with verbose logging:
```bash
PYTHONUNBUFFEREED=1 python -u scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m \
    2>&1 | tee logs/detailed_debug.log
```

Then check the log:
```bash
tail -100 logs/detailed_debug.log
```

---

### "I want to debug the environment"

**Solution - Test environment in isolation:**
```python
# Create debug_env.py
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
import numpy as np

# Initialize environment
env = UniversalTradingEnvV6BetterTiming(
    # Your config here
)

# Test observation
obs = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"First 5 values: {obs[:5]}")

# Test step
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"Reward: {reward}")
print(f"Done: {done}")
```

Run it:
```bash
python debug_env.py
```

---

## 📞 GETTING HELP

**Before asking for help, provide:**

1. Your error message (exact output)
2. Your setup:
   ```bash
   uname -a  # OS
   python --version
   nvidia-smi  # GPU
   ```
3. What you were doing:
   ```bash
   make train-v6-test  # or whatever command
   ```
4. Recent logs:
   ```bash
   tail -50 logs/*.log
   ```

---

## ✅ STILL STUCK?

```bash
# 1. Nuclear reset
git checkout feature/v6-advanced-optimization
git reset --hard HEAD
git pull origin

# 2. Fresh install
pip uninstall -y torch stable-baselines3 gymnasium
pip install torch stable-baselines3 gymnasium

# 3. Apply patches
python scripts/apply_v6_patches.py

# 4. Verify
python scripts/verify_v6_installation.py

# 5. Test
python scripts/train_v6_extended_with_optimizations.py --output models/v6_test_5m --timesteps 100000
```

If it still doesn't work, check GitHub issues or ask for help. 🚀
