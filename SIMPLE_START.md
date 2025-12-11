# 🚀 PLOUTOS V6 - COPY-PASTE GUIDE

**This is the SIMPLEST way to get started.**  
Just copy-paste the commands below. Nothing else.

---

## 😱 Step 1: Get the code (1 minute)

```bash
cd /root/ploutos/project_ploutos
git fetch origin
git checkout feature/v6-advanced-optimization
```

---

## 🔧 Step 2: Apply patches (1 minute)

```bash
python scripts/apply_v6_patches.py
```

You'll see:
```
✅ Backup created
✅ Patches applied
✅ SUCCESS!
```

---

## 🐍 Step 3: Verify (2 minutes)

```bash
python scripts/verify_v6_installation.py
```

You'll see:
```
✅ ALL CHECKS PASSED!
```

---

## 🚀 Step 4: Start training (automatic, 24 hours)

**Choose ONE of these:

### Option A: Using Makefile (if you have bash)

```bash
make train-v6-test
```

When asked "Continue? (y/n)" → type `y` and press Enter.

### Option B: Using bash script (if Makefile doesn't work)

```bash
bash scripts/quick_start.sh
```

Then select option `3` (Run quick 5M test)

### Option C: Direct command (if both above fail)

```bash
mkdir -p models logs/tensorboard
python scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m \
    --device cuda:0 \
    --timesteps 5000000
```

---

## 🎅 Step 5: Wait 24 hours

That's it. The training happens automatically.

If you want to watch:

```bash
# Terminal 1: Watch logs
tail -f logs/train_v6_extended_*.log

# Terminal 2: Monitor GPU
watch -n 2 nvidia-smi
```

---

## 🌟 Step 6: After 24h, check results

```bash
tail logs/train_v6_extended_*.log
```

Look for:
```
Final Sharpe: X.XX
Max Drawdown: -X%
Win Rate: X%
```

If Sharpe > 0.8 = SUCCESS!

Then run full training:

```bash
mkdir -p models logs/tensorboard
python scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_extended_full \
    --device cuda:0 \
    --timesteps 50000000
```

This takes 7-14 days. Go enjoy life 🎴

---

## 📠 Need Help?

**Issue?** Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Want details?** Read: [START_HERE.md](START_HERE.md)

**Want everything?** Read: [INSTALL_V6.md](INSTALL_V6.md)

---

## ✅ That's ALL

You're done. Go copy-paste the commands above. 🚀
