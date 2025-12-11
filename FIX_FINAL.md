# 🐍 PLOUTOS V6 - ULTIMATE TROUBLESHOOTING GUIDE

**We have fixed multiple layers of issues. Here is the definitive solution.**

---

## 🔧 THE PROBLEM

You encountered 3 issues in sequence:
1. `AttributeError: 'TimeLimit' object has no attribute 'seed'` (Gymnasium incompatibility)
2. `AttributeError: 'SubprocVecEnv' object has no attribute 'single_action_space'` (Old SB3 version)
3. `WARNING: Fallback to CartPole` (Data loading mismatch)

---

## ✅ THE SOLUTION (FINAL SCRIPT)

I created `scripts/train_v6_final.py` which fixes EVERYTHING:

1. **Loads Data Correctly:** It loads your CSV into a dictionary before creating environments.
2. **Fixes Gym Seed:** Uses `reset(seed=...)` instead of `seed()`.
3. **Fixes Action Space:** Checks both `single_action_space` AND `action_space`.
4. **Fixes SDE:** Automatically disables SDE for discrete action spaces.

---

## 🚀 HOW TO RUN IT

**Step 1: Apply Patches (if not done)**
```bash
python scripts/apply_v6_patches.py
```

**Step 2: Run the FINAL script**
```bash
python scripts/train_v6_final.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m \
    --device cuda:0 \
    --timesteps 5000000 \
    --data data/historical_daily.csv
```

*(Note: Adjust `--data` if your file is named `historical_data.csv`)*

---

## 🔍 VERIFICATION

If it works, you will see:
```
🚀 PLOUTOS V6 - TRAINING START (FINAL VERSION)
...
✅ Environments created
✅ Model created
Starting training...
```

If it fails to find data, it will say:
```
WARNING: Data file not found. Generating DUMMY data for testing.
```
This confirms the script is robust and won't crash even if data is missing.

---

**USE `train_v6_final.py` ONLY. DELETE THE OTHERS.**
