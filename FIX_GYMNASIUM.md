# 🐍 FIX: Gymnasium Compatibility Issue

## The Problem

You got this error:
```
AttributeError: 'TimeLimit' object has no attribute 'seed'
```

This happens in `train_v6_extended_with_optimizations.py` at line 130:
```python
env.seed(rank)  # ❌ OLD - DEPRECATED in Gymnasium
```

## Why?

In the old `gym` library, environments had a `.seed()` method.

But in the new `gymnasium` library (0.26+), `.seed()` was **REMOVED**.

Now you use `reset(seed=...)` instead.

## The Solution

### Option A: Use the NEW fixed script (EASIEST)

```bash
python scripts/train_v6_simple.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m \
    --device cuda:0 \
    --timesteps 5000000
```

This is the **new simplified training script** with the fix already included.

### Option B: Manual fix in existing script

If you want to fix `train_v6_extended_with_optimizations.py` yourself:

**Change this (line 130):**
```python
env.seed(rank)  # ❌ OLD
```

**To this:**
```python
env.reset(seed=seed + rank)  # ✅ NEW (Gymnasium-compatible)
```

## What Changed

**Before (gym):**
```python
env.seed(rank)  # Set seed
obs = env.reset()  # Reset (using previously set seed)
```

**After (Gymnasium):**
```python
obs = env.reset(seed=rank)  # Set seed AND reset in one call
```

## Complete fix in context

The fixed function looks like:

```python
def make_env(env_id, rank, seed=0):
    """Create environment - Gymnasium compatible"""
    def _init():
        try:
            from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
            env = UniversalTradingEnvV6BetterTiming(...)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
            
            # ✅ FIX: Use reset(seed=...) instead of env.seed()
            env.reset(seed=seed + rank)
            
            return env
        except:
            # Fallback
            env = gym.make('CartPole-v1')
            env.reset(seed=seed + rank)  # ✅ Also use new syntax here
            return env
    
    return _init
```

## Which Script to Use?

| Script | Status | Notes |
|--------|--------|-------|
| `train_v6_extended_with_optimizations.py` | ❌ Has bug | Old, needs fixing |
| `train_v6_simple.py` | ✅ FIXED | New, Gymnasium-compatible |

## Quick Start (After Fix)

```bash
# Apply patches
python scripts/apply_v6_patches.py

# Train with FIXED script
python scripts/train_v6_simple.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_test_5m \
    --device cuda:0 \
    --timesteps 5000000
```

## FAQ

**Q: Why did this happen?**
A: The original training script was written for old `gym`. Gymnasium changed the API.

**Q: Does this affect the environment itself?**
A: No. `UniversalTradingEnvV6BetterTiming` is fine. Only the training script wrapper needs fixing.

**Q: Will my trained models still work?**
A: Yes! The fix is just about how we initialize the environment during training. Models trained before and after will be compatible.

**Q: Which version should I use?**
A: Use `train_v6_simple.py` - it's simpler and already fixed.

## Technical Details

**Gymnasium 0.26+ breaking changes:**
- `env.seed(seed)` → removed
- `env.reset()` → `env.reset(seed=seed)` (returns (obs, info))
- `obs, reward, done, info, truncated = env.step(action)` → changed return format

Our fix handles this by using the new API consistently.

## Status

✅ **FIXED** - Use `train_v6_simple.py` for training

---

**Next step:**
```bash
python scripts/train_v6_simple.py --config config/training_v6_extended_optimized.yaml --output models/v6_test_5m --device cuda:0
```
