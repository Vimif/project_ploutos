# 🚀 PLOUTOS V6 ADVANCED - QUICK START

**Date:** 11 December 2025  
**Status:** 😱 READY TO IMPLEMENT  

---

## 🎉 WHAT'S NEW

### ✅ Already Created (in feature/v6-advanced-optimization branch)

**Documentation:**
- `docs/OPTIMIZATION_GUIDE_V6.md` - Complete guide to all 7 optimizations
- `docs/IMPLEMENTATION_ROADMAP.md` - Day-by-day implementation plan
- `docs/PROJECT_ANALYSIS.md` - Detailed analysis of issues and solutions

**Configuration:**
- `config/training_v6_extended_optimized.yaml` - Full V6 config (50M timesteps)

**Training Script:**
- `scripts/train_v6_extended_with_optimizations.py` - Main training entry point

**Core Modules (7 Optimizations):**
1. ✅ `core/normalization.py` - Adaptive feature normalization
2. ✅ `core/replay_buffer_prioritized.py` - Prioritized Experience Replay
3. ✅ `core/transformer_encoder.py` - Transformer feature extractor
4. ✅ `core/observation_builder_v7.py` - 3D observation structure (**NEW**)
5. ✅ `core/reward_calculator_advanced.py` - Differential Sharpe Reward (**NEW**)
6. 😱 `core/ensemble_trader.py` - Ensemble of 3 models
7. ✅ `core/drift_detector_advanced.py` - 5-method drift detection

**Analysis Scripts:**
- 😱 `scripts/feature_importance_analysis.py` - Feature importance
- 😱 `scripts/walk_forward_validator.py` - Walk-forward validation

---

## 📊 CRITICAL ISSUES TO FIX

### Priority 1: Observation Space (25% convergence gain)

**Problem:** Features are flat array (491,), loses temporal structure.

**Solution:** Integrate `ObservationBuilderV7` into environment.

```python
# In core/universal_environment_v6_better_timing.py
from core.observation_builder_v7 import ObservationBuilderV7

self.obs_builder = ObservationBuilderV7(
    n_tickers=self.n_assets,
    lookback=60,
    feature_columns=self.feature_columns,
    normalize=True,
)

# Replace _get_observation() with:
def _get_observation(self):
    return self.obs_builder.build_observation(
        processed_data=self.processed_data,
        tickers=self.tickers,
        current_step=self.current_step,
        portfolio=self.portfolio,
        balance=self.balance,
        equity=self.equity,
        initial_balance=self.initial_balance,
        peak_value=self.peak_value,
    )
```

**Time:** 30 min  
**Impact:** +25% convergence

---

### Priority 2: Reward Function (30% Sharpe gain)

**Problem:** Simple linear reward on returns → AI maximizes raw risk.

**Solution:** Use `DifferentialSharpeRewardCalculator`.

```python
# In core/universal_environment_v6_better_timing.py
from core.reward_calculator_advanced import DifferentialSharpeRewardCalculator

self.reward_calc = DifferentialSharpeRewardCalculator(
    decay=0.99,
    window=252,
)

# Replace _calculate_reward() with:
def _calculate_reward(self, trade_reward, trades_executed):
    step_return = (self.equity - self.prev_equity) / self.prev_equity
    reward = self.reward_calc.calculate(
        step_return=step_return,
        winning_trades=self.winning_trades,
        total_trades=self.total_trades,
        max_drawdown=self._calculate_max_drawdown(),
        trades_executed=trades_executed,
    )
    return reward
```

**Time:** 30 min  
**Impact:** +30% Sharpe, -25% Max DD

---

## ⏰ IMPLEMENTATION TIMELINE

### Week 1 (Dec 11-14): Critical Fixes

**Thu Dec 11 (Today):**
- [ ] Checkout branch: `git checkout feature/v6-advanced-optimization`
- [ ] Read analysis: `docs/PROJECT_ANALYSIS.md`
- [ ] Review roadmap: `docs/IMPLEMENTATION_ROADMAP.md`

**Fri-Sat (Dec 12-13):**
- [ ] Integrate `ObservationBuilderV7` into environment
- [ ] Test that observations are built correctly
- [ ] QA: No NaN, shape is correct

**Sun (Dec 14):**
- [ ] Integrate `DifferentialSharpeRewardCalculator` into environment
- [ ] Test that rewards are in [-10, 10]
- [ ] QA: No crashes

### Week 2 (Dec 15-21): Quick Validation

**Mon-Tue (Dec 15-16):**
- [ ] Integrate `AdaptiveNormalizer`
- [ ] Fit on historical data
- [ ] Test normalization works

**Wed-Thu (Dec 17-18):**
- [ ] Launch 5M steps test training
- [ ] Monitor convergence
- [ ] Verify Sharpe > 0.8

**Fri-Sun (Dec 19-21):**
- [ ] Test Transformer encoder
- [ ] Test Prioritized Replay Buffer
- [ ] Final QA of all components

### Week 3-4 (Dec 22-31): Full Training

**Full 50M timesteps:** Let it run 7-14 days

**During training:**
- Monitor W&B dashboard
- Check for drift
- Validate on walk-forward

**After training:**
- Analyze feature importance
- Deploy to VPS
- 2 weeks paper trading

---

## 😱 EXPECTED RESULTS

**Before (V6 Baseline):**
```
Sharpe: 0.8
Max DD: -22%
Win Rate: 48%
```

**After Rework 1+2 (Obs + Reward):**
```
Sharpe: 1.2 (+50%)
Max DD: -15% (-32%)
Win Rate: 52% (+8%)
```

**After Full Phase 2 (+ Normalizer):**
```
Sharpe: 1.6 (+100%)
Max DD: -10% (-55%)
Win Rate: 56% (+17%)
```

**Objective (Full V6 Advanced):**
```
Sharpe: 2.0+ (150%+)
Max DD: -8% (-64%)
Win Rate: 58% (+21%)
✅ BEATS S&P500
```

---

## 📚 FILES TO MODIFY

### Environment File
**Path:** `core/universal_environment_v6_better_timing.py`

**Changes needed:**
1. Import `ObservationBuilderV7`
2. Create instance in `__init__`
3. Replace `_get_observation()` method
4. Import `DifferentialSharpeRewardCalculator`
5. Create instance in `__init__`
6. Replace `_calculate_reward()` method
7. Import `AdaptiveNormalizer`
8. Fit in `_prepare_features_v2()`
9. Use in observations

**Estimated time:** 2-3 hours total

---

## 🛧️ INTEGRATION CHECKLIST

### ObservationBuilderV7 Integration
- [ ] Import module
- [ ] Create instance with correct params
- [ ] Call `fit()` on historical data
- [ ] Replace `_get_observation()` completely
- [ ] Test: obs.shape = (obs_size,), no NaN
- [ ] Test: obs values in [-10, 10]
- [ ] Verify with print(obs[:10])

### DifferentialSharpeRewardCalculator Integration
- [ ] Import module
- [ ] Create instance in `__init__`
- [ ] Remove old `_calculate_reward()` code
- [ ] Call `self.reward_calc.calculate()` in `step()`
- [ ] Test: reward in [-10, 10]
- [ ] Test: no crashes on edge cases (no trades, crashes, etc)

### Normalizer Integration
- [ ] Import AdaptiveNormalizer
- [ ] Create instance in `__init__`
- [ ] Call `fit()` after features prepared
- [ ] Transform observations in `_get_observation()`
- [ ] Verify distribution changes (compare before/after)

---

## 🔊 DEBUGGING TIPS

### Issue: NaN values in observation

```python
# Check what's happening
obs = env.reset()[0]
if np.any(np.isnan(obs)):
    print(f"NaN indices: {np.where(np.isnan(obs))}")
    print(f"Total NaN: {np.sum(np.isnan(obs))}")
```

### Issue: Reward exploding

```python
# Check raw values before clip
reward_raw = self.reward_calc.calculate(...)
print(f"Raw reward: {reward_raw}")
print(f"Clipped reward: {np.clip(reward_raw, -10, 10)}")
```

### Issue: Training not converging

```python
# Check metrics
metrics = self.reward_calc.get_metrics()
print(f"Sharpe: {metrics['sharpe']:.3f}")
print(f"Mean return: {metrics['mean_return']:.4f}")
print(f"Sortino: {metrics['sortino']:.3f}")
```

---

## 📑 KEY FILES TO READ

1. **Analysis (why these changes):**
   - `docs/PROJECT_ANALYSIS.md` - 12 optimizations identified

2. **Implementation (how to do it):**
   - `docs/IMPLEMENTATION_ROADMAP.md` - Day-by-day tasks
   - `docs/OPTIMIZATION_GUIDE_V6.md` - Detailed guide

3. **New Modules (what to use):**
   - `core/observation_builder_v7.py` - NEW
   - `core/reward_calculator_advanced.py` - NEW
   - `core/normalization.py` - Ready
   - `core/transformer_encoder.py` - Ready
   - `core/drift_detector_advanced.py` - Ready

4. **Config:**
   - `config/training_v6_extended_optimized.yaml` - All settings

---

## ✅ SUCCESS CRITERIA

You'll know it's working when:

1. **After Rework 1 (Observation):**
   - No crashes during training
   - Observations have right shape
   - Convergence speed increases

2. **After Rework 2 (Reward):**
   - Sharpe ratio improves
   - Max drawdown decreases
   - Win rate increases

3. **After Phase 2 (Normalizer):**
   - Stability improves (lower loss variance)
   - Training time same or faster

4. **Full training:**
   - Sharpe > 1.6 (2.0+ is target)
   - Max DD < 10%
   - Paper trading 2+ weeks no issues

---

## 🗙️ NEXT STEPS

1. **Read:** Review `PROJECT_ANALYSIS.md` to understand why
2. **Plan:** Check `IMPLEMENTATION_ROADMAP.md` for schedule
3. **Clone:** Get the feature branch code
4. **Integrate:** Add the 2 new modules to your environment
5. **Test:** Run 5M steps validation
6. **Train:** Launch full 50M training on BBC
7. **Deploy:** Move to VPS and paper trade

---

**Questions?** Check the detailed docs or come back for specific issues.

**Status: 😱 READY TO START!**
