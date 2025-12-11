# 🛣️ PLOUTOS V6 - ROADMAP D'IMPLÉMENTATION DÉTAILLÉE

**Date:** 11 Décembre 2025  
**Durée Estimée:** 4 semaines (Phase 1-3)  
**Priorité:** CRITIQUE  

---

## 🎯 OBJECTIF FINAL

```
Sharpe Ratio: 0.8 → 2.0+ (150% improvement)
Max Drawdown: -22% → -8% (64% reduction)
Win Rate: 48% → 58% (20% improvement)
```

---

## ⏰ PHASE 1: CRITICAL FIXES (Semaine 1 - Jeudi 11 à Dimanche 14 Décembre)

### Jour 1 (Jeudi 11 Décembre) - Setup

**Tâche 1.1:** Merge la branche V6 Advanced
```bash
cd /root/ploutos/project_ploutos
git fetch origin
git checkout feature/v6-advanced-optimization
git merge main --no-ff -m "merge: V6 advanced optimizations base"
```
**Temps:** 15 min  
**Responsable:** Toi  
**Status:** ✅ FAIT (déjà sur la branche)

---

### Jour 2-3 (Vendredi-Samedi 12-13 Décembre) - REWORK #1: Observation Space

**Tâche 2.1:** Créer ObservationBuilder nouveau

**Fichier:** `core/observation_builder_v7.py`

```python
import numpy as np
from typing import Dict
import pandas as pd

class ObservationBuilderV7:
    """Construction d'observation 3D optimisée pour Transformer."""
    
    def __init__(self, n_tickers: int, lookback: int, feature_columns: list):
        self.n_tickers = n_tickers
        self.lookback = lookback
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)
    
    def build_observation(
        self,
        processed_data: Dict[str, pd.DataFrame],
        tickers: list,
        current_step: int,
        portfolio: Dict,
        balance: float,
        equity: float,
        initial_balance: float,
        peak_value: float,
    ) -> np.ndarray:
        """
        Construire observation 3D: (n_tickers, lookback, n_features)
        Compatible avec Transformer!
        """
        
        # Part 1: Temporal features (3D)
        obs_temporal = np.zeros(
            (self.n_tickers, self.lookback, self.n_features),
            dtype=np.float32
        )
        
        for ticker_idx, ticker in enumerate(tickers):
            df = processed_data[ticker]
            
            # Historique des features
            for t in range(self.lookback):
                idx = current_step - self.lookback + t
                
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    features = row[self.feature_columns].values
                else:
                    features = np.zeros(self.n_features)
                
                # Normalize and clip
                features = np.nan_to_num(features, nan=0.0, posinf=10, neginf=-10)
                features = np.clip(features, -10, 10)
                
                obs_temporal[ticker_idx, t, :] = features
        
        # Part 2: Portfolio state (1D)
        obs_portfolio = []
        for ticker in tickers:
            price = self._get_current_price(processed_data[ticker], current_step)
            position_value = portfolio[ticker] * price if price > 0 else 0
            position_pct = position_value / (equity + 1e-8)
            obs_portfolio.append(position_pct)
        
        obs_portfolio = np.array(obs_portfolio, dtype=np.float32)
        obs_portfolio = np.clip(obs_portfolio, 0, 1)
        
        # Part 3: Account state (1D)
        cash_pct = np.clip(balance / (equity + 1e-8), 0, 1)
        total_return = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        drawdown = np.clip((peak_value - equity) / (peak_value + 1e-8), 0, 1)
        
        obs_account = np.array([cash_pct, total_return, drawdown], dtype=np.float32)
        
        # Combine: flatten temporal + portfolio + account
        obs_flat = np.concatenate([
            obs_temporal.flatten(),  # (n_tickers * lookback * n_features)
            obs_portfolio,           # (n_tickers)
            obs_account,             # (3)
        ])
        
        return obs_flat.astype(np.float32)
    
    def _get_current_price(self, df: pd.DataFrame, current_step: int) -> float:
        if current_step >= len(df):
            return df.iloc[-1]['Close']
        price = df.iloc[current_step]['Close']
        return price if price > 0 and not np.isnan(price) else df['Close'].median()
```

**Tâche 2.2:** Intégrer dans l'environment

**Fichier à modifier:** `core/universal_environment_v6_better_timing.py`

```python
# Dans __init__
from core.observation_builder_v7 import ObservationBuilderV7

self.obs_builder = ObservationBuilderV7(
    n_tickers=self.n_assets,
    lookback=config['environment']['lookback_period'],
    feature_columns=self.feature_columns,
)

# Mettre à jour observation_space
obs_size = (
    self.n_assets * self.lookback * len(self.feature_columns) +
    self.n_assets +
    3
)

# Dans _get_observation()
def _get_observation(self) -> np.ndarray:
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

**Temps:** 2-3 heures  
**QA:** Vérifier que obs.shape = (n_obs,) et pas de NaN  
**Impact:** +25% convergence

---

### Jour 4 (Dimanche 14 Décembre) - REWORK #2: Reward Function

**Tâche 3.1:** Créer DifferentialSharpeReward

**Fichier:** `core/reward_calculator_advanced.py`

```python
import numpy as np
from collections import deque
from typing import Tuple

class DifferentialSharpeRewardCalculator:
    """
    Differential Sharpe Ratio - Récompense l'IA pour améliorer le Sharpe
    à chaque step, plutôt que maximiser le retour brut.
    
    Formule: DSR = (dB * A_prev - 0.5 * dA * B_prev) / (variance + eps)
    où A = mean(returns), B = mean(returns²)
    """
    
    def __init__(self, decay: float = 0.99, window: int = 252):
        self.decay = decay  # EMA decay
        self.A = 0.0        # EMA of returns
        self.B = 0.0        # EMA of returns²
        self.returns_history = deque(maxlen=window)
    
    def calculate(
        self,
        step_return: float,
        winning_trades: int,
        total_trades: int,
        max_drawdown: float,
        trades_executed: int,
    ) -> float:
        """
        Calculer reward à chaque step.
        
        Args:
            step_return: Return du step (prix_actuel - prix_prev) / prix_prev
            winning_trades: Nombre de trades gagnants
            total_trades: Nombre total de trades
            max_drawdown: Drawdown max depuis début
            trades_executed: Nombre de trades executés ce step
        
        Returns:
            Reward scalaire
        """
        
        # 1. Store return
        self.returns_history.append(step_return)
        
        # 2. Update EMA
        prev_A = self.A
        prev_B = self.B
        
        self.A = self.decay * self.A + (1 - self.decay) * step_return
        self.B = self.decay * self.B + (1 - self.decay) * (step_return ** 2)
        
        # 3. Calculate variance
        variance = self.B - (self.A ** 2)
        if variance < 1e-8:
            dsr_reward = 0.0
        else:
            # Differential Sharpe Ratio
            dsr = (self.B * prev_A - 0.5 * self.A * prev_B) / ((variance + 1e-8) ** 1.5)
            dsr_reward = np.tanh(dsr / 2) * 2  # Scale to [-2, 2]
        
        # 4. Sortino bonus (penalizes downside)
        if len(self.returns_history) > 10:
            down_returns = [r for r in self.returns_history if r < 0]
            if down_returns:
                down_vol = np.std(down_returns)
                sortino = step_return / (down_vol + 1e-8) if step_return > 0 else -0.5
                sortino_bonus = np.clip(sortino * 0.1, -0.5, 0.5)
            else:
                sortino_bonus = 0.2 if step_return > 0 else 0.0
        else:
            sortino_bonus = 0.0
        
        # 5. Win rate bonus
        if total_trades > 5:
            win_rate = winning_trades / total_trades
            win_rate_bonus = (win_rate - 0.50) * 0.5 if win_rate > 0.50 else (win_rate - 0.50) * 0.3
        else:
            win_rate_bonus = 0.0
        
        # 6. Drawdown penalty
        dd_penalty = 0.0
        if max_drawdown < -0.10:
            dd_penalty = min(max_drawdown * 2, -0.1)  # Severe penalty if DD > 10%
        
        # 7. Overtrading penalty
        if trades_executed > 3:
            trade_penalty = (trades_executed - 2) * 0.05
        else:
            trade_penalty = 0.0
        
        # 8. Combine
        total_reward = (
            0.6 * dsr_reward +       # Main signal: improve Sharpe
            0.2 * sortino_bonus +    # Downside protection
            0.1 * win_rate_bonus +   # Consistency
            0.05 * dd_penalty +      # Risk penalty
            -0.05 * trade_penalty    # Reduce noise
        )
        
        return np.clip(total_reward, -10, 10)
    
    def get_metrics(self) -> dict:
        """Return current Sharpe-related metrics."""
        if len(self.returns_history) == 0:
            return {'sharpe': 0.0, 'mean_return': 0.0, 'std_return': 0.0}
        
        returns = np.array(list(self.returns_history))
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0
        
        return {
            'sharpe': sharpe,
            'mean_return': mean_ret,
            'std_return': std_ret,
            'max_return': returns.max(),
            'min_return': returns.min(),
        }
```

**Tâche 3.2:** Intégrer dans environment

**Fichier à modifier:** `core/universal_environment_v6_better_timing.py`

```python
# Dans __init__
from core.reward_calculator_advanced import DifferentialSharpeRewardCalculator

self.reward_calc = DifferentialSharpeRewardCalculator(
    decay=0.99,
    window=252,
)

# Dans step()
def step(self, actions):
    # ... existing code ...
    
    step_return = (self.equity - prev_equity) / prev_equity if prev_equity > 0 else 0
    
    reward = self.reward_calc.calculate(
        step_return=step_return,
        winning_trades=self.winning_trades,
        total_trades=self.total_trades,
        max_drawdown=self._calculate_max_drawdown(),
        trades_executed=trades_executed,
    )
    
    # ... rest of step ...
```

**Temps:** 2 heures  
**QA:** Vérifier que reward est dans [-10, 10]  
**Impact:** +30% Sharpe, -25% Max DD

---

## 📅 PHASE 2: OPTIMISATIONS (Semaine 2 - Lundi 15 à Dimanche 21 Décembre)

### Jour 1-2 (Lundi-Mardi 15-16) - Normalizer Integration

**Tâche 4.1:** Implémenter AdaptiveNormalizer dans environment

```python
# Dans __init__
from core.normalization import AdaptiveNormalizer

self.normalizer = AdaptiveNormalizer(
    feature_groups={
        'technical': 512,
        'ml_features': 400,
        'market_regime': 100,
        'portfolio_state': 150,
    }
)

# Dans _prepare_features_v2()
self.normalizer.fit({ticker: df for ticker, df in self.processed_data.items()})
self.normalizer.save("models/normalizer_v6_extended.pkl")

# Dans _get_observation()
obs = self.normalizer.transform(obs_dict)
```

**Temps:** 1-2 heures  
**Impact:** +15-25%

---

### Jour 3-4 (Mercredi-Jeudi 17-18) - Test 5M Steps

**Tâche 5.1:** Lancer training test rapide

```bash
cd /root/ai-factory  # BBC machine
python scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_extended_test_5m \
    --device cuda:0
```

**Temps:** 12-24 heures (exécution)  
**Validation:** 
- Sharpe devrait être > 0.8
- Max DD < 20%
- Convergence stable

---

### Jour 5-7 (Vendredi-Dimanche 19-21) - Transformer + PER

**Tâche 6.1:** Vérifier Transformer fonctionne
- Code déjà existant: `core/transformer_encoder.py`
- Juste tester dans training script

**Tâche 6.2:** Vérifier PER fonctionne
- Code déjà existant: `core/replay_buffer_prioritized.py`
- Intégrer dans PPO (optionnel, SB3 n'a pas PER natif)

**Temps:** 2-3 heures (testing seulement)

---

## 🚀 PHASE 3: FULL TRAINING (Semaine 3-4)

### Jour 1-7 (Semaine 3)

**Tâche 7.1:** Lancer full 50M timesteps training

```bash
python scripts/train_v6_extended_with_optimizations.py \
    --config config/training_v6_extended_optimized.yaml \
    --output models/v6_extended_full \
    --device cuda:0
```

**Temps:** 7-14 jours (exécution continue!)

**Monitoring:**
```bash
# Terminal 1: Watch training
tail -f logs/train_v6_extended_*.log

# Terminal 2: Weights & Biases
open https://wandb.ai/YOUR_USERNAME/Ploutos

# Terminal 3: TensorBoard
tensorboard --logdir logs/tensorboard
```

---

### Jour 8-14 (Semaine 4)

**Tâche 8.1:** Feature importance analysis

```bash
python scripts/feature_importance_analysis.py \
    --model models/v6_extended_full/stage_3_final.zip \
    --n-samples 1000 \
    --output results/feature_importance.json
```

**Tâche 8.2:** Walk-forward validation

```bash
python scripts/walk_forward_validator.py \
    --data data/historical_daily.csv \
    --train-window 252 \
    --test-window 63 \
    --output results/walk_forward_results.json
```

**Tâche 8.3:** Deploy to VPS for paper trading

```bash
# Copy to VPS
rsync -avz models/v6_extended_full/ root@VPS:/root/ploutos/models/v6_extended_full/

# Start paper trading
sudo systemctl restart ploutos-trader-v2
```

---

## 📊 MILESTONES & SUCCESS METRICS

| Phase | Milestone | Target | Status |
|-------|-----------|--------|--------|
| 1 | Obs Space Rework | +25% convergence | ⏳ TODO |
| 1 | Reward Rework | +30% Sharpe | ⏳ TODO |
| 1 | 5M Test | Sharpe > 0.8 | ⏳ TODO |
| 2 | Normalizer | +20% performance | ⏳ TODO |
| 2 | Walk-Forward | No overfitting | ⏳ TODO |
| 3 | Full 50M | Sharpe > 1.6 | ⏳ TODO |
| 3 | Paper Trading | 2 weeks clean | ⏳ TODO |
| 3 | Production | Live Trading ✅ | ⏳ TODO |

---

## 🎯 SUCCESS CRITERIA

✅ **Phase 1:** Training converges (5M steps, no crashes)  
✅ **Phase 2:** Sharpe > 1.0 (vs 0.8 before)  
✅ **Phase 3:** Sharpe > 1.6, Consistent results  
✅ **Production:** 2 weeks clean paper trading, no drift alerts  

---

## ⚠️ RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA OOM | HIGH | CRITICAL | Reduce n_envs to 16 |
| NaN values | MEDIUM | HIGH | Add preprocessing checks |
| Overfitting | HIGH | HIGH | Walk-forward validation |
| Drift | MEDIUM | CRITICAL | Drift detector active |
| Slow training | MEDIUM | LOW | Run on BBC (RTX 3080) |

---

## 📚 FILES TO CREATE/MODIFY

**New Files:**
- [ ] `core/observation_builder_v7.py` (new)
- [ ] `core/reward_calculator_advanced.py` (new)

**Modified Files:**
- [ ] `core/universal_environment_v6_better_timing.py` (integrate builders)
- [ ] `scripts/train_v6_extended_with_optimizations.py` (already created)
- [ ] `config/training_v6_extended_optimized.yaml` (already created)

**Already Exist:**
- ✅ `core/normalization.py`
- ✅ `core/transformer_encoder.py`
- ✅ `core/drift_detector_advanced.py`
- ✅ `docs/OPTIMIZATION_GUIDE_V6.md`

---

## 🔗 REFERENCES

- V6 Advanced Branch: `feature/v6-advanced-optimization`
- Main Training Script: `scripts/train_v6_extended_with_optimizations.py`
- Analysis Document: See `PROJECT_ANALYSIS.md`

---

**Total Timeline:** 4 weeks  
**Expected ROI:** +150% Sharpe (0.8 → 2.0+)  
**Status:** 🟢 Ready to Start!
