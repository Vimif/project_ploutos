# Risk Management

## Overview

The Risk Manager protects your capital by:
- Automatically calculating position sizes
- Activating circuit breakers on excessive losses
- Monitoring at-risk positions
- Generating detailed risk reports

## Key Features

### Position Sizing

Dynamic calculation based on actual risk:

```
Position Size = Risk Amount / Stop Loss Distance
```

**Example:**
- Portfolio: $100,000
- Risk per trade: 1% = $1,000
- Stop loss: 5%
- Entry price: $500

Result: 40 shares × $500 = $20,000 (20% of portfolio)

### Circuit Breaker

Automatic trading halt on excessive daily losses.

| Condition | Action |
|-----------|--------|
| Loss < 3% | Normal trading |
| Loss ≥ 3% | Close all positions, suspend trading, send alert |

### Position Monitoring

Evaluates each position by:
- Portfolio allocation (%)
- Unrealized P&L
- Holding duration

| Risk Score | Level | Action |
|------------|-------|--------|
| 0-1 | Low | Monitor |
| 2-3 | Medium | Increased surveillance |
| 4+ | Critical | Immediate closure |

### Kelly Criterion

Mathematical optimization for position sizing:

```
Kelly % = Win Rate - [(1 - Win Rate) / Win/Loss Ratio]
```

Uses half-Kelly for added safety.

## Configuration

```python
self.risk_manager = RiskManager(
    max_portfolio_risk=0.01,      # 1% per trade
    max_daily_loss=0.03,          # 3% daily limit
    max_position_size=0.05,       # 5% per position
    max_correlation=0.7           # Max correlation
)
```

## Performance Metrics

| Metric | Score | Assessment |
|--------|-------|------------|
| Sharpe Ratio | > 3 | Excellent |
| Sharpe Ratio | 2-3 | Good |
| Max Drawdown | < 10% | Excellent |
| Max Drawdown | 10-20% | Acceptable |

## Alerts

Triggered when:
- Circuit breaker activated
- 3+ high-risk positions
- Exposure > 85%
- Drawdown > 15%

## Usage

Automatic position sizing:
```python
qty = self.calculate_position_size_with_risk(
    symbol='NVDA',
    current_price=500.0,
    portfolio_value=100000
)
```

Daily risk check:
```python
self.check_risk_management()
risk_report = self.risk_manager.get_risk_report(positions, portfolio_value)
```

## Customization

Modify in `core/risk_manager.py`:
- Position sizing formulas
- Risk thresholds
- Calculated metrics
- Circuit breaker conditions
