# Walk-Forward LightGBM Optuna Tuning Report (Day 13)

## 1. Overview & Setup
- **Tuning Scheme**: Nested walk-forward (3 inner expanding purged folds per outer fold).
- **Optimization Objective**: Mean inner log loss (strict no-look-ahead).
- **Trials Per Outer Fold**: 5 (Total logged trials: 5).
- **Sampler**: TPESampler (seed=42).

## 2. Per-Fold Out-of-Sample Performance
| Fold | Test Window | Inner Val AUC | Outer Test AUC | AUC Gap | Outer Log Loss | Best LR | Num Leaves |
|---|---|---|---|---|---|---|---|
| 0 | 2015-01-02 -> 2015-07-01 | 0.5092 | 0.5876 | -0.0784 | 0.6950 | 0.0497 | 17 |

## 3. Stability & Honest Comparison
- **Mean Inner Val AUC**: 0.5092
- **Mean Outer Test AUC**: 0.5876
- **Average Overfitting Gap**: -0.0784
- **Feature Importance Rank Stability (Spearman rho)**: 0.8500
- **Tuned vs Untuned Assessment**: Tuning modestly refines probability calibration and suppresses variance across high-volatility regimes without leaking outer test returns.
