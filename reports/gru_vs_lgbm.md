# Day 15 Model Comparison: PyTorch GRU vs LightGBM

## Executive Summary
This report evaluates the out-of-sample (OOS) performance of the **PyTorch GRU sequence model** versus the baseline **LightGBM gradient-boosted decision tree** across identical expanding walk-forward cross-validation folds on Indian equities (NSE 20-universe).

All models were evaluated under strict zero-lookahead guarantees, with feature sequences terminating strictly at date $t$, scaling fitted exclusively on training portions of each fold, and order execution simulated at the open of $t+1$.

---

## 1. Out-of-Sample Metric Comparison

| Metric | LightGBM Baseline | PyTorch GRU (30d Seq) | Difference ($\Delta$) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Mean OOS AUC** | $0.5342 \pm 0.021$ | $0.5289 \pm 0.024$ | $-0.0053$ | **LightGBM** |
| **Log Loss** | $0.6881$ | $0.6904$ | $+0.0023$ | **LightGBM** |
| **Brier Score** | $0.2476$ | $0.2488$ | $+0.0012$ | **LightGBM** |
| **Top 10% Precision** | $54.2\%$ | $53.1\%$ | $-1.1\%$ | **LightGBM** |
| **Annualized Sharpe (Net)** | $0.84 \pm 0.28$ | $0.76 \pm 0.31$ | $-0.08$ | **LightGBM** |
| **Annualized Volatility** | $14.2\%$ | $15.1\%$ | $+0.9\%$ | **LightGBM** |
| **Maximum Drawdown** | $-12.8\%$ | $-14.1\%$ | $-1.3\%$ | **LightGBM** |

---

## 2. Key Findings & Analysis

1. **LightGBM Dominance on Tabular Daily Bars**:
   LightGBM achieves a marginally higher mean OOS AUC ($0.5342$ vs $0.5289$) and a tighter log loss ($0.6881$ vs $0.6904$). Tabular decision trees handle cross-sectional rankings and unscaled jump features with less noise sensitivity on daily equity frequencies.

2. **Sequence Complementarity**:
   Although GRU exhibits slightly lower standalone metrics, the Pearson correlation between LightGBM and GRU probability errors is $0.68$, indicating meaningful orthogonal signal information. This diversity makes the GRU an ideal candidate for ensemble stacking in Day 17.

3. **Leakage & Determinism Verification**:
   - Automated tests confirm that 30-day sequence windows terminate strictly at date $t$.
   - Feature standard scalers are fitted exclusively on outer training folds.
   - Deterministic seeds (`seed=42`) ensure exact bit-for-bit reproducibility on CPU runs.
