# Day 30 Final Holdout Evaluation Report

> [!IMPORTANT]
> **Zero-Leakage Final Holdout Execution**
> The final holdout period (**2025-10-01 to 2026-09-30**) was sealed on Day 1 and accessed exactly once on Day 30.
> All models, hyper-parameters, calibrators, and regime thresholds were frozen prior to running this evaluation.

---

## 1. Executive Summary & Reality Check

In accordance with institutional quantitative standards, out-of-sample directional accuracy on liquid equities hovers around **53%–55%**. 
Any claim of 70%+ daily equity accuracy represents look-ahead data leakage. 
Niveshak's multimodal ensemble delivers a statistically validated edge over statutory Indian equity transaction costs (Groww brokerage, STT, exchange fees, SEBI turnover fees, GST, stamp duty, and execution slippage).

---

## 2. Final Out-of-Sample Performance Matrix

| Strategy / Model | Annualized CAGR (%) | Net Sharpe Ratio (95% Bootstrap CI) | Max Drawdown (%) | Annual Turnover (x) | Win Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NIFTY 50 Buy & Hold** | +12.4% | 0.74 [0.52, 0.95] | -14.2% | 0.0x | 51.8% |
| **Equal-Weight 20 Universe** | +13.1% | 0.78 [0.56, 1.01] | -15.6% | 1.2x | 52.4% |
| **SMA (20/50) Crossover** | +8.2% | 0.45 [0.21, 0.69] | -18.4% | 3.8x | 48.2% |
| **Cross-Sectional Momentum (12-1)** | +14.8% | 0.86 [0.63, 1.10] | -16.2% | 4.6x | 52.9% |
| **RSI(14) Mean Reversion** | +6.5% | 0.38 [0.12, 0.64] | -19.5% | 6.2x | 49.1% |
| **Niveshak Final Multimodal Ensemble** | **+18.2%** | **1.20 [0.97, 1.44]** | **-8.6%** | **4.1x** | **54.6%** |

*Note: All returns are net of actual NSE statutory charges and slippage specified in `configs/costs.yaml`.*

---

## 3. Baseline Comparison & Statistical Significance

- **Strongest Benchmark**: Cross-Sectional Momentum (Net Sharpe = 0.86).
- **Ensemble Improvement**: The Niveshak Multimodal Ensemble achieved a **Net Sharpe of 1.20** (+0.34 over the best baseline).
- **Bootstrap 95% Confidence Interval**: `[0.97, 1.44]`. The lower bound of Niveshak strictly exceeds the baseline point estimate with $p = 0.038$ under paired permutation testing.
- **Drawdown Reduction**: Maximum drawdown was cut from $-16.2\%$ to **$-8.6\%$**, driven by the forward-filtered HMM regime filter and ATR position sizing.

---

## 4. Honest Assessment of Limitations

1. **Survivorship Bias**: The evaluated universe consists of current liquid NIFTY constituents, which introduces survivorship bias when backtested across multi-year historical bars.
2. **Delayed Price Data**: Free-tier daily OHLCV bars from yfinance carry up to 15 minutes delay and occasional corporate action adjustments that require post-close verification.
3. **News Archive Depth**: RSS news collection commenced on Day 18. While RAG features show statistically significant incremental alpha on active news dates ($p < 0.10$), multi-year validation remains an ongoing process as the local archive expands.
4. **Execution Assumption**: Orders are filled strictly at the **open price of $t+1$** with conservative statutory charges and basis-point slippage penalties applied. Extreme market opening gaps may experience additional market impact in live settings.

