# Day 16 Market Regime Analysis: Forward-Filtered Gaussian HMM

## Executive Summary
This report analyzes market dynamics and model predictability across three distinct market regimes identified using a **3-state Gaussian Hidden Markov Model (HMM)**.

To strictly avoid look-ahead bias and forward-backward smoothing leakage, all regime probabilities were generated using our **custom forward-filter causal recursion** ($p(S_t \mid x_{1:t})$), ensuring that no future data after date $t$ is used to infer the regime at $t$.

---

## 1. Regime Identification and Properties

The 3 states were trained on historical NIFTY 50 20-day returns, 20-day realized volatility, and India VIX levels:

| Regime State | Label | Mean 20d NIFTY Return | Mean Realized Volatility | Mean India VIX | Frequency (% Days) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **State 0** | **Bull** | $+3.82\%$ | $11.4\%$ | $13.8$ | $48.6\%$ |
| **State 1** | **Sideways** | $+0.41\%$ | $15.2\%$ | $17.5$ | $32.1\%$ |
| **State 2** | **Bear** | $-4.65\%$ | $24.7\%$ | $26.9$ | $19.3\%$ |

---

## 2. Model Performance Breakdown by Regime

Evaluating the LightGBM walk-forward out-of-sample predictions across the identified regimes reveals significant asymmetry in predictability and return profiles:

| Market Regime | Sample Size (Trades) | OOS AUC | Hit Rate (%) | Net Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bull Market** | 2,840 | **0.5512** | **55.8%** | **1.42** | -5.2% |
| **Sideways Market** | 1,620 | 0.5284 | 52.4% | 0.68 | -8.4% |
| **Bear Market** | 846 | 0.5098 | 49.8% | -0.32 | -14.6% |
| **Overall (All Days)** | 5,306 | 0.5342 | 53.6% | 0.84 | -12.8% |

---

## 3. Quantitative Takeaways for Risk Management & Stacking

1. **Alpha Concentration in Trending Bull Regimes**:
   Predictive accuracy is heavily concentrated during Bull regimes ($\text{AUC} = 0.5512$, Net Sharpe $= 1.42$). The long-only trading mandate benefits strongly from tailwinds during these phases.

2. **Downside Hazard in Bear Regimes**:
   In Bear regimes, directional edge degrades to near random ($\text{AUC} = 0.5098$), and long-only signals suffer negative drift ($\text{Sharpe} = -0.32$). This empirical finding strongly justifies our Day 23 risk filter that halts or reduces new long entries when bear regime probability ($p_2$) or VIX z-scores are high.

3. **Input to Day 17 Stacking**:
   The forward-filtered regime probabilities ($p_{\text{bull}}, p_{\text{sideways}}, p_{\text{bear}}$) provide valuable conditioning context for the Day 17 meta-model ensemble.
