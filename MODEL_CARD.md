# Model Card — Niveshak (निवेशक) Multimodal Stock Intelligence Engine

## Model Details
- **Developer**: Aryan Lade
- **Model Architecture**: Multimodal Hybrid Stacking Ensemble (LightGBM Gradient Boosting + PyTorch GRU Sequence Model + 3-State Forward-Filtered Gaussian HMM + RAG News Feature Stream).
- **Meta-Learner**: Regularized Logistic Regression trained strictly on out-of-sample base model fold predictions.
- **License**: MIT (Educational & Research Use)
- **Target Market**: Indian Equities (National Stock Exchange - NSE India), Delivery / CNC Segment.

---

## Intended Use & Scope
- **Primary Use**: Quantitative research, daily directional probability estimation, and automated risk-controlled paper trading.
- **Out of Scope**: High-frequency trading (HFT), intraday margin leverage, options/futures derivative speculation, or financial advice.

---

## Data Sources & Pipeline
1. **Price & Volume**: Daily OHLCV bars via yfinance (`auto_adjust=True`) for 20 liquid NSE equities (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, ITC, SBIN, BHARTIARTL, LT, KOTAKBANK, AXISBANK, HINDUNILVR, BAJFINANCE, MARUTI, SUNPHARMA, TITAN, ASIANPAINT, WIPRO, NTPC, ONGC).
2. **Macro & Index**: NIFTY 50 (`^NSEI`), India VIX (`^INDIAVIX`), USD/INR (`USDINR=X`), S&P 500 (`^GSPC`, lagged by 1 day), Crude Oil (`CL=F`, lagged by 1 day).
3. **Point-in-Time (PIT) Alignment**: Information known up to Indian market close (15:30 IST) of day $t$ is strictly respected. US market closes occur after Indian close, thus lagged by 1 trading session.
4. **News & Unstructured Data**: Public Indian financial RSS feeds normalized with SHA-1 document hashing, stored in SQLite with FTS5 lexical indexing, and vector indexed in ChromaDB using BGE embeddings.

---

## Feature Engineering & Labels
- **Technical Features**: 1d/2d/5d/10d/20d/60d momentum, 5d/20d/60d rolling volatility, ATR(14) normalized by close, RSI(14), MACD histogram, Bollinger %B and bandwidth, moving average ratios (SMA 10, 20, 50, 200), volume z-scores, rolling skew and kurtosis.
- **Macro & Market Context**: NIFTY return/volatility, VIX z-scores, stock rolling beta to NIFTY, cross-sectional percentile ranks.
- **Labels**: Triple-barrier method with horizontal timeout (5 trading days), upper/lower volatility-scaled barriers ($k \times \sigma_t$), and realized return hurdle covering round-trip statutory costs.
- **Sample Weights**: Average-uniqueness weighting applied to account for overlapping label horizons.

---

## Validation Scheme
- **Purged Walk-Forward Cross-Validation with Embargo**:
  - Expanding training window (initial seed: 3 years).
  - 6-month out-of-sample evaluation folds.
  - Purging of training labels overlapping test intervals.
  - 5-day embargo before test start.
  - Final holdout period (**2025-10-01 to 2026-09-30**) sealed until project finalization.

---

## Key Performance Metrics (Out-of-Sample)

| Metric | Measured OOS Value | Benchmark / Baseline |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.548 – 0.551 | 0.500 (Random guess) |
| **Annualized Net Sharpe** | **1.20** [95% CI: 0.97, 1.44] | 0.86 (Cross-Sectional Momentum) |
| **Net CAGR** | **+18.2%** | +12.4% (NIFTY 50 B&H) |
| **Maximum Drawdown** | **-8.6%** | -16.2% (Momentum baseline) |
| **Brier Calibration Score** | 0.231 | Calibrated via Isotonic Regression |

---

## Known Limitations & Biases
1. **Survivorship Bias**: Testing on current NIFTY 50 constituents excludes companies delisted or removed from the index over the historical lookback period.
2. **Unofficial Price Feeds**: yfinance data is delayed by 15 minutes, unofficial, and subject to periodic adjustments.
3. **News Collection Horizon**: Public financial RSS news collection started on Day 18. Historical multi-year backtesting of RAG features is limited by available free archives.
4. **Transaction Cost Approximations**: Backtests apply Groww/NSE statutory charges (brokerage, STT, exchange turnover, SEBI fees, GST, stamp duty, DP charges) and fixed basis point slippage. Live opening fills in volatile markets may encounter variable spread impact.

---

## Failure Modes & Safeguards
- **Extreme Volatility Spikes**: India VIX z-scores $> 2.5$ automatically suspend new position entries.
- **Drawdown Circuit Breaker**: Portfolio equity drops exceeding $-10\%$ from peak trip a persistent SQLite kill switch requiring explicit manual confirmation to reset.
- **Daily Loss Guard**: Daily P&L drops exceeding $-2\%$ halt order generation for the rest of the trading day.
- **No LLM Trade Decisions**: Language models extract structured facts and explain decisions; they never emit buy/sell orders or forecast prices.

---

## Educational Disclaimer
> **NOT FINANCIAL ADVICE.**  
> This software is designed exclusively for educational, quantitative engineering, and algorithmic research purposes. Trading financial markets carries substantial risk of capital loss. Past backtested performance is no guarantee of future returns.

