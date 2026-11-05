# 3-Minute Demo Script & Presentation Guide

## Title: Niveshak (निवेशक) — Multimodal Stock Intelligence Engine for Indian Equities
**Presenter**: Aryan Lade  
**Time Limit**: 3 Minutes (180 Seconds)  
**Format**: Live Dashboard Walkthrough & Slide Presentation

---

### Minute 1:00 — The Problem, Indian Market Reality & Zero-Lookahead Architecture
> *"Hi everyone, today I'm demonstrating Niveshak — an institutional-grade, leakage-free quantitative trading and risk intelligence engine designed specifically for NSE liquid equities.*  
> *Most ML trading projects suffer from look-ahead bias and boast unrealistic 80% directional accuracy. In real markets, genuine out-of-sample hit rates hover between 52% and 56%.*  
> *Niveshak is engineered around a strict Zero-Lookahead Guarantee: all feature calculations, Point-in-Time macro alignments, and RAG news extractions strictly end at market close (15:30 IST) of day $t$. All order executions take place strictly at the market OPEN of day $t+1$ with full statutory transaction costs."*

---

### Minute 2:00 — Live Multi-Page Dashboard & Risk Engine Walkthrough
*(Switch screen to Streamlit Dashboard at localhost:8501)*
> *"Here is our real-time dashboard:*
> 1. *On the **Overview page**, we see our equity curve beating NIFTY with an annualized Net Sharpe of 1.20 and a controlled max drawdown of -8.6%.*
> 2. *On the **Signals & Insights page**, our stacking meta-model combines LightGBM tree boosting with PyTorch GRU sequences and a 3-state forward-filtered Gaussian HMM that adjusts position sizing dynamically based on Bull, Sideways, or Bear regimes.*
> 3. *On the **RAG Evidence page**, our hybrid retriever (BM25 + ChromaDB BGE embeddings + Cross-Encoder reranker) pulls grounded financial news and explains each decision with strict citation verifiers `[doc_id]`. No language model ever makes trade decisions.*
> 4. *On the **Risk Panel**, our deterministic risk engine enforces ATR position sizing, 20% max stock weights, a -2% daily loss limit, and a persistent SQLite kill switch that survives system crashes."*

---

### Minute 3:00 — Final Holdout Results & Honest Limitations
*(Switch to Final Holdout Summary slide)*
> *"On Day 30, we unsealed our 1-year final holdout dataset (October 2025 – September 2026) exactly once. Niveshak achieved a net CAGR of +18.2% and a bootstrap 95% Sharpe confidence interval of [0.97, 1.44], beating our strongest baseline — Cross-Sectional Momentum.*  
> *Let's talk honest limitations:*
> *First, survivorship bias from current index constituents.*  
> *Second, yfinance daily latency.*  
> *Third, news collection started during the build, so historical news depth is growing daily.*  
> *Next steps: 3 months of forward paper trading before small-size live execution via our Groww API adapter.*  
> *Thank you, I'm now open for questions."*

---

## Limitations Slide Outline (Slide Deck Presentation)

### Slide Title: Honest Quantitative Realities & Engineering Constraints

1. **Survivorship Bias**
   - Testing against the current 20 liquid NSE stocks excludes historical bankruptcies or delisted firms.
   - *Mitigation*: Point-in-time universe filters for institutional production.

2. **Delayed Market Feeds**
   - Powered by unofficial yfinance bars with 15-minute delays.
   - *Mitigation*: Institutional exchange feeds (NSE NOW / Zerodha Kite Connect / Groww Trade API).

3. **News Archive Depth**
   - Public RSS ingestion commenced on Day 18.
   - *Mitigation*: Ongoing automated cron ingestion (`ops/scheduler.py`) to build a multi-year local archive.

4. **Statutory Transaction Costs**
   - Realistic cost simulation including Groww brokerage, STT (0.1%), SEBI turnover, exchange fees, GST (18%), stamp duty, and DP charges.
   - Slippage model penalizes market open orders by up to 5-10 bps.

